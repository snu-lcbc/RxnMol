"""
Decoder-only Transformer (LLM-style) for reaction modeling.

Features:
- Rotary Position Embeddings (RoPE)
- RMSNorm
- Grouped Query Attention (GQA)
- SwiGLU activation
- Flash Attention via PyTorch SDPA
- KV-cache for efficient generation

This module is standalone with no dependencies on transformer.py.
"""

from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Core Components (Standalone)
# =============================================================================


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Faster than LayerNorm as it doesn't compute mean.
    x_norm = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position by rotating query/key vectors, allowing relative
    position to emerge naturally from dot products.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin values for positions."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin for sequence length.

        Returns:
            cos: (seq_len, dim)
            sin: (seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        target_dtype = dtype or self.cos_cached.dtype
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=target_dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=target_dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key.

    Args:
        q: (batch, n_heads, seq_len, head_dim)
        k: (batch, n_kv_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        Rotated q, k with same shapes
    """
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU(x) = (x @ W_gate * SiLU(x @ W_up)) @ W_down

    Uses ~8/3 * d_model hidden dim to match parameter count of standard 4 * d_model FFN.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU = x * sigmoid(x) = Swish
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class CausalAttention(nn.Module):
    """
    Causal Self-Attention with GQA and RoPE for decoder-only models.

    Uses PyTorch's scaled_dot_product_attention for Flash Attention support.
    Always returns KV-cache for efficient generation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_kv_groups = n_heads // n_kv_heads

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input (batch, seq_len, d_model)
            mask: Attention mask
            cos, sin: RoPE embeddings
            kv_cache: Cached (key, value) for incremental decoding
            is_causal: Apply causal mask (for full-sequence forward)

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated cache (always returned)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: (batch, seq, n_heads * head_dim) -> (batch, n_heads, seq, head_dim)
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache - always update and return
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Always return new cache
        new_kv_cache = (k, v)

        # Expand KV heads for GQA
        if self.n_kv_groups > 1:
            k_expanded = k.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
            k_expanded = k_expanded.reshape(batch_size, self.n_heads, -1, self.head_dim)
            v_expanded = v.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
            v_expanded = v_expanded.reshape(batch_size, self.n_heads, -1, self.head_dim)
        else:
            k_expanded = k
            v_expanded = v

        # Flash Attention via PyTorch SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal and mask is None,
        )

        # Reshape back: (batch, n_heads, seq, head_dim) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_kv_cache


class CausalBlock(nn.Module):
    """Decoder-only block with self-attention and FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = CausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x = self.norm1(x)
        x, new_cache = self.self_attn(
            x,
            mask=attn_mask,
            cos=cos,
            sin=sin,
            kv_cache=kv_cache,
            is_causal=is_causal,
        )
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual

        return x, new_cache


# =============================================================================
# Main Model
# =============================================================================


class CausalTransformer(nn.Module):
    """
    Decoder-only Transformer for autoregressive reaction modeling.

    Input: tokenized sequence that includes reactants and products with delimiters.
    Output: logits over vocabulary for next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        # Config validation
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert (d_model // n_heads) % 2 == 0, f"head_dim ({d_model // n_heads}) must be even for RoPE"
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

        # Store config as individual attributes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_p = dropout
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.rope_base = rope_base

        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_scale = math.sqrt(d_model)

        self.layers = nn.ModuleList([
            CausalBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.rope = RotaryEmbedding(
            d_model // n_heads,
            max_seq_len,
            rope_base,
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, total_len) 1 for valid, 0 for pad
            kv_cache: per-layer KV cache for incremental decoding

        Returns:
            logits: (batch, seq_len, vocab_size)
            new_kv_cache: updated cache (always returned)
        """
        x = self.embed(input_ids) * self.embed_scale
        x = self.embed_dropout(x)
        seq_len = x.shape[1]

        # Get RoPE embeddings for correct positions
        if kv_cache is not None and 0 in kv_cache:
            start_pos = kv_cache[0][0].shape[2]
            cos, sin = self.rope(start_pos + seq_len, x.device, x.dtype)
            # FIX: Slice to exact range needed
            cos = cos[start_pos:start_pos + seq_len]
            sin = sin[start_pos:start_pos + seq_len]
            is_causal = False  # Using cache, no causal mask needed
        else:
            cos, sin = self.rope(seq_len, x.device, x.dtype)
            is_causal = True  # Full sequence, need causal mask

        # Build attention mask
        attn_mask = None
        if not is_causal:
            # With cache: only need padding mask if provided
            if attention_mask is not None:
                # attention_mask is for full sequence including cache
                attn_mask = (~attention_mask).unsqueeze(1).unsqueeze(2).to(x.dtype) * torch.finfo(x.dtype).min
        else:
            # Without cache: build causal + padding mask
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    torch.finfo(x.dtype).min,
                    device=x.device,
                    dtype=x.dtype,
                ),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                pad_mask = (~attention_mask).unsqueeze(1).unsqueeze(2).to(x.dtype) * torch.finfo(x.dtype).min
                attn_mask = causal_mask + pad_mask
            else:
                attn_mask = causal_mask

        # Forward through layers
        new_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache.get(i) if kv_cache else None
            x, layer_new_cache = layer(
                x,
                attn_mask=attn_mask,
                cos=cos,
                sin=sin,
                kv_cache=layer_cache,
                is_causal=is_causal and attn_mask is None,
            )
            new_cache[i] = layer_new_cache

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV-cache.

        Args:
            input_ids: prompt tokens (batch, seq_len)
            attention_mask: optional (batch, seq_len) mask for prompt
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (0 for greedy)
            top_k: top-k sampling
            top_p: nucleus sampling
            eos_id: end-of-sequence token ID
        """
        device = input_ids.device
        eos_id = eos_id if eos_id is not None else self.eos_id

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id)
        attention_mask = attention_mask.to(dtype=torch.bool, device=device)

        generated = input_ids
        kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if kv_cache:
                step_input = generated[:, -1:]
            else:
                step_input = generated

            logits, kv_cache = self.forward(
                step_input,
                attention_mask=attention_mask,
                kv_cache=kv_cache if kv_cache else None,
            )

            # FIX: Handle temperature=0 for greedy decoding
            if temperature > 0:
                next_logits = logits[:, -1, :] / temperature
            else:
                next_logits = logits[:, -1, :]

            if top_k is not None and top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if temperature > 0 and (top_k is None or top_k > 1) and (top_p is None or top_p < 1.0):
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, dtype=torch.bool, device=device)],
                dim=1,
            )

            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return generated

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate_with_instruction(
        self,
        reactants_ids: torch.Tensor,
        instruction_ids: torch.Tensor,
        sep_id: int,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate products with instruction prefix.

        Constructs prompt: <inst> instruction </inst> reactants <sep>
        Then generates: products </s>

        Args:
            reactants_ids: Tokenized reactants (batch, reactants_len)
            instruction_ids: Tokenized instruction with <inst></inst> (batch, inst_len)
            sep_id: Separator token ID
            attention_mask: Optional mask for the prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_id: End of sequence token ID

        Returns:
            Generated sequence including prompt (batch, total_len)
        """
        device = reactants_ids.device
        batch_size = reactants_ids.shape[0]

        # Build prompt: instruction + reactants + sep
        sep_token = torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device)
        prompt = torch.cat([instruction_ids, reactants_ids, sep_token], dim=1)

        # Generate products
        return self.generate(
            prompt,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
        )

    @torch.inference_mode()
    def parallel_beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        eos_id: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized beam search for decoder-only model.

        Args:
            input_ids: Prompt token IDs (batch, prompt_len)
            attention_mask: Optional attention mask for prompt
            max_new_tokens: Maximum tokens to generate
            beam_size: Number of beams
            length_penalty: Length penalty (>1 favors longer)
            early_stopping: Stop when beam_size hypotheses complete
            eos_id: End of sequence token ID

        Returns:
            (best_sequences, scores): (batch, total_len), (batch,)
            If return_all=True, returns (best_sequences, scores, all_sequences, all_scores)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        eos_id = eos_id or self.eos_id
        vocab_size = self.vocab_size
        prompt_len = input_ids.shape[1]

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id)
        attention_mask = attention_mask.to(dtype=torch.bool, device=device)

        # Expand for beams: (batch, len) -> (batch * beam, len)
        input_ids = input_ids.unsqueeze(1).repeat(1, beam_size, 1)
        input_ids = input_ids.view(batch_size * beam_size, -1)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, beam_size, 1)
        attention_mask = attention_mask.view(batch_size * beam_size, -1)

        # Initialize beam scores
        beam_scores = torch.zeros(batch_size * beam_size, device=device)
        beam_scores[~torch.arange(batch_size * beam_size, device=device).fmod(beam_size).eq(0)] = -1e9

        generated = input_ids
        # FIX: Use accumulated finished mask
        finished_mask = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)
        finished_scores = torch.full((batch_size, beam_size), float('-inf'), device=device)
        finished_seqs = torch.zeros(batch_size, beam_size, prompt_len + max_new_tokens, dtype=torch.long, device=device)
        num_finished = torch.zeros(batch_size, dtype=torch.long, device=device)

        kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        for step in range(max_new_tokens):
            if kv_cache:
                step_input = generated[:, -1:]
            else:
                step_input = generated

            logits, kv_cache = self.forward(step_input, attention_mask=attention_mask, kv_cache=kv_cache if kv_cache else None)
            next_logits = F.log_softmax(logits[:, -1, :], dim=-1)

            # Add beam scores
            next_scores = next_logits + beam_scores.unsqueeze(-1)
            next_scores[finished_mask] = -1e9
            next_scores[finished_mask, self.pad_id] = beam_scores[finished_mask]

            # Reshape: (batch, beam * vocab)
            next_scores = next_scores.view(batch_size, beam_size * vocab_size)

            # Top-k per batch
            topk_scores, topk_indices = torch.topk(
                next_scores, k=2 * beam_size, dim=-1, largest=True, sorted=True
            )

            topk_beam_idx = topk_indices // vocab_size
            topk_token_idx = topk_indices % vocab_size

            new_beam_scores = torch.zeros(batch_size, beam_size, device=device)
            new_tokens = torch.zeros(batch_size, beam_size, dtype=torch.long, device=device)
            beam_mapping = torch.zeros(batch_size, beam_size, dtype=torch.long, device=device)

            for b in range(batch_size):
                selected = 0
                for k in range(2 * beam_size):
                    if selected >= beam_size:
                        break

                    score = topk_scores[b, k]
                    beam_idx = topk_beam_idx[b, k]
                    token_idx = topk_token_idx[b, k]
                    global_beam = b * beam_size + beam_idx

                    if token_idx.item() == eos_id:
                        if num_finished[b] < beam_size:
                            seq_len = step + 1
                            final_score = score / (seq_len ** length_penalty)
                            idx = num_finished[b].item()
                            finished_scores[b, idx] = final_score
                            # FIX: Store sequence including EOS
                            finished_seqs[b, idx, :generated.shape[1]] = generated[global_beam]
                            finished_seqs[b, idx, generated.shape[1]] = eos_id
                            num_finished[b] += 1

                        if early_stopping and num_finished[b] >= beam_size:
                            finished_mask[b * beam_size:(b + 1) * beam_size] = True
                        continue

                    new_beam_scores[b, selected] = score
                    new_tokens[b, selected] = token_idx
                    beam_mapping[b, selected] = global_beam
                    selected += 1

                # FIX: Use -inf for remaining slots
                while selected < beam_size:
                    new_beam_scores[b, selected] = float('-inf')
                    new_tokens[b, selected] = self.pad_id
                    beam_mapping[b, selected] = b * beam_size
                    selected += 1

            if finished_mask.all():
                break

            beam_mapping_flat = beam_mapping.view(-1)
            finished_mask = finished_mask[beam_mapping_flat]
            generated = generated[beam_mapping_flat]
            generated = torch.cat([generated, new_tokens.view(-1, 1)], dim=1)
            attention_mask = torch.cat(
                [attention_mask[beam_mapping_flat],
                 torch.ones(batch_size * beam_size, 1, dtype=torch.bool, device=device)],
                dim=1
            )

            beam_scores = new_beam_scores.view(-1)

            if kv_cache:
                new_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
                for layer_idx, (k, v) in kv_cache.items():
                    new_cache[layer_idx] = (
                        k.index_select(0, beam_mapping_flat),
                        v.index_select(0, beam_mapping_flat),
                    )
                kv_cache = new_cache

            # FIX: Accumulate finished mask instead of overwriting
            finished_mask = finished_mask | (new_tokens.view(-1) == self.pad_id)

        # Select best per batch
        best_sequences = []
        best_scores_list = []

        for b in range(batch_size):
            if num_finished[b] > 0:
                best_idx = finished_scores[b, :num_finished[b]].argmax()
                best_seq = finished_seqs[b, best_idx]
                best_score = finished_scores[b, best_idx]
            else:
                best_beam = b * beam_size
                best_seq = generated[best_beam]
                best_score = beam_scores[best_beam] / (generated.shape[1] ** length_penalty)

            best_sequences.append(best_seq)
            best_scores_list.append(best_score)

        max_seq_len = max(len(s) for s in best_sequences)
        padded_sequences = torch.full(
            (batch_size, max_seq_len), self.pad_id, dtype=torch.long, device=device
        )
        for i, seq in enumerate(best_sequences):
            seq_len = min(len(seq), max_seq_len)
            padded_sequences[i, :seq_len] = seq[:seq_len]

        if not return_all:
            return padded_sequences, torch.stack(best_scores_list)

        generated_beams = generated.view(batch_size, beam_size, -1)
        beam_scores_view = beam_scores.view(batch_size, beam_size)
        all_sequences = []
        all_scores = []
        for b in range(batch_size):
            if num_finished[b] > 0:
                scores = finished_scores[b, :num_finished[b]]
                seqs = finished_seqs[b, :num_finished[b], :]
            else:
                scores = beam_scores_view[b]
                seqs = generated_beams[b]

            order = torch.argsort(scores, descending=True)
            seqs = seqs.index_select(0, order)
            scores = scores.index_select(0, order)

            all_sequences.append([seqs[i] for i in range(seqs.shape[0])])
            all_scores.append(scores)

        return padded_sequences, torch.stack(best_scores_list), all_sequences, all_scores

    @classmethod
    def create_draft_model(
        cls,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        rope_base: float = 10000.0,
        share_embeddings: bool = True,
        parent_model: Optional["CausalTransformer"] = None,
    ) -> "CausalTransformer":
        """
        Create a smaller draft model for speculative decoding.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_kv_heads: Number of key-value heads (for GQA)
            n_layers: Number of layers (default 2 for draft model)
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            pad_id: Padding token ID
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
            rope_base: RoPE base frequency
            share_embeddings: Whether to share embedding weights
            parent_model: Parent model to share embeddings with

        Returns:
            Smaller CausalTransformer for drafting
        """
        draft_model = cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=0.0,  # No dropout for draft model
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            rope_base=rope_base,
        )

        if share_embeddings and parent_model is not None:
            draft_model.embed.weight = parent_model.embed.weight
            draft_model.lm_head.weight = parent_model.lm_head.weight

        return draft_model

    @torch.inference_mode()
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        draft_model: Optional["CausalTransformer"] = None,
        gamma: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Speculative decoding for faster generation.

        Uses a smaller draft model to propose gamma tokens, then verifies
        them with the target model in a single forward pass.

        Args:
            input_ids: Prompt token IDs
            attention_mask: Optional attention mask
            draft_model: Smaller model for speculation
            gamma: Tokens to speculate per step
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            eos_id: End of sequence token ID

        Returns:
            Generated sequence
        """
        if draft_model is None:
            return self.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_id=eos_id,
            )

        batch_size = input_ids.shape[0]
        device = input_ids.device
        eos_id = eos_id or self.eos_id
        max_total_len = input_ids.shape[1] + max_new_tokens

        # FIX: Handle temperature=0
        temp = max(temperature, 1e-10)

        generated = input_ids.clone()
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).to(torch.bool)
        else:
            attention_mask = attention_mask.clone()

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        target_kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        draft_kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        while generated.shape[1] < max_total_len:
            if finished.all():
                break

            # Draft generates gamma tokens
            draft_tokens = []
            draft_probs = []
            current = generated.clone()
            current_mask = attention_mask.clone()

            for _ in range(gamma):
                if draft_kv_cache:
                    draft_input = current[:, -1:]
                else:
                    draft_input = current

                draft_logits, draft_kv_cache = draft_model.forward(
                    draft_input, attention_mask=current_mask, kv_cache=draft_kv_cache if draft_kv_cache else None
                )
                draft_next_logits = draft_logits[:, -1, :] / temp
                draft_next_probs = F.softmax(draft_next_logits, dim=-1)

                if temperature > 0:
                    next_token = torch.multinomial(draft_next_probs, num_samples=1)
                else:
                    next_token = draft_next_logits.argmax(dim=-1, keepdim=True)

                draft_tokens.append(next_token)
                draft_probs.append(draft_next_probs)
                current = torch.cat([current, next_token], dim=1)
                current_mask = torch.cat(
                    [current_mask, torch.ones_like(next_token, dtype=torch.bool)], dim=1
                )

            draft_tokens = torch.cat(draft_tokens, dim=1)

            # Target verifies all tokens at once
            # FIX: Don't use cache for verification to get all positions
            verify_input = torch.cat([generated, draft_tokens], dim=1)
            verify_mask = torch.cat([attention_mask, torch.ones_like(draft_tokens, dtype=torch.bool)], dim=1)

            # Forward without cache to get all logits we need
            target_logits, _ = self.forward(verify_input, attention_mask=verify_mask, kv_cache=None)

            # Get probabilities for verification positions
            # We need positions [len(generated)-1, ..., len(generated)+gamma-1]
            start_idx = generated.shape[1] - 1
            target_probs = F.softmax(target_logits[:, start_idx:start_idx + gamma + 1, :] / temp, dim=-1)

            # Verify and accept
            accepted = torch.ones(batch_size, dtype=torch.long, device=device) * gamma

            for i in range(gamma):
                draft_p = draft_probs[i]
                target_p = target_probs[:, i, :]
                token = draft_tokens[:, i:i+1]

                draft_token_p = draft_p.gather(1, token).squeeze(-1)
                target_token_p = target_p.gather(1, token).squeeze(-1)
                accept_prob = torch.clamp(target_token_p / (draft_token_p + 1e-10), max=1.0)

                uniform = torch.rand(batch_size, device=device)
                reject_mask = uniform > accept_prob
                not_accepted_yet = accepted == gamma
                first_reject = reject_mask & not_accepted_yet
                accepted[first_reject] = i

            # Build sequences
            new_tokens_list = []
            for b in range(batch_size):
                n_accepted = accepted[b].item()

                if n_accepted == gamma:
                    final_logits = target_probs[b, -1, :]
                    if temperature > 0:
                        final_token = torch.multinomial(final_logits.unsqueeze(0), num_samples=1)
                    else:
                        final_token = final_logits.unsqueeze(0).argmax(dim=-1, keepdim=True)
                    accepted_tokens = torch.cat([draft_tokens[b:b+1, :], final_token], dim=1)
                else:
                    draft_p = draft_probs[n_accepted][b]
                    target_p = target_probs[b, n_accepted, :]
                    adjusted = torch.clamp(target_p - draft_p, min=0)
                    adjusted = adjusted / (adjusted.sum() + 1e-10)

                    if temperature > 0 and adjusted.sum() > 0:
                        new_token = torch.multinomial(adjusted.unsqueeze(0), num_samples=1)
                    else:
                        new_token = target_p.unsqueeze(0).argmax(dim=-1, keepdim=True)

                    accepted_tokens = torch.cat([draft_tokens[b:b+1, :n_accepted], new_token], dim=1)

                new_tokens_list.append(accepted_tokens)

            max_new = max(t.shape[1] for t in new_tokens_list)
            new_tokens_padded = torch.full(
                (batch_size, max_new), self.pad_id, dtype=torch.long, device=device
            )
            for b, t in enumerate(new_tokens_list):
                new_tokens_padded[b, :t.shape[1]] = t.squeeze(0)

            generated = torch.cat([generated, new_tokens_padded], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, max_new, dtype=torch.bool, device=device)],
                dim=1
            )

            if generated.shape[1] >= max_total_len:
                generated = generated[:, :max_total_len]
                attention_mask = attention_mask[:, :max_total_len]
                break

            eos_positions = (generated == eos_id).any(dim=1)
            finished = finished | eos_positions

            # Reset caches for next iteration (simpler than trying to maintain them)
            target_kv_cache = {}
            draft_kv_cache = {}

        return generated

    def compile_for_inference(
        self,
        mode: str = "reduce-overhead",
        backend: str = "inductor",
        dynamic: bool = True,
    ) -> "CausalTransformer":
        """
        Apply torch.compile() for faster inference.

        Args:
            mode: Compilation mode
            backend: Compilation backend
            dynamic: Support dynamic shapes

        Returns:
            Self with compiled layers
        """
        for i, layer in enumerate(self.layers):
            self.layers[i] = torch.compile(layer, mode=mode, backend=backend, dynamic=dynamic)

        return self
