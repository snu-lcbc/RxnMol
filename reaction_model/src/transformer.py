"""
Modern Transformer for Reaction Prediction.

Features:
- Rotary Position Embeddings (RoPE)
- RMSNorm (faster than LayerNorm)
- Grouped Query Attention (GQA)
- SwiGLU activation
- Flash Attention via PyTorch SDPA
- KV-cache for efficient inference
"""

import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Keep cache on the same device as inv_freq to avoid CPU/GPU mismatches.
        positions = torch.arange(
            seq_len,
            dtype=torch.float32,
            device=self.inv_freq.device,
        )
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


# =============================================================================
# Grouped Query Attention (GQA)
# =============================================================================


class Attention(nn.Module):
    """
    Grouped Query Attention with RoPE and Flash Attention.

    GQA uses fewer key-value heads than query heads, reducing KV-cache memory.
    Uses PyTorch's scaled_dot_product_attention for Flash Attention support.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_kv_groups = n_heads // n_kv_heads
        self.is_cross_attention = is_cross_attention

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Query input (batch, seq_len, d_model)
            kv: Key/Value input for cross-attention (batch, kv_len, d_model)
            mask: Attention mask
            cos, sin: RoPE embeddings
            kv_cache: Cached (key, value) for incremental decoding
            is_causal: Apply causal mask (for self-attention in decoder)

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated cache if kv_cache was provided
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x)

        if self.is_cross_attention:
            if kv is not None:
                # Compute K/V from encoder output
                k = self.k_proj(kv)
                v = self.v_proj(kv)
            elif kv_cache is not None:
                # Use cached K/V for cross-attention (encoder K/V is static)
                k = None
                v = None
            else:
                raise ValueError("Cross-attention requires either kv or kv_cache")
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)

        # Reshape Q
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Handle K/V based on whether we're using cache or computing fresh
        if k is not None and v is not None:
            # Reshape: (batch, seq, n_heads * head_dim) -> (batch, n_heads, seq, head_dim)
            k = k.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE (only for self-attention)
            if cos is not None and sin is not None and not self.is_cross_attention:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Handle KV cache - concatenate with cached values for self-attention
            if kv_cache is not None and not self.is_cross_attention:
                cached_k, cached_v = kv_cache
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)

            # Store for cache
            new_kv_cache = (k, v)
        else:
            # Cross-attention with cached encoder K/V (k, v are None)
            # Just use the cache directly without concatenation
            assert kv_cache is not None, "Expected kv_cache when k, v are None"
            k, v = kv_cache
            new_kv_cache = (k, v)  # Return same cache

        # Expand KV heads for GQA: (batch, n_kv_heads, seq, head_dim) -> (batch, n_heads, seq, head_dim)
        if self.n_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
            k = k.reshape(batch_size, self.n_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_kv_groups, -1, -1)
            v = v.reshape(batch_size, self.n_heads, -1, self.head_dim)

        # Flash Attention via PyTorch SDPA
        # Note: is_causal=True automatically creates efficient causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
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


# =============================================================================
# Transformer Blocks
# =============================================================================


class EncoderBlock(nn.Module):
    """Encoder block with self-attention and FFN."""

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
        self.self_attn = Attention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, mask=mask, cos=cos, sin=sin)
        x = self.dropout(x) + residual

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual

        return x


class DecoderBlock(nn.Module):
    """Decoder block with self-attention, cross-attention, and FFN."""

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
        self.self_attn = Attention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = Attention(d_model, n_heads, n_kv_heads, dropout, is_cross_attention=True)
        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.

        Args:
            x: Decoder input
            encoder_out: Encoder output
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Encoder padding mask for cross-attention
            cos, sin: RoPE embeddings
            kv_cache: Dict with "self" and "cross" keys for cached KV

        Returns:
            output, updated_kv_cache
        """
        new_kv_cache = {}

        # Self-attention (causal)
        residual = x
        x = self.norm1(x)
        self_cache = kv_cache.get("self") if kv_cache else None
        x, new_self_cache = self.self_attn(
            x,
            mask=self_attn_mask,
            cos=cos,
            sin=sin,
            kv_cache=self_cache,
            is_causal=not kv_cache,  # Only use causal mask when not using cache
        )
        x = self.dropout(x) + residual
        if new_self_cache is not None:
            new_kv_cache["self"] = new_self_cache

        # Cross-attention
        # For cross-attention, encoder K/V is static - cache once and reuse
        residual = x
        x = self.norm2(x)
        cross_cache = kv_cache.get("cross") if kv_cache else None

        if cross_cache is not None:
            # Use cached encoder K/V, don't recompute
            x, new_cross_cache = self.cross_attn(
                x,
                kv=None,  # Don't pass encoder_out, use cache
                mask=cross_attn_mask,
                kv_cache=cross_cache,
            )
        else:
            # First call: compute and cache encoder K/V
            x, new_cross_cache = self.cross_attn(
                x,
                kv=encoder_out,
                mask=cross_attn_mask,
                kv_cache=None,
            )

        x = self.dropout(x) + residual
        new_kv_cache["cross"] = new_cross_cache

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x) + residual

        return x, new_kv_cache if new_kv_cache else None


# =============================================================================
# Encoder & Decoder
# =============================================================================


class Encoder(nn.Module):
    """Transformer Encoder."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, n_kv_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len, rope_base)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Padding mask (batch, seq_len)
        """
        seq_len = x.shape[1]
        cos, sin = self.rope(seq_len, x.device, x.dtype)

        # Convert padding mask to attention mask if provided
        attn_mask = None
        if mask is not None:
            # mask: (batch, seq_len) with 1 for valid, 0 for pad
            # attn_mask: (batch, 1, 1, seq_len) with 0 for valid, -inf for pad
            attn_mask = (~mask).unsqueeze(1).unsqueeze(2).to(x.dtype) * torch.finfo(x.dtype).min

        for layer in self.layers:
            x = layer(x, mask=attn_mask, cos=cos, sin=sin)

        return self.norm(x)


class Decoder(nn.Module):
    """Transformer Decoder."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, n_kv_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len, rope_base)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            x: Decoder input (batch, seq_len, d_model)
            encoder_out: Encoder output (batch, src_len, d_model)
            encoder_mask: Encoder padding mask (batch, src_len)
            decoder_mask: Decoder padding mask (batch, tgt_len)
            kv_cache: KV cache dict indexed by layer

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated cache
        """
        seq_len = x.shape[1]

        # Get current position for RoPE
        if kv_cache is not None and 0 in kv_cache and "self" in kv_cache[0]:
            # During generation, we only process new tokens
            start_pos = kv_cache[0]["self"][0].shape[2]
            cos, sin = self.rope(start_pos + seq_len, x.device, x.dtype)
            # FIX: Slice to exact range needed for new tokens
            cos = cos[start_pos:start_pos + seq_len]
            sin = sin[start_pos:start_pos + seq_len]
        else:
            cos, sin = self.rope(seq_len, x.device, x.dtype)

        self_attn_mask = None
        if not kv_cache:
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
            if decoder_mask is not None:
                pad_mask = (~decoder_mask).unsqueeze(1).unsqueeze(2).to(x.dtype) * torch.finfo(x.dtype).min
                self_attn_mask = causal_mask + pad_mask
            else:
                self_attn_mask = causal_mask

        # Convert encoder padding mask to cross-attention mask
        cross_attn_mask = None
        if encoder_mask is not None:
            cross_attn_mask = (~encoder_mask).unsqueeze(1).unsqueeze(2).to(x.dtype) * torch.finfo(x.dtype).min

        new_kv_cache = {}
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache.get(i) if kv_cache else None
            x, layer_new_cache = layer(
                x,
                encoder_out,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                cos=cos,
                sin=sin,
                kv_cache=layer_cache,
            )
            if layer_new_cache:
                new_kv_cache[i] = layer_new_cache

        return self.norm(x), new_kv_cache if new_kv_cache else None


# =============================================================================
# Full Transformer
# =============================================================================


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Features:
    - Shared vocabulary with optional weight tying
    - RoPE positional embeddings
    - GQA with Flash Attention
    - KV-cache for efficient generation
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

        # Store config as attributes
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

        # Embeddings (shared for encoder and decoder)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_scale = math.sqrt(d_model)

        # Encoder and Decoder
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_base=rope_base,
        )

        # Output projection (tied with embedding weights)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode source sequence.

        Args:
            src: Source token IDs (batch, src_len)
            src_mask: Padding mask (batch, src_len), 1 for valid, 0 for pad

        Returns:
            Encoder output (batch, src_len, d_model)
        """
        x = self.embed(src) * self.embed_scale
        x = self.embed_dropout(x)
        return self.encoder(x, mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Decode target sequence.

        Args:
            tgt: Target token IDs (batch, tgt_len)
            encoder_out: Encoder output
            encoder_mask: Encoder padding mask
            tgt_mask: Target padding mask
            kv_cache: KV cache for generation

        Returns:
            Logits (batch, tgt_len, vocab_size), updated_kv_cache
        """
        x = self.embed(tgt) * self.embed_scale
        x = self.embed_dropout(x)
        x, new_cache = self.decoder(x, encoder_out, encoder_mask, tgt_mask, kv_cache)
        logits = self.lm_head(x)
        return logits, new_cache

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass for training.

        Args:
            src: Source token IDs (batch, src_len)
            tgt: Target input IDs (batch, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target padding mask

        Returns:
            Logits (batch, tgt_len, vocab_size)
        """
        encoder_out = self.encode(src, src_mask)
        logits, _ = self.decode(tgt, encoder_out, src_mask, tgt_mask)
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate sequences using greedy/sampling with KV-cache.

        Args:
            src: Source token IDs (batch, src_len)
            src_mask: Source padding mask
            max_len: Maximum generation length
            temperature: Sampling temperature (1.0 = greedy if top_k=1)
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
            eos_id: End of sequence token ID

        Returns:
            Generated token IDs (batch, gen_len)
        """
        batch_size = src.shape[0]
        device = src.device
        eos_id = eos_id or self.eos_id

        # Encode source
        encoder_out = self.encode(src, src_mask)

        # Initialize with BOS
        generated = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        kv_cache = {}

        for _ in range(max_len):
            # Only process the last token when using cache
            if kv_cache:
                input_ids = generated[:, -1:]
            else:
                input_ids = generated

            logits, kv_cache = self.decode(
                input_ids, encoder_out, src_mask, kv_cache=kv_cache
            )

            # Get logits for last position
            # FIX: Handle temperature=0 for greedy decoding
            if temperature > 0:
                next_logits = logits[:, -1, :] / temperature
            else:
                next_logits = logits[:, -1, :]

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
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

            # Update finished mask
            finished = finished | (next_token.squeeze(-1) == eos_id)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return generated

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 256,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        eos_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.

        Args:
            src: Source token IDs (batch, src_len)
            src_mask: Source padding mask
            max_len: Maximum generation length
            beam_size: Number of beams
            length_penalty: Length penalty (>1 favors longer, <1 favors shorter)
            eos_id: End of sequence token ID

        Returns:
            (best_sequences, scores): (batch, max_len), (batch,)
        """
        batch_size = src.shape[0]
        device = src.device
        eos_id = eos_id or self.eos_id

        # Encode source
        encoder_out = self.encode(src, src_mask)

        # Expand for beam search: (batch * beam, ...)
        encoder_out = encoder_out.unsqueeze(1).expand(-1, beam_size, -1, -1)
        encoder_out = encoder_out.reshape(batch_size * beam_size, -1, encoder_out.shape[-1])

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).expand(-1, beam_size, -1)
            src_mask = src_mask.reshape(batch_size * beam_size, -1)

        # Initialize beams
        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_scores[:, 1:] = float("-inf")  # Only first beam active initially
        beam_scores = beam_scores.view(-1)

        # Initialize sequences with BOS
        generated = torch.full(
            (batch_size * beam_size, 1), self.bos_id, dtype=torch.long, device=device
        )

        # Track finished beams
        done = [False] * batch_size
        generated_hyps = [[] for _ in range(batch_size)]

        kv_cache = {}

        for step in range(max_len):
            # Get logits for last token
            if kv_cache:
                input_ids = generated[:, -1:]
            else:
                input_ids = generated

            logits, kv_cache = self.decode(
                input_ids, encoder_out, src_mask, kv_cache=kv_cache
            )

            next_logits = F.log_softmax(logits[:, -1, :], dim=-1)  # (batch*beam, vocab)

            # Add beam scores
            next_scores = next_logits + beam_scores.unsqueeze(-1)  # (batch*beam, vocab)

            # Reshape for beam selection: (batch, beam * vocab)
            next_scores = next_scores.view(batch_size, beam_size * self.vocab_size)

            # Get top 2*beam candidates
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )

            # Compute beam and token indices
            next_beam_indices = next_tokens // self.vocab_size
            next_tokens = next_tokens % self.vocab_size

            # Build next beams
            next_beam_scores = []
            next_beam_tokens = []
            next_beam_idx = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # FIX: Use -inf for finished batches to avoid score interference
                    next_beam_scores.extend([float('-inf')] * beam_size)
                    next_beam_tokens.extend([self.pad_id] * beam_size)
                    next_beam_idx.extend(list(range(batch_idx * beam_size, (batch_idx + 1) * beam_size)))
                    continue

                beam_count = 0
                for idx in range(2 * beam_size):
                    beam_id = next_beam_indices[batch_idx, idx]
                    token_id = next_tokens[batch_idx, idx]
                    score = next_scores[batch_idx, idx]

                    global_beam_idx = batch_idx * beam_size + beam_id

                    if token_id.item() == eos_id:
                        # Finished hypothesis - store with EOS token
                        seq = generated[global_beam_idx].clone()
                        # Append EOS to the sequence
                        seq = torch.cat([seq, torch.tensor([eos_id], device=device)])
                        seq_len = (step + 1)
                        final_score = score / (seq_len ** length_penalty)
                        generated_hyps[batch_idx].append((final_score.item(), seq))

                        if len(generated_hyps[batch_idx]) >= beam_size:
                            done[batch_idx] = True
                    else:
                        next_beam_scores.append(score.item())
                        next_beam_tokens.append(token_id.item())
                        next_beam_idx.append(global_beam_idx.item())
                        beam_count += 1

                    if beam_count >= beam_size:
                        break

                # FIX: Pad remaining slots with -inf if not enough candidates
                while beam_count < beam_size:
                    next_beam_scores.append(float('-inf'))
                    next_beam_tokens.append(self.pad_id)
                    next_beam_idx.append(batch_idx * beam_size)
                    beam_count += 1

            if all(done):
                break

            # Reorder beams
            beam_indices = torch.tensor(next_beam_idx, device=device)
            generated = generated[beam_indices]
            generated = torch.cat([
                generated,
                torch.tensor(next_beam_tokens, device=device).unsqueeze(-1)
            ], dim=-1)
            beam_scores = torch.tensor(next_beam_scores, device=device)

            # Reorder KV cache
            if kv_cache is not None:
                new_cache = {}
                for layer_idx, layer_cache in kv_cache.items():
                    new_layer_cache = {}
                    for key, (k, v) in layer_cache.items():
                        new_layer_cache[key] = (
                            k.index_select(0, beam_indices),
                            v.index_select(0, beam_indices),
                        )
                    new_cache[layer_idx] = new_layer_cache
                kv_cache = new_cache

        # Select best hypothesis for each batch
        best_sequences = []
        best_scores = []

        for batch_idx in range(batch_size):
            if generated_hyps[batch_idx]:
                hyps = sorted(generated_hyps[batch_idx], key=lambda x: x[0], reverse=True)
                best_score, best_seq = hyps[0]
            else:
                # Use current best beam if no hypothesis finished
                best_seq = generated[batch_idx * beam_size]
                best_score = beam_scores[batch_idx * beam_size].item()

            best_sequences.append(best_seq)
            best_scores.append(best_score)

        # Pad sequences to same length
        max_seq_len = max(len(seq) for seq in best_sequences)
        padded_sequences = torch.full(
            (batch_size, max_seq_len), self.pad_id, dtype=torch.long, device=device
        )
        for i, seq in enumerate(best_sequences):
            padded_sequences[i, :len(seq)] = seq

        return padded_sequences, torch.tensor(best_scores, device=device)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.inference_mode()
    def parallel_beam_search(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 256,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        eos_id: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized beam search - all beams processed in parallel.

        More efficient than sequential beam search as it:
        - Processes all beams for all batches in a single forward pass
        - Uses vectorized operations instead of Python loops
        - Maintains better GPU utilization

        Args:
            src: Source token IDs (batch, src_len)
            src_mask: Source padding mask
            max_len: Maximum generation length
            beam_size: Number of beams
            length_penalty: Length penalty (>1 favors longer, <1 favors shorter)
            early_stopping: Stop when beam_size hypotheses complete per batch
            eos_id: End of sequence token ID

        Returns:
            (best_sequences, scores): (batch, max_len), (batch,)
            If return_all=True, returns (best_sequences, scores, all_sequences, all_scores)
        """
        batch_size = src.shape[0]
        device = src.device
        eos_id = eos_id or self.eos_id

        # Encode source once
        encoder_out = self.encode(src, src_mask)

        # Expand encoder output for beams: (batch, src_len, d) -> (batch * beam, src_len, d)
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1)
        encoder_out = encoder_out.view(batch_size * beam_size, -1, encoder_out.shape[-1])

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).repeat(1, beam_size, 1)
            src_mask = src_mask.view(batch_size * beam_size, -1)

        # Initialize beam scores: (batch * beam,)
        beam_scores = torch.zeros(batch_size * beam_size, device=device)
        beam_scores[~torch.arange(batch_size * beam_size, device=device).fmod(beam_size).eq(0)] = -1e9

        # Initialize sequences with BOS: (batch * beam, 1)
        generated = torch.full(
            (batch_size * beam_size, 1), self.bos_id, dtype=torch.long, device=device
        )

        # Track finished beams per batch
        finished_mask = torch.zeros(batch_size * beam_size, dtype=torch.bool, device=device)
        finished_scores = torch.full((batch_size, beam_size), float('-inf'), device=device)
        finished_seqs = torch.zeros(batch_size, beam_size, max_len + 1, dtype=torch.long, device=device)
        num_finished = torch.zeros(batch_size, dtype=torch.long, device=device)

        kv_cache: Dict = {}

        for step in range(max_len):
            # Process only last token when using cache
            if kv_cache:
                input_ids = generated[:, -1:]
            else:
                input_ids = generated

            # Get logits for all beams in parallel
            logits, kv_cache = self.decode(input_ids, encoder_out, src_mask, kv_cache=kv_cache)
            next_logits = F.log_softmax(logits[:, -1, :], dim=-1)  # (batch * beam, vocab)

            # Add current beam scores
            next_scores = next_logits + beam_scores.unsqueeze(-1)  # (batch * beam, vocab)

            # Mask finished beams (except pad token)
            next_scores[finished_mask] = -1e9
            next_scores[finished_mask, self.pad_id] = beam_scores[finished_mask]

            # Reshape for batch-level beam selection: (batch, beam * vocab)
            next_scores = next_scores.view(batch_size, beam_size * self.vocab_size)

            # Get top-k candidates per batch: (batch, 2 * beam)
            topk_scores, topk_indices = torch.topk(
                next_scores, k=2 * beam_size, dim=-1, largest=True, sorted=True
            )

            # Decode beam and token indices
            topk_beam_idx = topk_indices // self.vocab_size  # Which beam
            topk_token_idx = topk_indices % self.vocab_size  # Which token

            # Select top beam_size candidates per batch
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
                        # Store finished hypothesis with EOS token
                        if num_finished[b] < beam_size:
                            seq_len = step + 1
                            final_score = score / (seq_len ** length_penalty)
                            idx = num_finished[b].item()
                            finished_scores[b, idx] = final_score
                            # FIX: Store sequence including EOS token
                            finished_seqs[b, idx, :generated.shape[1]] = generated[global_beam]
                            finished_seqs[b, idx, generated.shape[1]] = eos_id
                            num_finished[b] += 1

                        if early_stopping and num_finished[b] >= beam_size:
                            # Mark all beams as finished for this batch
                            finished_mask[b * beam_size:(b + 1) * beam_size] = True
                        continue

                    new_beam_scores[b, selected] = score
                    new_tokens[b, selected] = token_idx
                    beam_mapping[b, selected] = global_beam
                    selected += 1

                # Pad remaining if not enough non-EOS candidates
                while selected < beam_size:
                    new_beam_scores[b, selected] = float('-inf')
                    new_tokens[b, selected] = self.pad_id
                    beam_mapping[b, selected] = b * beam_size
                    selected += 1

            # Check if all batches are done
            if finished_mask.all():
                break

            # Flatten beam mapping for reordering
            beam_mapping_flat = beam_mapping.view(-1)

            # Reorder sequences and extend
            finished_mask = finished_mask[beam_mapping_flat]
            generated = generated[beam_mapping_flat]
            generated = torch.cat([generated, new_tokens.view(-1, 1)], dim=1)

            # Update beam scores
            beam_scores = new_beam_scores.view(-1)

            # Reorder KV cache
            if kv_cache:
                new_cache: Dict = {}
                for layer_idx, layer_cache in kv_cache.items():
                    new_layer_cache: Dict = {}
                    for key, (k, v) in layer_cache.items():
                        new_layer_cache[key] = (
                            k.index_select(0, beam_mapping_flat),
                            v.index_select(0, beam_mapping_flat),
                        )
                    new_cache[layer_idx] = new_layer_cache
                kv_cache = new_cache

            # FIX: Accumulate finished mask instead of overwriting
            finished_mask = finished_mask | (new_tokens.view(-1) == self.pad_id)

        # Select best hypothesis per batch
        best_sequences = []
        best_scores_list = []

        for b in range(batch_size):
            if num_finished[b] > 0:
                # Get best from finished hypotheses
                best_idx = finished_scores[b, :num_finished[b]].argmax()
                best_seq = finished_seqs[b, best_idx]
                best_score = finished_scores[b, best_idx]
            else:
                # Use best beam if no hypothesis finished
                best_beam = b * beam_size
                best_seq = generated[best_beam]
                best_score = beam_scores[best_beam] / (generated.shape[1] ** length_penalty)

            best_sequences.append(best_seq)
            best_scores_list.append(best_score)

        # Pad to same length
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

    @torch.inference_mode()
    def speculative_generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        draft_model: Optional["Transformer"] = None,
        gamma: int = 5,
        max_len: int = 256,
        temperature: float = 1.0,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Speculative decoding for faster generation.

        Uses a smaller draft model to propose gamma tokens, then verifies
        them with the target model in a single forward pass.

        Args:
            src: Source token IDs (batch, src_len)
            src_mask: Source padding mask
            draft_model: Smaller model for speculation (must share tokenizer)
            gamma: Number of tokens to speculate per step
            max_len: Maximum generation length
            temperature: Sampling temperature
            eos_id: End of sequence token ID

        Returns:
            Generated token IDs (batch, gen_len)
        """
        if draft_model is None:
            # Fall back to regular generation if no draft model
            return self.generate(
                src, src_mask, max_len=max_len, temperature=temperature, eos_id=eos_id
            )

        batch_size = src.shape[0]
        device = src.device
        eos_id = eos_id or self.eos_id

        # FIX: Handle temperature=0
        temp = max(temperature, 1e-10)

        # Encode source for both models
        encoder_out = self.encode(src, src_mask)
        draft_encoder_out = draft_model.encode(src, src_mask)

        # Initialize with BOS
        generated = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Don't use KV cache for speculative decoding - it's complex and error-prone
        # The overhead is acceptable given the speedup from speculation

        while generated.shape[1] < max_len + 1:
            if finished.all():
                break

            # Step 1: Draft model generates gamma tokens
            draft_tokens = []
            draft_probs = []
            current_input = generated

            for _ in range(gamma):
                # Forward without cache for simplicity
                draft_logits, _ = draft_model.decode(
                    current_input, draft_encoder_out, src_mask, kv_cache=None
                )
                draft_next_logits = draft_logits[:, -1, :] / temp
                draft_next_probs = F.softmax(draft_next_logits, dim=-1)

                # Sample from draft
                if temperature > 0:
                    next_token = torch.multinomial(draft_next_probs, num_samples=1)
                else:
                    next_token = draft_next_logits.argmax(dim=-1, keepdim=True)

                draft_tokens.append(next_token)
                draft_probs.append(draft_next_probs)
                current_input = torch.cat([current_input, next_token], dim=1)

            draft_tokens = torch.cat(draft_tokens, dim=1)  # (batch, gamma)

            # Step 2: Target model verifies all tokens in one pass
            verify_input = torch.cat([generated, draft_tokens], dim=1)

            # Forward through target model without cache
            target_logits, _ = self.decode(
                verify_input, encoder_out, src_mask, kv_cache=None
            )

            # Get target probabilities for verification positions
            # We need positions [len(generated)-1, ..., len(generated)+gamma-1]
            start_idx = generated.shape[1] - 1
            target_probs = F.softmax(target_logits[:, start_idx:start_idx + gamma + 1, :] / temp, dim=-1)

            # Step 3: Verify and accept tokens
            accepted = torch.ones(batch_size, dtype=torch.long, device=device) * gamma

            for i in range(gamma):
                # Get probabilities for this position
                draft_p = draft_probs[i]
                target_p = target_probs[:, i, :]

                # Get the drafted token
                token = draft_tokens[:, i:i+1]

                # Compute acceptance probability: min(1, target_p / draft_p)
                draft_token_p = draft_p.gather(1, token).squeeze(-1)
                target_token_p = target_p.gather(1, token).squeeze(-1)

                accept_prob = torch.clamp(target_token_p / (draft_token_p + 1e-10), max=1.0)

                # Sample acceptance
                uniform = torch.rand(batch_size, device=device)
                reject_mask = uniform > accept_prob

                # Update accepted count for rejected samples
                not_accepted_yet = accepted == gamma
                first_reject = reject_mask & not_accepted_yet
                accepted[first_reject] = i

            # Step 4: Build final sequences
            new_tokens_list = []
            for b in range(batch_size):
                n_accepted = accepted[b].item()

                if n_accepted == gamma:
                    # All tokens accepted, sample one more from target
                    final_logits = target_probs[b, -1, :]
                    if temperature > 0:
                        final_token = torch.multinomial(final_logits.unsqueeze(0), num_samples=1)
                    else:
                        final_token = final_logits.unsqueeze(0).argmax(dim=-1, keepdim=True)
                    accepted_tokens = torch.cat([draft_tokens[b:b+1, :], final_token], dim=1)
                else:
                    # Sample from adjusted distribution at rejection point
                    draft_p = draft_probs[n_accepted][b]
                    target_p = target_probs[b, n_accepted, :]

                    # Compute adjusted distribution: max(0, target_p - draft_p) / Z
                    adjusted = torch.clamp(target_p - draft_p, min=0)
                    adjusted = adjusted / (adjusted.sum() + 1e-10)

                    if temperature > 0 and adjusted.sum() > 0:
                        new_token = torch.multinomial(adjusted.unsqueeze(0), num_samples=1)
                    else:
                        new_token = target_p.unsqueeze(0).argmax(dim=-1, keepdim=True)

                    accepted_tokens = torch.cat([draft_tokens[b:b+1, :n_accepted], new_token], dim=1)

                new_tokens_list.append(accepted_tokens)

            # Pad and concatenate
            max_new = max(t.shape[1] for t in new_tokens_list)
            new_tokens_padded = torch.full(
                (batch_size, max_new), self.pad_id, dtype=torch.long, device=device
            )
            for b, t in enumerate(new_tokens_list):
                new_tokens_padded[b, :t.shape[1]] = t.squeeze(0)

            generated = torch.cat([generated, new_tokens_padded], dim=1)

            if generated.shape[1] >= max_len + 1:
                generated = generated[:, :max_len + 1]
                break

            # Check for EOS
            eos_positions = (generated == eos_id).any(dim=1)
            finished = finished | eos_positions

        return generated

    @classmethod
    def create_draft_model(
        cls,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        rope_base: float = 10000.0,
        num_layers: int = 2,
        share_embeddings: bool = True,
        parent_model: Optional["Transformer"] = None,
    ) -> "Transformer":
        """
        Create a smaller draft model for speculative decoding.

        The draft model has fewer layers but same vocabulary and embedding size,
        allowing fast speculation with compatible outputs.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_kv_heads: Number of key-value heads
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            pad_id: Padding token ID
            bos_id: Beginning of sequence token ID
            eos_id: End of sequence token ID
            rope_base: RoPE base frequency
            num_layers: Number of layers in draft model (default: 2)
            share_embeddings: Whether to share embedding weights with parent
            parent_model: Parent model to share embeddings with

        Returns:
            Smaller Transformer model for drafting
        """
        draft_model = cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=0.0,  # No dropout for inference
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            rope_base=rope_base,
        )

        # Optionally share embeddings with parent
        if share_embeddings and parent_model is not None:
            draft_model.embed.weight = parent_model.embed.weight
            draft_model.lm_head.weight = parent_model.lm_head.weight

        return draft_model

    def compile_for_inference(
        self,
        mode: str = "reduce-overhead",
        backend: str = "inductor",
        dynamic: bool = True,
    ) -> "Transformer":
        """
        Apply torch.compile() for faster inference.

        Compiles the encoder and decoder blocks for optimized execution.
        Best used with static batch sizes for maximum benefit.

        Args:
            mode: Compilation mode
                - "default": Good balance of compile time and speedup
                - "reduce-overhead": Minimize framework overhead (best for inference)
                - "max-autotune": Maximum performance, longer compile time
            backend: Compilation backend (default: "inductor")
            dynamic: Support dynamic shapes (default: True for variable seq lengths)

        Returns:
            Self with compiled modules
        """
        # Compile encoder
        self.encoder = torch.compile(
            self.encoder,
            mode=mode,
            backend=backend,
            dynamic=dynamic,
        )

        # Compile decoder
        self.decoder = torch.compile(
            self.decoder,
            mode=mode,
            backend=backend,
            dynamic=dynamic,
        )

        return self
