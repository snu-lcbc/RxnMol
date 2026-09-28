"""PyTorch Lightning module for training and inference."""

import lightning as L
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from transformers import get_cosine_schedule_with_warmup

from .config import ModelConfig, TrainConfig
from .transformer import Transformer
from .decoder_model import CausalTransformer
from .tokenizer import SMILESTokenizer


class ReactionModule(L.LightningModule):
    """
    PyTorch Lightning module for reaction prediction.

    Supports both encoder-decoder (Transformer) and decoder-only (CausalTransformer) models.
    Handles training, validation, and generation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        tokenizer: Optional[SMILESTokenizer] = None,
        model_type: str = "transformer",
        ignore_index: int = -100,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the module.

        Args:
            model_config: Model architecture configuration
            train_config: Training configuration
            tokenizer: SMILES tokenizer (for decoding during validation)
            model_type: "transformer" or "decoder_only"
            ignore_index: Index to ignore in loss computation for causal models
            config: Full resolved config for logging
        """
        super().__init__()
        if config is not None:
            # Log only the resolved config to avoid duplicate entries in hparams.yaml.
            self.save_hyperparameters(config)
        else:
            self.save_hyperparameters(ignore=["tokenizer"])

        self.model_config = model_config
        self.train_config = train_config
        self.tokenizer = tokenizer
        if model_type in {"encoder_decoder", "seq2seq"}:
            model_type = "transformer"
        self.model_type = model_type
        self.ignore_index = ignore_index
        self.l2sp_lambda = float(getattr(train_config, "l2sp_lambda", 0.0) or 0.0)
        self._l2sp_base: Optional[Dict[str, torch.Tensor]] = None
        if config and isinstance(config.get("tokenizer"), dict):
            self.tokenizer_task = config["tokenizer"].get("task", "forward")
        else:
            self.tokenizer_task = "forward"

        # Build appropriate model based on type
        if model_type == "decoder_only":
            self.model = CausalTransformer(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                n_heads=model_config.n_heads,
                n_kv_heads=model_config.n_kv_heads,
                n_layers=model_config.n_layers,
                d_ff=model_config.d_ff,
                max_seq_len=model_config.max_seq_len,
                dropout=model_config.dropout,
                pad_id=model_config.pad_id,
                bos_id=model_config.bos_id,
                eos_id=model_config.eos_id,
                rope_base=model_config.rope_base,
            )
        elif model_type == "transformer":
            self.model = Transformer(
                vocab_size=model_config.vocab_size,
                d_model=model_config.d_model,
                n_heads=model_config.n_heads,
                n_kv_heads=model_config.n_kv_heads,
                n_layers=model_config.n_layers,
                d_ff=model_config.d_ff,
                max_seq_len=model_config.max_seq_len,
                dropout=model_config.dropout,
                pad_id=model_config.pad_id,
                bos_id=model_config.bos_id,
                eos_id=model_config.eos_id,
                rope_base=model_config.rope_base,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Loss function with label smoothing
        # For decoder-only, we ignore the ignore_index used for prompt tokens
        loss_ignore_index = ignore_index if model_type == "decoder_only" else model_config.pad_id
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=loss_ignore_index,
            label_smoothing=train_config.label_smoothing,
        )

        # Note: Using Lightning's sync_dist=True for validation metrics
        # No manual all_gather needed

    def set_l2sp_base(self) -> None:
        """Capture base weights for L2-SP regularization."""
        if self.l2sp_lambda <= 0:
            return
        self._l2sp_base = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._l2sp_base[name] = param.detach().clone()

    def forward(
        self,
        src: torch.Tensor = None,
        tgt: torch.Tensor = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for both encoder-decoder and decoder-only models.

        For encoder-decoder: uses src, tgt, src_mask, tgt_mask
        For decoder-only: uses input_ids, attention_mask
        """
        if self.model_type == "decoder_only":
            # Decoder-only model
            logits, _ = self.model(input_ids, attention_mask=attention_mask)
            return logits
        else:
            # Encoder-decoder model
            return self.model(src, tgt, src_mask, tgt_mask)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for both model types.

        Args:
            batch: Dict with model-specific keys
                - Encoder-decoder: src, src_mask, tgt_in, tgt_out, tgt_mask
                - Decoder-only: input_ids, attention_mask, labels
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        if self.model_type == "decoder_only":
            # Decoder-only training
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward pass
            logits = self(input_ids=input_ids, attention_mask=attention_mask)

            # Shift logits and labels for next-token prediction
            # logits: predict position i -> i+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute loss
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            # Encoder-decoder training
            src = batch["src"]
            src_mask = batch["src_mask"]
            tgt_in = batch["tgt_in"]
            tgt_out = batch["tgt_out"]
            tgt_mask = batch["tgt_mask"]

            # Forward pass
            logits = self(src, tgt_in, src_mask, tgt_mask)

            # Compute loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1)
            )

        if self.l2sp_lambda > 0 and self._l2sp_base:
            l2sp = torch.zeros((), device=loss.device, dtype=loss.dtype)
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                base = self._l2sp_base.get(name)
                if base is None:
                    continue
                if base.device != param.device:
                    base = base.to(param.device)
                    self._l2sp_base[name] = base
                l2sp = l2sp + (param - base).pow(2).sum()
            loss = loss + self.l2sp_lambda * l2sp
            self.log("train/l2sp", l2sp, on_step=True, on_epoch=False, prog_bar=False)

        # Log metrics
        # Step-level: no sync (fast), Epoch-level: sync_dist=True (accurate across GPUs)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss_epoch", loss, on_epoch=True, prog_bar=False, sync_dist=True)
        # LR logging handled by LearningRateMonitor callback

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step for both model types.

        Args:
            batch: Dict with model-specific keys
            batch_idx: Batch index

        Returns:
            Dict with loss
        """
        if self.model_type == "decoder_only":
            # Decoder-only validation
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # Forward pass
            logits = self(input_ids=input_ids, attention_mask=attention_mask)

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute loss
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Compute token-level accuracy (only on non-ignored positions)
            preds = shift_logits.argmax(dim=-1)
            mask = shift_labels != self.ignore_index
            correct = ((preds == shift_labels) & mask).sum()
            total = mask.sum()

            topk = min(5, shift_logits.size(-1))
            topk_preds = shift_logits.topk(topk, dim=-1).indices
            topk_correct = ((topk_preds == shift_labels.unsqueeze(-1)) & mask.unsqueeze(-1)).any(dim=-1)
            token_acc_topk = topk_correct.sum().float() / total.float().clamp(min=1)
            seq_exact_match = (((preds == shift_labels) | ~mask).all(dim=-1)).float().mean()
        else:
            # Encoder-decoder validation
            src = batch["src"]
            src_mask = batch["src_mask"]
            tgt_in = batch["tgt_in"]
            tgt_out = batch["tgt_out"]
            tgt_mask = batch["tgt_mask"]

            # Forward pass
            logits = self(src, tgt_in, src_mask, tgt_mask)

            # Compute loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                tgt_out.view(-1)
            )

            # Compute token-level accuracy
            preds = logits.argmax(dim=-1)
            mask = tgt_mask.bool()
            correct = ((preds == tgt_out) & mask).sum()
            total = mask.sum()

            topk = min(5, logits.size(-1))
            topk_preds = logits.topk(topk, dim=-1).indices
            topk_correct = ((topk_preds == tgt_out.unsqueeze(-1)) & mask.unsqueeze(-1)).any(dim=-1)
            token_acc_topk = topk_correct.sum().float() / total.float().clamp(min=1)
            seq_exact_match = (((preds == tgt_out) | ~mask).all(dim=-1)).float().mean()

        token_acc = correct.float() / total.float().clamp(min=1)

        # Log metrics - Lightning handles sync_dist automatically across DDP ranks
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/token_acc", token_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/token_acc_top5", token_acc_topk, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/seq_exact_match", seq_exact_match, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss}

    def on_validation_epoch_end(self):
        """No manual aggregation needed - Lightning handles sync_dist automatically."""
        pass

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.lr,
            betas=self.train_config.betas,
            eps=self.train_config.eps,
            weight_decay=self.train_config.weight_decay,
        )

        # Build scheduler based on config
        scheduler_type = self.train_config.scheduler.lower()

        if scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.train_config.warmup_steps,
                num_training_steps=self.train_config.max_steps,
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif scheduler_type == "cosine_restarts":
            from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.train_config.warmup_steps,
                num_training_steps=self.train_config.max_steps,
                num_cycles=self.train_config.num_cycles,
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.train_config.warmup_steps,
                num_training_steps=self.train_config.max_steps,
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif scheduler_type == "constant":
            from transformers import get_constant_schedule_with_warmup
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.train_config.warmup_steps,
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        elif scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.train_config.scheduler_factor,
                patience=self.train_config.scheduler_patience,
                verbose=True,
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            }

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor = None,
        src_mask: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_len: int = 256,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate predictions for both model types.

        Args:
            src: Source token IDs for encoder-decoder (batch, src_len)
            src_mask: Source padding mask for encoder-decoder
            input_ids: Input token IDs for decoder-only (batch, prompt_len)
            attention_mask: Attention mask for decoder-only
            max_len: Maximum generation length (encoder-decoder)
            max_new_tokens: Maximum new tokens (decoder-only)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated token IDs (batch, gen_len)
        """
        self.eval()

        if self.model_type == "decoder_only":
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=self.model_config.eos_id,
            )
        else:
            return self.model.generate(
                src=src,
                src_mask=src_mask,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=self.model_config.eos_id,
            )

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor = None,
        src_mask: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_len: int = 256,
        max_new_tokens: int = None,
        beam_size: int = 5,
        length_penalty: float = 1.0,
    ) -> tuple:
        """
        Generate using beam search for both model types.

        Args:
            src: Source token IDs for encoder-decoder
            src_mask: Source padding mask for encoder-decoder
            input_ids: Input token IDs for decoder-only
            attention_mask: Attention mask for decoder-only
            max_len: Maximum generation length
            max_new_tokens: Maximum new tokens (decoder-only)
            beam_size: Beam size
            length_penalty: Length penalty

        Returns:
            (sequences, scores)
        """
        self.eval()

        if self.model_type == "decoder_only":
            return self.model.parallel_beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                eos_id=self.model_config.eos_id,
            )
        else:
            return self.model.parallel_beam_search(
                src=src,
                src_mask=src_mask,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                eos_id=self.model_config.eos_id,
            )

    def predict_smiles(
        self,
        src_smiles: str,
        max_len: int = 256,
        beam_size: int = 5,
    ) -> List[str]:
        """
        Predict product SMILES from reactant SMILES.

        Args:
            src_smiles: Reactant SMILES string
            max_len: Maximum generation length
            beam_size: Beam size for beam search

        Returns:
            List of predicted product SMILES
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for SMILES prediction")

        self.eval()
        device = next(self.parameters()).device

        # Encode source using reaction format
        encoded = self.tokenizer.encode_reaction(
            reactants=src_smiles,
            products=None,
            task=self.tokenizer_task,
            model_type="transformer",
            max_src_length=max_len,
            truncation=True,
            return_tensors=None,
        )
        src_ids = encoded["src"]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = torch.ones_like(src, dtype=torch.bool)

        # Generate
        if beam_size > 1:
            sequences, scores = self.beam_search(
                src, src_mask, max_len=max_len, beam_size=beam_size
            )
        else:
            sequences = self.generate(
                src, src_mask, max_len=max_len, top_k=1
            )

        # Decode
        predictions = []
        for seq in sequences:
            smiles = self.tokenizer.decode(seq.tolist(), skip_special=True, stop_at_eos=True)
            predictions.append(smiles)

        return predictions

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Add extra info to checkpoint."""
        checkpoint["model_config"] = self.model_config
        checkpoint["train_config"] = self.train_config
        checkpoint["model_type"] = self.model_type

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer: Optional[SMILESTokenizer] = None,
        **kwargs,
    ) -> "ReactionModule":
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            tokenizer: Optional tokenizer for inference
            **kwargs: Additional arguments

        Returns:
            Loaded ReactionModule
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model_config = checkpoint.get("model_config")
        train_config = checkpoint.get("train_config")
        model_type = checkpoint.get("model_type", "transformer")

        if model_config is None or train_config is None:
            # Try loading from hparams
            hparams = checkpoint.get("hyper_parameters", {})
            model_config = hparams.get("model_config")
            train_config = hparams.get("train_config")
            model_type = hparams.get("model_type", "transformer")

        module = cls(
            model_config=model_config,
            train_config=train_config,
            tokenizer=tokenizer,
            model_type=model_type,
        )
        module.load_state_dict(checkpoint["state_dict"])

        return module


def compute_metrics(
    predictions: List[str],
    targets: List[str],
) -> Dict[str, float]:
    """
    Compute reaction prediction metrics.

    Args:
        predictions: List of predicted SMILES
        targets: List of target SMILES

    Returns:
        Dict with accuracy metrics
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")

    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0}

    # Exact match accuracy
    exact_matches = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())

    return {
        "exact_match": exact_matches / n,
        "total": n,
        "correct": exact_matches,
    }
