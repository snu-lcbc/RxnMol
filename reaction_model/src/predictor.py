"""
Standalone Reaction Predictor for use in external projects.

Usage:
    from mechanistic_rxn.src.predictor import RxnPredictor

    # Load predictor
    predictor = RxnPredictor(
        model_dir="/path/to/model_dir",  # Contains checkpoint + tokenizer.json or tokenizer/
        device="cuda"
    )

    # Single prediction
    product = predictor.predict("CC(=O)O.CCO")  # Returns predicted product SMILES

    # Batch prediction
    products = predictor.predict_batch(["CC(=O)O.CCO", "c1ccccc1.Br"])
"""

import os
import logging
import time
import gc
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .tokenizer import SMILESTokenizer
from .transformer import Transformer

logger = logging.getLogger(__name__)


class RxnPredictor:
    """
    Standalone reaction predictor for inference.

    Designed to be easily imported into other projects.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 128,
        max_len: int = 256,
        beam_size: int = 1,
        length_penalty: float = 1.0,
        task: str = "forward",
    ):
        """
        Initialize predictor.

        Args:
            model_dir: Directory containing checkpoint and tokenizer.
                       If provided, looks for:
                       - {model_dir}/checkpoints/last.ckpt or best.ckpt
                       - {model_dir}/tokenizer.json or {model_dir}/tokenizer/
            checkpoint_path: Explicit path to checkpoint (overrides model_dir)
            tokenizer_path: Explicit path to tokenizer (overrides model_dir)
            device: Device to run on ("cuda" or "cpu")
            batch_size: Max batch size for predictions
            max_len: Maximum sequence length
            beam_size: Beam size for generation (1 = greedy)
        """
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.task = task

        # Resolve paths
        if model_dir:
            model_dir = Path(model_dir)

            if tokenizer_path is None:
                if (model_dir / "tokenizer.json").exists():
                    tokenizer_path = model_dir / "tokenizer.json"
                elif (model_dir / "tokenizer").exists():
                    tokenizer_path = model_dir / "tokenizer"

            if checkpoint_path is None:
                # Priority: best_model.pt > model.pt > checkpoints/last.ckpt > any .ckpt
                if (model_dir / "best_model.pt").exists():
                    checkpoint_path = model_dir / "best_model.pt"
                elif (model_dir / "model.pt").exists():
                    checkpoint_path = model_dir / "model.pt"
                else:
                    ckpt_dir = model_dir / "checkpoints"
                    if ckpt_dir.exists():
                        if (ckpt_dir / "last.ckpt").exists():
                            checkpoint_path = ckpt_dir / "last.ckpt"
                        else:
                            ckpts = list(ckpt_dir.glob("*.ckpt"))
                            if ckpts:
                                checkpoint_path = sorted(ckpts)[-1]

        if checkpoint_path is None:
            raise ValueError("No checkpoint found. Provide checkpoint_path or model_dir with checkpoints.")

        if tokenizer_path is None:
            raise ValueError("No tokenizer found. Provide tokenizer_path or model_dir with tokenizer.json or tokenizer/.")

        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = SMILESTokenizer.from_pretrained(str(tokenizer_path))

        # Load model
        t0 = time.time()
        logger.info(f"Loading model from {checkpoint_path}")
        self._load_model(str(checkpoint_path))
        logger.info(f"Model loaded in {time.time() - t0:.2f}s")

        # Auto-configure batch size based on GPU memory
        if device == "cuda" and torch.cuda.is_available():
            self._auto_configure_batch_size()

    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract config
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
            model_type = checkpoint.get("model_type", "transformer")
        elif "hyper_parameters" in checkpoint:
            model_config = checkpoint["hyper_parameters"].get("model_config")
            model_type = checkpoint["hyper_parameters"].get("model_type", "transformer")
        else:
            raise ValueError("Cannot find model_config in checkpoint")

        if model_type in {"encoder_decoder", "seq2seq"}:
            model_type = "transformer"

        # Build model
        if model_type == "decoder_only":
            from .decoder_model import CausalTransformer
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
        else:
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

        # Load state dict - handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            # Our simple model.pt format
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            # Lightning checkpoint format
            state_dict = checkpoint["state_dict"]
            # Handle Lightning module prefix
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        else:
            raise ValueError("Cannot find state_dict in checkpoint")

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.config = model_config
        self.model_type = model_type

    def _auto_configure_batch_size(self):
        """Auto-configure batch size based on GPU memory."""
        if not torch.cuda.is_available():
            return

        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)

        if total_mem_gb < 12:  # 11GB (2080Ti)
            self.batch_size = min(self.batch_size, 512)
        elif total_mem_gb < 26:  # 24GB (3090/4090)
            self.batch_size = min(self.batch_size, 1024)
        else:  # 48GB+ (A6000/A100)
            self.batch_size = min(self.batch_size, 2048)

        logger.info(f"Using GPU: {gpu_name} ({total_mem_gb:.1f}GB), batch_size={self.batch_size}")

    def predict(
        self,
        reactants: str,
        beam_size: Optional[int] = None,
        length_penalty: Optional[float] = None,
    ) -> str:
        """
        Predict product SMILES from reactants.

        Args:
            reactants: Reactant SMILES (e.g., "CC(=O)O.CCO")
            beam_size: Override default beam size

        Returns:
            Predicted product SMILES
        """
        predictions = self.predict_batch(
            [reactants],
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
        return predictions[0]

    def predict_batch(
        self,
        reactants_list: List[str],
        beam_size: Optional[int] = None,
        length_penalty: Optional[float] = None,
    ) -> List[str]:
        """
        Predict products for a batch of reactions.

        Args:
            reactants_list: List of reactant SMILES
            beam_size: Override default beam size

        Returns:
            List of predicted product SMILES
        """
        if not reactants_list:
            return []

        beam_size = beam_size or self.beam_size
        length_penalty = self.length_penalty if length_penalty is None else length_penalty
        all_predictions = []

        # Process in batches
        for i in range(0, len(reactants_list), self.batch_size):
            batch = reactants_list[i:i + self.batch_size]
            predictions = self._predict_batch_single(batch, beam_size, length_penalty)
            all_predictions.extend(predictions)

            # Clear cache on CPU to avoid memory buildup
            if self.device == "cpu":
                gc.collect()

        return all_predictions

    @torch.inference_mode()
    def _predict_batch_single(
        self,
        reactants_list: List[str],
        beam_size: int,
        length_penalty: float,
    ) -> List[str]:
        """Predict a single batch without chunking."""
        if self.model_type == "decoder_only":
            encoded_list = []
            for reactants in reactants_list:
                encoded = self.tokenizer.encode_reaction(
                    reactants=reactants,
                    products=None,
                    task=self.task,
                    model_type="decoder_only",
                    max_length=self.max_len,
                    max_src_length=self.max_len,
                    truncation=True,
                    return_tensors=None,
                )
                encoded_list.append(encoded)

            max_len = max(len(e["input_ids"]) for e in encoded_list)
            input_ids = torch.full((len(encoded_list), max_len), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(encoded_list), max_len), dtype=torch.bool, device=self.device)
            for i, enc in enumerate(encoded_list):
                seq = enc["input_ids"]
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                attention_mask[i, :len(seq)] = True
        else:
            # Encode inputs using reaction-format prompt
            encoded_list = []
            for reactants in reactants_list:
                encoded = self.tokenizer.encode_reaction(
                    reactants=reactants,
                    products=None,
                    task=self.task,
                    model_type="transformer",
                    max_src_length=self.max_len,
                    truncation=True,
                    return_tensors=None,
                )
                encoded_list.append(encoded["src"])

            max_len = max(len(seq) for seq in encoded_list)
            src = torch.full((len(encoded_list), max_len), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
            src_mask = torch.zeros((len(encoded_list), max_len), dtype=torch.bool, device=self.device)
            for i, seq in enumerate(encoded_list):
                src[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)
                src_mask[i, :len(seq)] = True

        # Generate
        if self.model_type == "decoder_only":
            if beam_size > 1:
                sequences, _ = self.model.parallel_beam_search(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    eos_id=self.tokenizer.eos_id,
                )
            else:
                sequences = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_len,
                    top_k=1,
                    eos_id=self.tokenizer.eos_id,
                )
            predictions = []
            for seq in sequences:
                seq_ids = seq.tolist()
                if self.tokenizer.sep_id in seq_ids:
                    seq_ids = seq_ids[seq_ids.index(self.tokenizer.sep_id) + 1:]
                predictions.append(
                    self.tokenizer.decode(seq_ids, skip_special=True, stop_at_eos=True)
                )
        else:
            if beam_size > 1:
                # Beam search (one at a time for simplicity)
                predictions = []
                for i in range(len(reactants_list)):
                    seq, _ = self.model.beam_search(
                        src[i:i+1],
                        src_mask[i:i+1],
                        max_len=self.max_len,
                        beam_size=beam_size,
                        length_penalty=length_penalty,
                    )
                    pred = self.tokenizer.decode(seq[0].tolist(), skip_special=True, stop_at_eos=True)
                    predictions.append(pred)
            else:
                # Greedy decoding (batched)
                sequences = self.model.generate(
                    src,
                    src_mask,
                    max_len=self.max_len,
                    top_k=1,
                )
                predictions = [
                    self.tokenizer.decode(seq.tolist(), skip_special=True, stop_at_eos=True)
                    for seq in sequences
                ]

        return predictions

    def __repr__(self) -> str:
        return (
            f"RxnPredictor("
            f"vocab_size={self.tokenizer.vocab_size}, "
            f"device={self.device}, "
            f"batch_size={self.batch_size})"
        )


def load_predictor(
    model_dir: str,
    device: str = "cuda",
    **kwargs,
) -> RxnPredictor:
    """
    Convenience function to load predictor.

    Args:
        model_dir: Path to model directory
        device: Device to use
        **kwargs: Additional arguments for RxnPredictor

    Returns:
        RxnPredictor instance
    """
    return RxnPredictor(model_dir=model_dir, device=device, **kwargs)
