#!/usr/bin/env python3
"""
Evaluation script for reaction prediction models.

Computes:
- Sequence-level accuracy (exact match, top-1)
- Token-level accuracy (placeholder)
- Invalid SMILES rate
- Tanimoto similarity (Morgan fingerprint, radius=2, nBits=2048)


Top-k accuracies: Canonicalized exact match within top-k beam predictions
Exact match (raw): String-level match of top-1 prediction vs target
Exact match (canon): Chemical structure match of top-1 vs target (same as Top-1 Accuracy)
Tanimoto mean: Average molecular similarity of top-1 predictions vs targets

Usage:
    python evaluate.py --checkpoint path/to/checkpoint.ckpt --data path/to/test.txt
    python evaluate.py --checkpoint path/to/checkpoint.ckpt --data path/to/test.txt --beam-size 10
"""

import argparse
import json
import math
import os
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
from dataclasses import dataclass, asdict

from src.tokenizer import SMILESTokenizer, tokenize_smiles
from src.lightning_module import ReactionModule
from src.config import ModelConfig, TrainConfig

# Try to import RDKit at module level for performance
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    DataStructs = None
    AllChem = None
    RDKIT_AVAILABLE = False


@dataclass
class EvalMetrics:
    """Evaluation metrics."""
    total_samples: int = 0
    top1_accuracy: float = 0.0
    top3_accuracy: Optional[float] = None
    top5_accuracy: Optional[float] = None
    top10_accuracy: Optional[float] = None
    exact_match_accuracy: float = 0.0
    canonicalized_match_accuracy: Optional[float] = None
    bleu: Optional[float] = None
    validity: Optional[float] = None
    token_accuracy: float = 0.0
    invalid_smiles_rate: float = 0.0
    avg_sequence_length: float = 0.0
    tanimoto_mean: Optional[float] = None
    tanimoto_valid_rate: Optional[float] = None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize SMILES string using RDKit.
    Returns None if SMILES is invalid.
    """
    if not RDKIT_AVAILABLE:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _bleu_update(
    pred: str,
    ref: str,
    max_n: int,
    clipped_counts: List[int],
    total_counts: List[int],
    length_acc: Dict[str, int],
) -> None:
    pred_tokens = tokenize_smiles(pred)
    ref_tokens = tokenize_smiles(ref)
    length_acc["pred"] += len(pred_tokens)
    length_acc["ref"] += len(ref_tokens)

    for n in range(1, max_n + 1):
        pred_ngrams: Dict[Tuple[str, ...], int] = {}
        for i in range(len(pred_tokens) - n + 1):
            ng = tuple(pred_tokens[i:i + n])
            pred_ngrams[ng] = pred_ngrams.get(ng, 0) + 1
        ref_ngrams: Dict[Tuple[str, ...], int] = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        clipped = 0
        for ng, count in pred_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))
        clipped_counts[n - 1] += clipped
        total_counts[n - 1] += max(len(pred_tokens) - n + 1, 0)


def compute_bleu(
    clipped_counts: List[int],
    total_counts: List[int],
    length_acc: Dict[str, int],
    max_n: int = 4,
    smooth: float = 1e-9,
) -> float:
    if length_acc["pred"] == 0:
        return 0.0

    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(smooth)
        else:
            p_n = clipped_counts[n] / total_counts[n]
            precisions.append(max(p_n, smooth))

    bp = 1.0 if length_acc["pred"] > length_acc["ref"] else math.e ** (
        1 - length_acc["ref"] / max(length_acc["pred"], 1)
    )
    log_p = sum((1.0 / max_n) * math.log(p) for p in precisions)
    return bp * math.exp(log_p)


def _resolve_experiment_dir(checkpoint_path: Path, hparams: Dict[str, Any]) -> Optional[Path]:
    logging_cfg = hparams.get("logging")
    if isinstance(logging_cfg, dict):
        output_dir = logging_cfg.get("output_dir")
        experiment_name = logging_cfg.get("experiment_name")
        if output_dir and experiment_name:
            return Path(output_dir) / experiment_name
    return None


def tanimoto_similarity(smiles_a: Optional[str], smiles_b: Optional[str]) -> Optional[float]:
    """
    Compute Tanimoto similarity using Morgan fingerprint (radius=2, nBits=2048).
    Returns None if RDKit is unavailable or either SMILES is invalid.
    """
    if not RDKIT_AVAILABLE or smiles_a is None or smiles_b is None:
        return None
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def load_test_data(
    path: str,
    max_samples: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Load test data from file.

    Supports formats:
    - SMILES>>SMILES
    - SMILES>>SMILES|ID

    Returns:
        List of (reactants, products) tuples
    """
    samples = []
    path = Path(path)

    if path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        # Try common column names
        for col in ["updated_reaction", "reaction", "rxn", "REACTION"]:
            if col in df.columns:
                for rxn in df[col].dropna():
                    rxn = str(rxn).strip()
                    if ">>" in rxn:
                        parts = rxn.split(">>", 1)
                        if len(parts) == 2 and parts[0] and parts[1]:
                            samples.append((parts[0], parts[1]))
                break
    else:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Remove ID suffix if present
                if "|" in line:
                    line = line.rsplit("|", 1)[0]
                if ">>" not in line:
                    continue
                parts = line.split(">>", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    samples.append((parts[0], parts[1]))

    if max_samples:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} test samples from {path}")
    return samples


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[ReactionModule, SMILESTokenizer, str, Optional[Path]]:
    """
    Load model from checkpoint.

    Returns:
        (model, tokenizer, model_type, experiment_dir)
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    hparams = checkpoint.get("hyper_parameters", {})
    experiment_dir = _resolve_experiment_dir(checkpoint_path, hparams)

    # Find tokenizer
    # Prefer logging.output_dir/experiment_name when available
    tokenizer_paths = []
    if experiment_dir:
        tokenizer_paths.append(experiment_dir / "tokenizer.json")
    tokenizer_paths.extend([
        checkpoint_path.parent.parent / "tokenizer.json",
        checkpoint_path.parent / "tokenizer.json",
        checkpoint_path.parent.parent.parent / "tokenizer.json",
    ])

    tokenizer_path = None
    for tp in tokenizer_paths:
        if tp.exists():
            tokenizer_path = tp
            break

    if tokenizer_path is None:
        raise FileNotFoundError(f"Could not find tokenizer.json near {checkpoint_path}")

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SMILESTokenizer.from_pretrained(str(tokenizer_path))

    # Get hyperparameters
    model_config = checkpoint.get("model_config")
    train_config = checkpoint.get("train_config")
    model_type = checkpoint.get("model_type", "transformer")

    if model_config is None or train_config is None:
        if "model_config" in hparams or "train_config" in hparams:
            model_config = hparams.get("model_config", model_config)
            train_config = hparams.get("train_config", train_config)
            model_type = hparams.get("model_type", model_type)
        else:
            model_config = hparams.get("model", model_config)
            train_config = hparams.get("training", train_config)
            model_type = hparams.get("model", {}).get("type", model_type)

    if model_type in {"encoder_decoder", "seq2seq"}:
        model_type = "transformer"

    # Create configs
    if isinstance(model_config, dict):
        model_config = ModelConfig(**{k: v for k, v in model_config.items() if k != "type"})
    if train_config is None:
        train_config = TrainConfig()
    elif isinstance(train_config, dict):
        train_config = TrainConfig(**train_config)

    # Update vocab size from tokenizer
    model_config.vocab_size = len(tokenizer)
    model_config.pad_id = getattr(tokenizer, "pad_id", model_config.pad_id)
    model_config.bos_id = getattr(tokenizer, "bos_id", model_config.bos_id)
    model_config.eos_id = getattr(tokenizer, "eos_id", model_config.eos_id)

    # Create model
    module = ReactionModule(
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        model_type=model_type,
    )

    # Load weights (support Lightning .ckpt and model.pt formats)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Lightning checkpoints store module weights with "model." prefix
        module.load_state_dict(state_dict)
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        # model.pt/best_model.pt store only model weights
        module.model.load_state_dict(state_dict)
    else:
        raise KeyError("Checkpoint missing state_dict or model_state_dict")
    module.to(device)
    module.eval()

    print(f"Loaded {model_type} model with {module.model.count_parameters():,} parameters")

    return module, tokenizer, model_type, experiment_dir


def _encode_batch_for_eval(
    tokenizer: SMILESTokenizer,
    reactants_batch: List[str],
    task: str,
    model_type: str,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[int]]]:
    """Encode a batch of reactants for evaluation."""
    # Encode each sample
    encoded_list = []
    prompt_lengths = None
    for reactants in reactants_batch:
        encoded = tokenizer.encode_reaction(
            reactants=reactants,
            products=None,
            task=task,
            model_type=model_type,
            return_tensors=None,  # Return lists, not tensors
        )
        if model_type == "transformer":
            encoded_list.append(encoded["src"])
        else:
            encoded_list.append(encoded["input_ids"])

    if model_type == "decoder_only":
        prompt_lengths = [len(seq) for seq in encoded_list]

    # Pad to max length in batch
    max_len = max(len(seq) for seq in encoded_list)
    pad_id = tokenizer.pad_id

    batch_size = len(encoded_list)
    src = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    src_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, seq in enumerate(encoded_list):
        src[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        src_mask[i, :len(seq)] = True

    return src.to(device), src_mask.to(device), prompt_lengths


@torch.no_grad()
def evaluate_beam_search(
    module: ReactionModule,
    tokenizer: SMILESTokenizer,
    test_data: List[Tuple[str, str]],
    model_type: str,
    beam_size: int = 5,
    max_len: int = 256,
    length_penalty: float = 1.0,
    task: str = "forward",
    batch_size: int = 64,
    device: str = "cuda",
    use_canonical: bool = True,
    verbose: bool = False,
) -> EvalMetrics:
    """
    Evaluate model using beam search decoding with batching.

    Args:
        module: Trained model
        tokenizer: Tokenizer
        test_data: List of (reactants, products) tuples
        model_type: "transformer" or "decoder_only"
        beam_size: Number of beams
        max_len: Maximum generation length
        length_penalty: Length penalty for beam search scoring
        task: Task type for encoding
        batch_size: Batch size for evaluation (default 64)
        device: Device to use
        use_canonical: Canonicalize SMILES before comparison

    Returns:
        EvalMetrics dataclass with all metrics
    """
    model = module.model

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    correct_top10 = 0
    exact_match = 0
    canon_match = 0
    invalid_count = 0
    valid_count = 0
    total_length = 0
    tanimoto_sum = 0.0
    tanimoto_count = 0

    bleu_max_n = 4
    bleu_clipped = [0] * bleu_max_n
    bleu_total = [0] * bleu_max_n
    bleu_len = {"pred": 0, "ref": 0}

    n_samples = len(test_data)
    n_batches = (n_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(n_batches), desc="Evaluating")

    for batch_idx in pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = test_data[start_idx:end_idx]

        # Extract reactants and targets
        reactants_batch = [r for r, _ in batch_data]
        targets_raw = [t.strip() for _, t in batch_data]
        targets_batch = targets_raw

        # Pre-canonicalize targets
        if use_canonical:
            targets_batch = [canonicalize_smiles(t) or t for t in targets_raw]

        # Encode batch
        src, src_mask, prompt_lengths = _encode_batch_for_eval(
            tokenizer, reactants_batch, task, model_type, device
        )

        # Generate with parallel beam search (batched)
        all_sequences = None
        if model_type == "transformer":
            if hasattr(model, "parallel_beam_search"):
                sequences, scores, all_sequences, _ = model.parallel_beam_search(
                    src=src,
                    src_mask=src_mask,
                    max_len=max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    return_all=True,
                )
            elif hasattr(model, "beam_search"):
                # Fallback to regular beam search
                sequences, scores = model.beam_search(
                    src=src,
                    src_mask=src_mask,
                    max_len=max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                )
            else:
                sequences = model.generate(
                    src=src,
                    src_mask=src_mask,
                    max_len=max_len,
                )
                scores = torch.zeros(len(batch_data))
        else:
            # Decoder-only model
            if hasattr(model, "parallel_beam_search"):
                sequences, scores, all_sequences, _ = model.parallel_beam_search(
                    input_ids=src,
                    attention_mask=src_mask,
                    max_new_tokens=max_len,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    return_all=True,
                )
            else:
                sequences = model.generate(
                    src,
                    attention_mask=src_mask,
                    max_new_tokens=max_len,
                )
                scores = torch.zeros(len(batch_data))

        # Process each sample in batch
        for i, (reactants, target) in enumerate(zip(reactants_batch, targets_batch)):
            if sequences.dim() == 1:
                seq_ids = sequences.tolist()
            else:
                seq_ids = sequences[i].tolist()

            if model_type == "decoder_only" and prompt_lengths is not None:
                seq_ids = seq_ids[prompt_lengths[i]:]

            pred_raw = tokenizer.decode(seq_ids, skip_special=True).strip()

            total_length += len(pred_raw)
            _bleu_update(pred_raw, targets_raw[i], bleu_max_n, bleu_clipped, bleu_total, bleu_len)

            # Verbose output
            if verbose and start_idx + i < 10:
                print(f"\n[Sample {start_idx + i}]")
                print(f"  Input:  {reactants[:80]}...")
                print(f"  Target: {target[:80]}...")
                print(f"  Pred:   {pred_raw[:80]}...")

            # Canonicalize prediction (once)
            pred_for_eval = pred_raw
            if use_canonical:
                canonical_pred = canonicalize_smiles(pred_raw)
                if canonical_pred is None:
                    invalid_count += 1
                    pred_for_eval = None
                else:
                    pred_for_eval = canonical_pred

            if pred_for_eval is not None:
                valid_count += 1
                if use_canonical and pred_for_eval == target:
                    canon_match += 1

            sim = tanimoto_similarity(pred_for_eval, target)
            if sim is not None:
                tanimoto_sum += sim
                tanimoto_count += 1

            if pred_for_eval is None:
                continue

            # Compare (top-1 only with batched beam search)
            if pred_for_eval == target:
                correct_top1 += 1
            if pred_raw == targets_raw[i]:
                exact_match += 1

            if all_sequences is not None:
                beams = all_sequences[i]
                topk_preds = []
                for seq in beams:
                    seq_ids = seq.tolist()
                    if model_type == "decoder_only" and prompt_lengths is not None:
                        seq_ids = seq_ids[prompt_lengths[i]:]
                    pred_k = tokenizer.decode(seq_ids, skip_special=True).strip()
                    if use_canonical:
                        pred_k = canonicalize_smiles(pred_k) or pred_k
                    topk_preds.append(pred_k)

                if target in topk_preds[:3]:
                    correct_top3 += 1
                if target in topk_preds[:5]:
                    correct_top5 += 1
                if target in topk_preds[:10]:
                    correct_top10 += 1

        # Update progress bar
        processed = end_idx
        pbar.set_postfix({
            "top1": f"{correct_top1/processed:.1%}",
        })

    return EvalMetrics(
        total_samples=n_samples,
        top1_accuracy=correct_top1 / n_samples if n_samples > 0 else 0,
        top3_accuracy=(correct_top3 / n_samples if n_samples > 0 and beam_size >= 3 else None),
        top5_accuracy=(correct_top5 / n_samples if n_samples > 0 and beam_size >= 5 else None),
        top10_accuracy=(correct_top10 / n_samples if n_samples > 0 and beam_size >= 10 else None),
        exact_match_accuracy=exact_match / n_samples if n_samples > 0 else 0,
        canonicalized_match_accuracy=canon_match / n_samples if (n_samples > 0 and use_canonical) else None,
        bleu=compute_bleu(bleu_clipped, bleu_total, bleu_len, max_n=bleu_max_n),
        validity=(valid_count / n_samples) if n_samples > 0 else None,
        token_accuracy=0.0,
        invalid_smiles_rate=invalid_count / n_samples if n_samples > 0 else 0,
        avg_sequence_length=total_length / n_samples if n_samples > 0 else 0,
        tanimoto_mean=(tanimoto_sum / tanimoto_count) if tanimoto_count > 0 else None,
        tanimoto_valid_rate=(tanimoto_count / n_samples) if n_samples > 0 else None,
    )


@torch.no_grad()
def evaluate_greedy(
    module: ReactionModule,
    tokenizer: SMILESTokenizer,
    test_data: List[Tuple[str, str]],
    model_type: str,
    max_len: int = 256,
    task: str = "forward",
    batch_size: int = 64,
    device: str = "cuda",
    use_canonical: bool = True,
    verbose: bool = False,
) -> EvalMetrics:
    """
    Evaluate model using greedy decoding with batching (faster, top-1 only).
    """
    model = module.model

    correct = 0
    exact_match = 0
    canon_match = 0
    invalid_count = 0
    valid_count = 0
    total_length = 0
    tanimoto_sum = 0.0
    tanimoto_count = 0

    bleu_max_n = 4
    bleu_clipped = [0] * bleu_max_n
    bleu_total = [0] * bleu_max_n
    bleu_len = {"pred": 0, "ref": 0}

    n_samples = len(test_data)
    n_batches = (n_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(n_batches), desc="Evaluating (greedy)")

    for batch_idx in pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = test_data[start_idx:end_idx]

        # Extract reactants and targets
        reactants_batch = [r for r, _ in batch_data]
        targets_raw = [t.strip() for _, t in batch_data]
        targets_batch = targets_raw

        # Pre-canonicalize targets
        if use_canonical:
            targets_batch = [canonicalize_smiles(t) or t for t in targets_raw]

        # Encode batch
        src, src_mask, prompt_lengths = _encode_batch_for_eval(
            tokenizer, reactants_batch, task, model_type, device
        )

        # Generate (batched greedy)
        if model_type == "transformer":
            sequences = model.generate(
                src=src,
                src_mask=src_mask,
                max_len=max_len,
                temperature=0.0,  # Greedy
            )
        else:
            sequences = model.generate(
                src,
                attention_mask=src_mask,
                max_new_tokens=max_len,
                temperature=0.0,
            )

        # Process each sample in batch
        for i, (reactants, target) in enumerate(zip(reactants_batch, targets_batch)):
            seq_ids = sequences[i].tolist()
            if model_type == "decoder_only" and prompt_lengths is not None:
                seq_ids = seq_ids[prompt_lengths[i]:]
            pred_raw = tokenizer.decode(seq_ids, skip_special=True).strip()
            total_length += len(pred_raw)
            _bleu_update(pred_raw, targets_raw[i], bleu_max_n, bleu_clipped, bleu_total, bleu_len)

            # Verbose output for first few samples
            if verbose and start_idx + i < 10:
                print(f"\n[Sample {start_idx + i}]")
                print(f"  Input:  {reactants[:80]}...")
                print(f"  Target: {target[:80]}...")
                print(f"  Pred:   {pred_raw[:80]}...")

            # Canonicalize prediction (once)
            is_correct = False
            pred_for_eval = pred_raw
            if use_canonical:
                canonical_pred = canonicalize_smiles(pred_raw)
                if canonical_pred is None:
                    invalid_count += 1
                    pred_for_eval = None
                else:
                    pred_for_eval = canonical_pred

            sim = tanimoto_similarity(pred_for_eval, target)
            if sim is not None:
                tanimoto_sum += sim
                tanimoto_count += 1

            if pred_for_eval is not None:
                valid_count += 1
                if use_canonical and pred_for_eval == target:
                    canon_match += 1

            if pred_for_eval is not None and pred_for_eval == target:
                correct += 1
            if pred_raw == targets_raw[i]:
                exact_match += 1
                is_correct = True

            if verbose and start_idx + i < 10:
                print(f"  Match:  {is_correct}")

        # Update progress
        processed = end_idx
        pbar.set_postfix({"acc": f"{correct/processed:.1%}"})

    n_samples = len(test_data)

    return EvalMetrics(
        total_samples=n_samples,
        top1_accuracy=correct / n_samples if n_samples > 0 else 0,
        exact_match_accuracy=exact_match / n_samples if n_samples > 0 else 0,
        canonicalized_match_accuracy=canon_match / n_samples if (n_samples > 0 and use_canonical) else None,
        bleu=compute_bleu(bleu_clipped, bleu_total, bleu_len, max_n=bleu_max_n),
        validity=(valid_count / n_samples) if n_samples > 0 else None,
        invalid_smiles_rate=invalid_count / n_samples if n_samples > 0 else 0,
        avg_sequence_length=total_length / n_samples if n_samples > 0 else 0,
        tanimoto_mean=(tanimoto_sum / tanimoto_count) if tanimoto_count > 0 else None,
        tanimoto_valid_rate=(tanimoto_count / n_samples) if n_samples > 0 else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate reaction prediction model")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", "-d", required=True, help="Path to test data file")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--max-len", type=int, default=256, help="Maximum generation length")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty for beam search")
    parser.add_argument("--task", default="forward", choices=["forward", "retro", "elem", "mech"])
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (faster)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--no-canonical", action="store_true", help="Skip SMILES canonicalization")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print sample predictions")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Load model
    module, tokenizer, model_type, experiment_dir = load_model(args.checkpoint, args.device)

    # Load test data
    test_data = load_test_data(args.data, args.max_samples)

    # Evaluate
    if args.greedy:
        metrics = evaluate_greedy(
            module=module,
            tokenizer=tokenizer,
            test_data=test_data,
            model_type=model_type,
            max_len=args.max_len,
            task=args.task,
            batch_size=args.batch_size,
            device=args.device,
            use_canonical=not args.no_canonical,
            verbose=args.verbose,
        )
    else:
        metrics = evaluate_beam_search(
            module=module,
            tokenizer=tokenizer,
            test_data=test_data,
            model_type=model_type,
            beam_size=args.beam_size,
            max_len=args.max_len,
            length_penalty=args.length_penalty,
            task=args.task,
            batch_size=args.batch_size,
            device=args.device,
            use_canonical=not args.no_canonical,
            verbose=args.verbose,
        )

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples:        {metrics.total_samples}")
    print(f"Top-1 Accuracy:       {metrics.top1_accuracy:.2%}")
    print(f"Exact Match (raw):    {metrics.exact_match_accuracy:.2%}")
    if metrics.canonicalized_match_accuracy is not None:
        print(f"Exact Match (canon):  {metrics.canonicalized_match_accuracy:.2%}")
    if metrics.bleu is not None:
        print(f"BLEU (SMILES tok):    {metrics.bleu:.4f}")
    if metrics.validity is not None:
        print(f"Validity:            {metrics.validity:.2%}")
    if not args.greedy:
        if metrics.top3_accuracy is not None:
            print(f"Top-3 Accuracy:       {metrics.top3_accuracy:.2%}")
        if metrics.top5_accuracy is not None:
            print(f"Top-5 Accuracy:       {metrics.top5_accuracy:.2%}")
        if metrics.top10_accuracy is not None:
            print(f"Top-10 Accuracy:      {metrics.top10_accuracy:.2%}")
        if metrics.top3_accuracy is None and metrics.top5_accuracy is None and metrics.top10_accuracy is None:
            print("Top-3/5/10 Accuracy:  N/A (beam search returns top-1 only)")
    print(f"Invalid SMILES Rate:  {metrics.invalid_smiles_rate:.2%}")
    print(f"Avg Sequence Length:  {metrics.avg_sequence_length:.1f}")
    if metrics.tanimoto_mean is not None:
        print(f"Tanimoto (r=2, 2048): {metrics.tanimoto_mean:.4f}")
        print(f"Tanimoto Valid Rate:  {metrics.tanimoto_valid_rate:.2%}")
    else:
        print("Tanimoto (r=2, 2048): N/A (RDKit unavailable or no valid pairs)")
    print("=" * 50)

    # Save results
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir() or str(args.output).endswith(("/", os.sep)) or output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = (output_path / Path(args.checkpoint).name).with_suffix(".json")
    else:
        eval_dir = (experiment_dir / "evals") if experiment_dir else (Path(args.checkpoint).parent / "evals")
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = (eval_dir / Path(args.checkpoint).name).with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "beam_size": args.beam_size if not args.greedy else 1,
        "length_penalty": args.length_penalty,
        "task": args.task,
        "metrics": asdict(metrics),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
