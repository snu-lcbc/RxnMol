#!/usr/bin/env python3
"""Inference script for reaction prediction model with multiple strategies."""

import argparse
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from omegaconf import OmegaConf

from src.config import AppConfig
from src.tokenizer import SMILESTokenizer
from src.lightning_module import ReactionModule
from src.transformer import Transformer
from src.decoder_model import CausalTransformer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with trained model")

    # YAML config (optional, for inference settings)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--override", nargs="*", default=[], help="Override config values (e.g., inference.max_len=128)")

    # Model paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt or .ckpt)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer (auto-detected if not specified)")

    # Input/output
    parser.add_argument("--input", type=str, default=None, help="Input SMILES or file path")
    parser.add_argument("--output", type=str, default=None, help="Output file path")

    # Inference strategy
    parser.add_argument("--strategy", type=str, default=None, choices=["greedy", "beam", "speculative"],
                        help="Inference strategy")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)

    # Beam search settings
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=None)

    # Speculative decoding settings
    parser.add_argument("--draft_layers", type=int, default=None)
    parser.add_argument("--gamma", type=int, default=None, help="Tokens to speculate per step")

    # torch.compile settings
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="Use torch.compile for inference")
    parser.add_argument("--compile_mode", type=str, default=None,
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile_backend", type=str, default=None,
                        help="torch.compile backend (e.g., inductor)")

    # Sampling settings
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args, unknown = parser.parse_known_args()

    # Treat unknown key=value pairs as Hydra-like overrides
    cli_overrides = []
    for item in unknown:
        if "=" in item and not item.startswith("--"):
            cli_overrides.append(item)
        else:
            parser.error(f"Unknown argument: {item}")

    args.override = (args.override or []) + cli_overrides

    return args


def load_inference_config(config_path: Optional[str], overrides: list[str]) -> AppConfig:
    """Load AppConfig from YAML and apply overrides (inference-only use)."""
    base_cfg = OmegaConf.structured(AppConfig)

    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        file_cfg = OmegaConf.load(path)
        file_cfg = OmegaConf.create(OmegaConf.to_container(file_cfg, resolve=False))
        file_cfg.pop("hydra", None)
        cfg = OmegaConf.merge(base_cfg, file_cfg)
    else:
        cfg = base_cfg

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)

    return OmegaConf.to_object(cfg)


def load_model(
    checkpoint_path: str,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
) -> tuple:
    """
    Load model and tokenizer from checkpoint.

    Supports both .pt (simple) and .ckpt (Lightning) checkpoints.
    """
    checkpoint_path = Path(checkpoint_path)

    # Auto-detect tokenizer path if not specified
    if tokenizer_path is None:
        parent_dir = checkpoint_path.parent
        # Try common locations
        for tok_path in [
            parent_dir / "tokenizer.json",
            parent_dir / "tokenizer",
            parent_dir.parent / "tokenizer.json",
            parent_dir.parent / "tokenizer",
        ]:
            if tok_path.exists():
                tokenizer_path = str(tok_path)
                print(f"Auto-detected tokenizer: {tokenizer_path}")
                break

    if tokenizer_path is None:
        raise ValueError("Could not auto-detect tokenizer path. Please specify with --tokenizer")

    # Load tokenizer (unified SMILESTokenizer handles both files and directories)
    tokenizer = SMILESTokenizer.from_pretrained(str(tokenizer_path))

    # Load model
    if str(checkpoint_path).endswith(".ckpt"):
        # Lightning checkpoint
        module = ReactionModule.load_from_checkpoint(
            str(checkpoint_path),
            tokenizer=tokenizer,
        )
    else:
        # Simple .pt checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = ckpt["model_config"]
        train_config = ckpt.get("train_config")
        model_type = ckpt.get("model_type", "encoder_decoder")
        if model_type in {"encoder_decoder", "seq2seq"}:
            model_type = "transformer"

        module = ReactionModule(
            model_config=model_config,
            train_config=train_config,
            tokenizer=tokenizer,
            model_type=model_type,
        )
        module.model.load_state_dict(ckpt["model_state_dict"])

    module = module.to(device)
    module.eval()

    return module, tokenizer


def setup_speculative_decoding(
    module: ReactionModule,
    draft_layers: int = 2,
    device: str = "cuda",
):
    """Create and setup draft model for speculative decoding."""
    model_config = module.model_config

    if module.model_type == "decoder_only":
        draft_model = CausalTransformer.create_draft_model(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_kv_heads=model_config.n_kv_heads,
            n_layers=draft_layers,
            d_ff=model_config.d_ff,
            max_seq_len=model_config.max_seq_len,
            pad_id=model_config.pad_id,
            bos_id=model_config.bos_id,
            eos_id=model_config.eos_id,
            rope_base=model_config.rope_base,
            share_embeddings=True,
            parent_model=module.model,
        )
    else:
        draft_model = Transformer.create_draft_model(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_kv_heads=model_config.n_kv_heads,
            d_ff=model_config.d_ff,
            max_seq_len=model_config.max_seq_len,
            pad_id=model_config.pad_id,
            bos_id=model_config.bos_id,
            eos_id=model_config.eos_id,
            rope_base=model_config.rope_base,
            num_layers=draft_layers,
            share_embeddings=True,
            parent_model=module.model,
        )

    draft_model = draft_model.to(device)
    draft_model.eval()

    return draft_model


@torch.inference_mode()
def predict_single(
    module: ReactionModule,
    tokenizer: SMILESTokenizer,
    smiles: str,
    strategy: str = "beam",
    max_len: int = 256,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    draft_model=None,
    gamma: int = 5,
    device: str = "cuda",
    instruction: str = "Predict the products of this reaction",
) -> str:
    """Predict products for a single reactant SMILES."""
    is_decoder_only = module.model_type == "decoder_only"

    if is_decoder_only:
        # Decoder-only prompt
        encoded = tokenizer.encode_reaction(
            reactants=smiles,
            products=None,  # No products for inference
            task=module.tokenizer_task if hasattr(module, "tokenizer_task") else "forward",
            model_type="decoder_only",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        if strategy == "speculative" and draft_model is not None:
            output = module.model.speculative_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                draft_model=draft_model,
                gamma=gamma,
                max_new_tokens=max_len,
                temperature=temperature,
                eos_id=module.model_config.eos_id,
            )
        elif strategy == "beam":
            output, _ = module.model.parallel_beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                eos_id=module.model_config.eos_id,
            )
        else:
            output = module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=module.model_config.eos_id,
            )

        # Decode - extract products after separator token (if present)
        seq = output[0].tolist()
        if tokenizer.sep_id in seq:
            seq = seq[seq.index(tokenizer.sep_id) + 1:]
        prediction = tokenizer.decode(seq, skip_special=True, stop_at_eos=True)
    else:
        # Encoder-decoder format
        encoded = tokenizer.encode_reaction(
            reactants=smiles,
            products=None,
            task=module.tokenizer_task if hasattr(module, "tokenizer_task") else "forward",
            model_type="transformer",
            max_src_length=max_len,
            truncation=True,
            return_tensors=None,
        )
        src_ids = encoded["src"]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = torch.ones_like(src, dtype=torch.bool)

        if strategy == "speculative" and draft_model is not None:
            output = module.model.speculative_generate(
                src=src,
                src_mask=src_mask,
                draft_model=draft_model,
                gamma=gamma,
                max_len=max_len,
                temperature=temperature,
                eos_id=module.model_config.eos_id,
            )
        elif strategy == "beam":
            output, _ = module.model.parallel_beam_search(
                src=src,
                src_mask=src_mask,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                eos_id=module.model_config.eos_id,
            )
        else:
            output = module.model.generate(
                src=src,
                src_mask=src_mask,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_id=module.model_config.eos_id,
            )

        prediction = tokenizer.decode(output[0].tolist(), skip_special=True, stop_at_eos=True)

    return prediction


def predict_file(
    module: ReactionModule,
    tokenizer: SMILESTokenizer,
    input_path: str,
    output_path: str,
    strategy: str = "beam",
    batch_size: int = 32,
    max_len: int = 256,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    draft_model=None,
    gamma: int = 5,
    device: str = "cuda",
    instruction: str = "Predict the products of this reaction",
):
    """Predict products for all reactions in a file."""
    # Read input SMILES
    smiles_list = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle reaction format (take reactants only)
            if ">>" in line:
                line = line.split(">>")[0]
            # Handle ID suffix
            if "|" in line:
                line = line.rsplit("|", 1)[0]
            smiles_list.append(line)

    print(f"Loaded {len(smiles_list)} reactions")

    # Predict
    predictions = []
    for i in tqdm(range(len(smiles_list)), desc=f"Predicting ({strategy})"):
        pred = predict_single(
            module, tokenizer, smiles_list[i],
            strategy=strategy,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            draft_model=draft_model,
            gamma=gamma,
            device=device,
            instruction=instruction,
        )
        predictions.append(pred)

    # Write output
    with open(output_path, "w") as f:
        for src, pred in zip(smiles_list, predictions):
            f.write(f"{src}>>{pred}\n")

    print(f"Predictions saved to {output_path}")

    return smiles_list, predictions


def interactive_mode(
    module: ReactionModule,
    tokenizer: SMILESTokenizer,
    strategy: str = "beam",
    max_len: int = 256,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    draft_model=None,
    gamma: int = 5,
    device: str = "cuda",
    instruction: str = "Predict the products of this reaction",
):
    """Interactive prediction mode."""
    print("\n=== Interactive Reaction Prediction ===")
    print(f"Strategy: {strategy}")
    print(f"Model type: {module.model_type}")
    print("Enter reactant SMILES to predict products.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            smiles = input("Reactants: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if smiles.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break

        if not smiles:
            continue

        try:
            prediction = predict_single(
                module, tokenizer, smiles,
                strategy=strategy,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                draft_model=draft_model,
                gamma=gamma,
                device=device,
                instruction=instruction,
            )
            print(f"Products:  {prediction}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main inference function."""
    args = parse_args()

    # Load inference config from YAML if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
    app_config = load_inference_config(args.config, args.override)

    tokenizer_config = app_config.tokenizer
    inference_config = app_config.inference

    instruction = tokenizer_config.instruction or "Predict the products of this reaction"

    # Use config values as defaults when CLI values are unset
    args.strategy = args.strategy or inference_config.strategy
    args.batch_size = inference_config.batch_size if args.batch_size is None else args.batch_size
    args.max_len = inference_config.max_len if args.max_len is None else args.max_len
    args.beam_size = inference_config.beam_size if args.beam_size is None else args.beam_size
    args.length_penalty = inference_config.length_penalty if args.length_penalty is None else args.length_penalty
    args.draft_layers = (
        inference_config.speculative.draft_layers
        if args.draft_layers is None
        else args.draft_layers
    )
    args.gamma = inference_config.speculative.gamma if args.gamma is None else args.gamma
    args.temperature = inference_config.temperature if args.temperature is None else args.temperature
    args.top_k = inference_config.top_k if args.top_k is None else args.top_k
    args.top_p = inference_config.top_p if args.top_p is None else args.top_p
    args.compile = inference_config.compile if args.compile is None else args.compile
    args.compile_mode = inference_config.compile_mode if args.compile_mode is None else args.compile_mode
    args.compile_backend = inference_config.compile_backend if args.compile_backend is None else args.compile_backend

    print(f"\nLoading model from {args.checkpoint}")
    module, tokenizer = load_model(
        args.checkpoint,
        args.tokenizer,
        device=args.device,
    )

    vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size
    print(f"Model loaded. Type: {module.model_type}, Vocab size: {vocab_size}")
    tokenizer.print_info()

    # Apply torch.compile if requested
    if args.compile:
        backend = args.compile_backend or "inductor"
        print(f"Compiling model with mode={args.compile_mode}, backend={backend}...")
        module.model.compile_for_inference(mode=args.compile_mode, backend=backend)
        print("Compilation complete.")

    # Setup speculative decoding if needed
    draft_model = None
    if args.strategy == "speculative":
        print(f"Setting up speculative decoding with {args.draft_layers} draft layers, gamma={args.gamma}")
        draft_model = setup_speculative_decoding(
            module, draft_layers=args.draft_layers, device=args.device
        )

    print(f"Inference strategy: {args.strategy}")

    if args.input is None:
        # Interactive mode
        interactive_mode(
            module, tokenizer,
            strategy=args.strategy,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            draft_model=draft_model,
            gamma=args.gamma,
            device=args.device,
            instruction=instruction,
        )
    elif Path(args.input).is_file():
        # File mode
        output_path = args.output or str(Path(args.input).with_suffix(".predictions.txt"))
        predict_file(
            module, tokenizer,
            input_path=args.input,
            output_path=output_path,
            strategy=args.strategy,
            batch_size=args.batch_size,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            draft_model=draft_model,
            gamma=args.gamma,
            device=args.device,
            instruction=instruction,
        )
    else:
        # Single SMILES
        prediction = predict_single(
            module, tokenizer, args.input,
            strategy=args.strategy,
            max_len=args.max_len,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            draft_model=draft_model,
            gamma=args.gamma,
            device=args.device,
            instruction=instruction,
        )
        print(f"Input:  {args.input}")
        print(f"Output: {prediction}")


if __name__ == "__main__":
    main()
