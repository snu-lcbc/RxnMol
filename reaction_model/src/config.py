"""Configuration dataclasses and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


# =============================================================================
# Dataclass Schemas
# =============================================================================


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Model type
    type: str = "transformer"  # transformer | decoder_only

    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2  # For GQA: num_groups = n_heads // n_kv_heads
    n_layers: int = 8
    d_ff: Optional[int] = None  # If None, computed as 8/3 * d_model for SwiGLU
    max_seq_len: int = 512
    dropout: float = 0.1

    # Vocabulary (set from tokenizer)
    vocab_size: int = 5000

    # Special token ids (set from tokenizer)
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    sep_id: int = 4  # Separator token for instruction format

    # RoPE settings
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_ff is None:
            # SwiGLU uses 8/3 * d_model to match parameter count of 4 * d_model FFN
            self.d_ff = int(8 * self.d_model / 3)
            # Round to multiple of 64 for efficiency
            self.d_ff = ((self.d_ff + 63) // 64) * 64

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )


@dataclass
class TrainConfig:
    """Training configuration."""

    # Optimization
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    lr: float = 3e-4
    l2sp_lambda: float = 0.0
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-8

    # Scheduler
    warmup_steps: int = 1000
    max_steps: int = 100000
    scheduler: str = "cosine_restarts"  # cosine | cosine_restarts | linear | constant | reduce_on_plateau
    num_cycles: float = 2.0  # For cosine_restarts: number of restart cycles
    scheduler_patience: int = 10  # For reduce_on_plateau: epochs to wait before reducing LR
    scheduler_factor: float = 0.5  # For reduce_on_plateau: factor to reduce LR by

    # Training settings
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    precision: str = "bf16-mixed"  # "32", "16-mixed", "bf16-mixed"

    # Validation
    val_check_interval: Optional[float] = 500
    num_val_samples: int = 1000

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 50
    early_stopping_monitor: str = "val/loss"
    early_stopping_mode: str = "min"


@dataclass
class DataConfig:
    """Data configuration."""

    # Paths
    train_path: str = "data/flower_reaction_wise/train_clean_rxn.txt"
    val_path: str = "data/flower_reaction_wise/val_clean_rxn.txt"
    test_path: Optional[str] = None

    # Tokenizer
    tokenizer_path: Optional[str] = None  # Path to saved vocab (if needed)

    # Sequence settings
    max_src_len: int = 256
    max_tgt_len: int = 256

    # Caching settings (HuggingFace datasets)
    use_hf_cache: bool = True  # Use HF datasets for caching tokenized data
    cache_dir: str = ".cache/datasets"  # Directory to store cached datasets
    tokenize_num_proc: int = 8  # Number of processes for parallel tokenization

    # Auto-split settings (used when val_path is missing or invalid)
    auto_split: bool = False
    split_ratio: float = 0.1  # Fraction of data to use for validation
    split_seed: int = 42
    split_dir: Optional[str] = None  # Defaults to {logging.output_dir}/splits


@dataclass
class SpecialTokensConfig:
    """Tokenizer special tokens."""

    pad: str = "<pad>"
    bos: str = "<s>"
    eos: str = "</s>"
    unk: str = "<unk>"
    sep: str = "<sep>"


@dataclass
class TokenizerConfig:
    """Tokenizer and formatting settings."""

    path: Optional[str] = "tokenizers/tokenizer.json"  # Path to existing tokenizer
    task: str = "forward"  # forward | retro | elem | mech
    instruction: str = "Predict the products of this reaction"

    # Auto-train settings
    auto_train: bool = False
    use_fast: bool = True
    train_min_frequency: int = 1
    train_num_proc: int = 4

    # Formatting flags (for plain SMILES experiments)
    use_task_tokens: bool = True
    use_role_tokens: bool = True

    # Special tokens
    special_tokens: SpecialTokensConfig = field(default_factory=SpecialTokensConfig)


@dataclass
class SpeculativeConfig:
    """Speculative decoding settings."""

    enabled: bool = False
    draft_layers: int = 2
    gamma: int = 5


@dataclass
class InferenceConfig:
    """Inference settings."""

    strategy: str = "beam"  # greedy | beam | speculative
    batch_size: int = 128
    max_len: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    # Beam search
    beam_size: int = 5
    length_penalty: float = 1.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0

    # Speculative decoding
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)

    # torch.compile()
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_backend: str = "inductor"


@dataclass
class FSDPConfig:
    """FSDP distributed training settings."""

    sharding: str = "FULL_SHARD"  # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD | HYBRID_SHARD
    cpu_offload: bool = False
    activation_checkpointing: bool = True
    backward_prefetch: str = "BACKWARD_PRE"


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    strategy: str = "ddp"  # auto | ddp | fsdp
    devices: int = 2
    num_nodes: int = 1
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""

    output_dir: str = "runs"
    experiment_name: str = "flower-clean-rxn"

    # Optional run directory management
    run_dir: Optional[str] = None  # Explicit full run directory (overrides output_dir/experiment_name)
    create_run_dir: bool = False  # If true, append run_name or timestamp
    run_name: Optional[str] = None  # If None and create_run_dir, use timestamp_format
    timestamp_format: str = "%Y%m%d_%H%M%S"

    wandb: bool = False
    wandb_project: str = "reaction-prediction"
    log_every_n_steps: int = 30
    val_check_interval: Optional[float] = None
    save_top_k: int = 20


@dataclass
class HardwareConfig:
    """Hardware configuration."""

    accelerator: str = "auto"  # gpu | cpu | auto
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 42


@dataclass
class RuntimeConfig:
    """Runtime controls (not training hyperparameters)."""

    resume_from: Optional[str] = None
    init_from: Optional[str] = None
    init_strict: bool = True
    check_model_config: bool = True
    load_model_config: bool = False
    copy_init_to_output: bool = True
    freeze_n_layers: int = 0
    freeze_embeddings: bool = False


@dataclass
class AppConfig:
    """Top-level configuration container."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def app_config_to_dict(app_config: AppConfig) -> Dict[str, Any]:
    """Convert AppConfig dataclass to a plain dictionary."""
    return asdict(app_config)


def save_resolved_config(
    output_dir: Path,
    app_config: AppConfig,
    config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> None:
    """Save fully-resolved config (YAML + JSON) and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_dict = app_config_to_dict(app_config)
    resolved_cfg = OmegaConf.create(resolved_dict)

    yaml_path = output_dir / "config.resolved.yaml"
    json_path = output_dir / "config.resolved.json"
    meta_path = output_dir / "config.meta.json"

    yaml_path.write_text(OmegaConf.to_yaml(resolved_cfg, resolve=True))
    json_path.write_text(json.dumps(resolved_dict, indent=2))

    overrides_list = overrides or []
    if overrides_list and not isinstance(overrides_list, list):
        try:
            overrides_list = OmegaConf.to_container(overrides_list, resolve=False)
        except Exception:
            overrides_list = list(overrides_list)
    meta = {
        "config_path": str(config_path) if config_path else None,
        "overrides": overrides_list,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def resolve_output_dir(logging_config: LoggingConfig) -> Path:
    """Resolve output directory based on logging config."""
    if logging_config.run_dir:
        return Path(logging_config.run_dir)

    base = Path(logging_config.output_dir)
    exp = logging_config.experiment_name

    if logging_config.create_run_dir:
        run_name = logging_config.run_name
        if not run_name:
            from datetime import datetime

            run_name = datetime.now().strftime(logging_config.timestamp_format)
        return base / exp / run_name

    return base / exp
