#!/usr/bin/env python3
"""Training script for reaction prediction model with YAML configuration."""

import os
import shutil
from pathlib import Path

import hydra
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    RichProgressBar,
    Callback,
)
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.config import (
    AppConfig,
    ModelConfig,
    TrainConfig,
    TokenizerConfig,
    DistributedConfig,
    app_config_to_dict,
    resolve_output_dir,
    save_resolved_config,
)
from src.tokenizer import SMILESTokenizer
from src.dataset import ReactionDataModule
from src.lightning_module import ReactionModule

torch.set_float32_matmul_precision('high')

# Allow custom config classes to be unpickled (required for PyTorch 2.6+)
torch.serialization.add_safe_globals([ModelConfig, TrainConfig])


@rank_zero_only
def rank_zero_print(message: str) -> None:
    print(message)


def _get_cfg_value(cfg, key: str):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(key)
    return getattr(cfg, key, None)


def _load_init_checkpoint(init_from: str) -> dict:
    init_path = Path(init_from)
    if not init_path.exists():
        raise FileNotFoundError(f"init_from not found: {init_path}")
    checkpoint = torch.load(init_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format at {init_path}")
    return checkpoint


def _apply_model_config_from_init(
    base_cfg: ModelConfig,
    init_cfg,
) -> ModelConfig:
    fields_to_copy = [
        "type",
        "d_model",
        "n_heads",
        "n_kv_heads",
        "n_layers",
        "d_ff",
        "rope_base",
    ]
    base_dict = base_cfg.__dict__.copy()
    changes = []
    for field in fields_to_copy:
        init_val = _get_cfg_value(init_cfg, field)
        if init_val is None:
            continue
        if base_dict.get(field) != init_val:
            changes.append(f"{field}: {base_dict.get(field)} -> {init_val}")
        base_dict[field] = init_val

    if changes:
        rank_zero_print("Loaded model config from init checkpoint: " + "; ".join(changes))

    return ModelConfig(**base_dict)


def _check_model_config_compat(init_cfg, model_cfg: ModelConfig) -> None:
    if init_cfg is None:
        return
    fields = ["type", "d_model", "n_heads", "n_kv_heads", "n_layers", "d_ff", "vocab_size"]
    mismatches = []
    for field in fields:
        init_val = _get_cfg_value(init_cfg, field)
        current_val = getattr(model_cfg, field, None)
        if init_val is None or current_val is None:
            continue
        if init_val != current_val:
            mismatches.append(f"{field}: init={init_val} current={current_val}")
    if mismatches:
        raise ValueError(
            "Model config mismatch between init checkpoint and current config: "
            + "; ".join(mismatches)
        )


def _load_init_weights(
    init_from: str,
    module: ReactionModule,
    model_config: ModelConfig,
    strict: bool,
    check_model_config: bool,
    checkpoint: dict | None = None,
) -> None:
    if checkpoint is None:
        rank_zero_print(f"Loading init weights from {init_from}")
        checkpoint = _load_init_checkpoint(init_from)
    else:
        rank_zero_print(f"Loading init weights from {init_from} (cached)")

    if check_model_config:
        _check_model_config_compat(checkpoint.get("model_config"), model_config)

    target = None
    state_dict = None
    target_name = None
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        target = module.model
        target_name = "module.model"
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        target = module
        target_name = "module"
    else:
        state_dict = checkpoint
        if any(k.startswith("model.") for k in state_dict.keys()):
            target = module
            target_name = "module"
        else:
            target = module.model
            target_name = "module.model"

    incompatible = target.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        rank_zero_print(f"Missing keys when loading init weights: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        rank_zero_print(f"Unexpected keys when loading init weights: {incompatible.unexpected_keys}")
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise ValueError("init_from strict load failed due to missing/unexpected keys.")

    rank_zero_print(f"Init weights loaded into {target_name}")


def _apply_freeze(module: ReactionModule, freeze_n_layers: int, freeze_embeddings: bool) -> None:
    if freeze_n_layers <= 0 and not freeze_embeddings:
        return

    model = module.model
    if freeze_embeddings and hasattr(model, "embed"):
        for param in model.embed.parameters():
            param.requires_grad = False
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = False

    if freeze_n_layers > 0:
        if hasattr(model, "encoder") and hasattr(model, "decoder"):
            encoder_layers = getattr(model.encoder, "layers", [])
            decoder_layers = getattr(model.decoder, "layers", [])
            max_freeze = min(freeze_n_layers, len(encoder_layers), len(decoder_layers))
            for layer in encoder_layers[:max_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
            for layer in decoder_layers[:max_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
            rank_zero_print(f"Froze {max_freeze} encoder/decoder layers")
        elif hasattr(model, "layers"):
            layers = getattr(model, "layers", [])
            max_freeze = min(freeze_n_layers, len(layers))
            for layer in layers[:max_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
            rank_zero_print(f"Froze {max_freeze} layers")
        else:
            raise ValueError("freeze_n_layers requested but model does not expose layers")


def _count_trainable_params(module: ReactionModule) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@rank_zero_only
def _save_tokenizer_assets(tokenizer: SMILESTokenizer, tokenizer_path: Path) -> None:
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved to {tokenizer_path}")
    tokenizer.print_info()


class SaveBestModelCallback(Callback):
    """Save best model as model.pt and best_model.pt for easy loading."""

    def __init__(self, output_dir: Path, monitor: str = "val/loss", every_n_steps: int = 1000):
        self.output_dir = output_dir
        self.monitor = monitor
        self.every_n_steps = every_n_steps
        self.best_score = float("inf")

    def state_dict(self):
        return {"best_score": self.best_score}

    def load_state_dict(self, state_dict):
        self.best_score = state_dict.get("best_score", self.best_score)

    def _maybe_save(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        if torch.is_tensor(current):
            current = current.item()
        current = float(current)

        if current < self.best_score:
            self.best_score = current
            model_path = self.output_dir / "best_model.pt"
            torch.save({
                "model_state_dict": pl_module.model.state_dict(),
                "model_config": pl_module.model_config,
                "train_config": pl_module.train_config,
                "model_type": pl_module.model_type,
                "best_score": current,
                "step": trainer.global_step,
            }, model_path)
            rank_zero_print(f"\n Saved best model to {model_path} (score: {current:.4f})")

    def on_validation_end(self, trainer, pl_module):
        if self.monitor.startswith("val/"):
            self._maybe_save(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.monitor.startswith("train/"):
            return
        if not self.every_n_steps:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return
        self._maybe_save(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.monitor.startswith("train/"):
            self._maybe_save(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        model_path = self.output_dir / "model.pt"
        torch.save({
            "model_state_dict": pl_module.model.state_dict(),
            "model_config": pl_module.model_config,
            "train_config": pl_module.train_config,
            "model_type": pl_module.model_type,
            "final_step": trainer.global_step,
        }, model_path)
        rank_zero_print(f"\n Saved final model to {model_path}")


def build_tokenizer(
    tokenizer_config: TokenizerConfig,
    output_dir: Path,
) -> SMILESTokenizer:
    """Load pretrained tokenizer and stage it in the output directory."""
    tokenizer_path = output_dir / "tokenizer.json"
    config_path = Path(tokenizer_config.path) if tokenizer_config.path else None

    if config_path:
        if not config_path.exists():
            raise FileNotFoundError(f"tokenizer.path not found: {config_path}")
        rank_zero_print(f"Loading tokenizer from {config_path}")
        tokenizer = SMILESTokenizer.from_pretrained(str(config_path))
    elif tokenizer_path.exists():
        rank_zero_print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = SMILESTokenizer.from_pretrained(str(tokenizer_path))
    else:
        raise FileNotFoundError(
            "Tokenizer not found. Set tokenizer.path in config.yaml or provide "
            f"{tokenizer_path}. Build it with `python src/tokenizer.py ...`."
        )

    # Apply formatting flags (useful for plain SMILES experiments)
    tokenizer.use_task_tokens = tokenizer_config.use_task_tokens
    tokenizer.use_role_tokens = tokenizer_config.use_role_tokens

    _save_tokenizer_assets(tokenizer, tokenizer_path)

    return tokenizer


def build_fsdp_strategy(distributed_config: DistributedConfig):
    """Build FSDP strategy if configured."""
    if distributed_config.strategy != "fsdp":
        return distributed_config.strategy

    from lightning.pytorch.strategies import FSDPStrategy
    from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial

    from src.transformer import EncoderBlock, DecoderBlock
    from src.decoder_model import CausalBlock

    # Map string to enum
    sharding_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }

    prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }

    fsdp_cfg = distributed_config.fsdp

    # Auto-wrap policy for transformer blocks
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={EncoderBlock, DecoderBlock, CausalBlock},
    )

    strategy = FSDPStrategy(
        sharding_strategy=sharding_map.get(fsdp_cfg.sharding, ShardingStrategy.FULL_SHARD),
        cpu_offload=fsdp_cfg.cpu_offload,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=prefetch_map.get(fsdp_cfg.backward_prefetch, BackwardPrefetch.BACKWARD_PRE),
        activation_checkpointing_policy={EncoderBlock, DecoderBlock, CausalBlock} if fsdp_cfg.activation_checkpointing else None,
    )

    return strategy


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.pop("hydra", None)
    schema = OmegaConf.structured(AppConfig)
    cfg = OmegaConf.merge(schema, cfg)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    app_config: AppConfig = OmegaConf.to_object(cfg)

    init_checkpoint = None
    if app_config.runtime.init_from:
        init_checkpoint = _load_init_checkpoint(app_config.runtime.init_from)

    if app_config.runtime.load_model_config:
        if not app_config.runtime.init_from:
            raise ValueError("runtime.load_model_config requires runtime.init_from.")
        init_model_cfg = init_checkpoint.get("model_config") if init_checkpoint else None
        if init_model_cfg is None:
            raise ValueError("init_from checkpoint is missing model_config.")
        app_config.model = _apply_model_config_from_init(app_config.model, init_model_cfg)

    overrides = []
    try:
        overrides = HydraConfig.get().overrides.task
    except Exception:
        pass
    if overrides:
        rank_zero_print(f"Overrides: {overrides}")

    model_config = app_config.model
    train_config = app_config.training
    data_config = app_config.data
    tokenizer_config = app_config.tokenizer
    distributed_config = app_config.distributed
    logging_config = app_config.logging
    hardware_config = app_config.hardware

    # Get model type from config
    model_type = model_config.type
    if model_type in {"encoder_decoder", "seq2seq"}:
        model_type = "transformer"
    model_config.type = model_type

    # Set random seed
    L.seed_everything(hardware_config.seed, workers=True)

    # Setup output directory
    output_dir = resolve_output_dir(logging_config)
    logging_config.run_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    if app_config.runtime.init_from and app_config.runtime.copy_init_to_output:
        init_path = Path(app_config.runtime.init_from)
        init_copy = output_dir / f"init_{init_path.name}"
        if not init_copy.exists():
            shutil.copy2(init_path, init_copy)
            rank_zero_print(f"Copied init checkpoint to {init_copy}")
        else:
            rank_zero_print(f"Init checkpoint copy already exists: {init_copy}")

    # Get task from config
    task = tokenizer_config.task

    rank_zero_print("\n" + "=" * 60)
    rank_zero_print(f"  Model Type: {model_type}")
    rank_zero_print(f"  Task: {task}")
    rank_zero_print(f"  Output Directory: {output_dir}")
    rank_zero_print("=" * 60)

    # Build tokenizer
    tokenizer = build_tokenizer(tokenizer_config, output_dir)
    rank_zero_print(f"Tokenizer class: {tokenizer.__class__.__name__}")

    # Update model config with tokenizer info
    model_config.vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else tokenizer.vocab_size
    model_config.pad_id = getattr(tokenizer, 'pad_id', None) or getattr(tokenizer, 'pad_token_id', 0)
    model_config.bos_id = getattr(tokenizer, 'bos_id', None) or getattr(tokenizer, 'bos_token_id', 1)
    model_config.eos_id = getattr(tokenizer, 'eos_id', None) or getattr(tokenizer, 'eos_token_id', 2)
    if model_type == "decoder_only":
        min_len = data_config.max_src_len + data_config.max_tgt_len
    else:
        min_len = max(data_config.max_src_len, data_config.max_tgt_len)
    if model_config.max_seq_len < min_len:
        raise ValueError(f"model.max_seq_len ({model_config.max_seq_len}) < required ({min_len})")

    # Compute effective num_workers per rank (avoid excessive workers in DDP)
    effective_num_workers = hardware_config.num_workers
    world_size = max(1, distributed_config.devices * distributed_config.num_nodes)
    try:
        cpu_available = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_available = os.cpu_count() or 1
    cpu_hint_env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS")
    if cpu_hint_env:
        try:
            cpu_hint = int(cpu_hint_env)
        except ValueError:
            cpu_hint = cpu_available
    else:
        cpu_hint = cpu_available
        if world_size > 1:
            cpu_hint = max(1, cpu_hint // world_size)

    if cpu_hint < effective_num_workers:
        rank_zero_print(
            f"Clamping num_workers {effective_num_workers} -> {cpu_hint} "
            f"(cpu_available={cpu_available}, world_size={world_size})"
        )
        effective_num_workers = cpu_hint

    hardware_config.num_workers = effective_num_workers

    # Save resolved config (post-tokenizer updates and worker clamping)
    save_resolved_config(output_dir, app_config, config_path=None, overrides=overrides)

    # Build data module
    data_module = ReactionDataModule(
        tokenizer=tokenizer,
        train_path=data_config.train_path,
        val_path=data_config.val_path if data_config.val_path else None,
        max_src_len=data_config.max_src_len,
        max_tgt_len=data_config.max_tgt_len,
        batch_size=train_config.batch_size,
        num_workers=effective_num_workers,
        task=task,
        model_type=model_type,
        # HuggingFace datasets caching
        use_hf_cache=data_config.use_hf_cache,
        cache_dir=data_config.cache_dir,
        tokenize_num_proc=data_config.tokenize_num_proc,
        use_task_tokens=tokenizer_config.use_task_tokens,
        use_role_tokens=tokenizer_config.use_role_tokens,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader() if data_config.val_path else None
    has_val = val_loader is not None
    val_check_interval = train_config.val_check_interval or logging_config.val_check_interval
    if not has_val:
        val_check_interval = None
    else:
        num_train_batches = len(train_loader)
        per_rank_batches = num_train_batches
        if isinstance(val_check_interval, float):
            if val_check_interval > 1:
                val_check_interval = int(val_check_interval)
            else:
                val_check_interval = max(1, int(per_rank_batches * val_check_interval))
        else:
            val_check_interval = int(val_check_interval)
        if per_rank_batches and val_check_interval > per_rank_batches:
            rank_zero_print(
                f"val_check_interval {val_check_interval} > train batches "
                f"{per_rank_batches}; clamping to {per_rank_batches}"
            )
            val_check_interval = per_rank_batches
        if val_check_interval < 1:
            val_check_interval = 1

    # Build model
    resolved_config = app_config_to_dict(app_config)

    module = ReactionModule(
        model_config=model_config,
        train_config=train_config,
        tokenizer=tokenizer,
        model_type=model_type,
        config=resolved_config,
    )

    if app_config.runtime.resume_from and app_config.runtime.init_from:
        raise ValueError("Set only one of runtime.resume_from or runtime.init_from, not both.")

    if app_config.runtime.init_from:
        _load_init_weights(
            init_from=app_config.runtime.init_from,
            module=module,
            model_config=model_config,
            strict=app_config.runtime.init_strict,
            check_model_config=app_config.runtime.check_model_config,
            checkpoint=init_checkpoint,
        )

    if app_config.runtime.freeze_n_layers or app_config.runtime.freeze_embeddings:
        _apply_freeze(
            module,
            freeze_n_layers=app_config.runtime.freeze_n_layers,
            freeze_embeddings=app_config.runtime.freeze_embeddings,
        )

    if train_config.l2sp_lambda and train_config.l2sp_lambda > 0:
        if not app_config.runtime.init_from:
            raise ValueError("training.l2sp_lambda requires runtime.init_from to avoid regularizing to random weights.")
        module.set_l2sp_base()

    rank_zero_print(f"\nModel parameters: {module.model.count_parameters():,}")
    rank_zero_print(f"Trainable parameters: {_count_trainable_params(module):,}")

    # Determine monitor metric
    monitor_metric = "val/loss" if has_val else "train/loss_epoch"

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="epoch={epoch}-step={step}-val_loss={val/loss:.4f}" if has_val else "epoch={epoch}-step={step}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=logging_config.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
            verbose=False,  # Suppress duplicate logs from all ranks
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
        SaveBestModelCallback(
            output_dir,
            monitor=monitor_metric,
            every_n_steps=val_check_interval if isinstance(val_check_interval, int) else None,
        ),
    ]

    if has_val and train_config.early_stopping_enabled:
        callbacks.append(
            EarlyStopping(
                monitor=train_config.early_stopping_monitor,
                patience=train_config.early_stopping_patience,
                mode=train_config.early_stopping_mode,
                verbose=False,  # Suppress duplicate logs from all ranks
            )
        )

    # Logger
    if logging_config.wandb:
        logger = WandbLogger(
            project=logging_config.wandb_project,
            name=logging_config.experiment_name,
            save_dir=output_dir,
        )
    else:
        logger = CSVLogger(
            save_dir=output_dir,
            name="logs",
        )

    # Build strategy
    strategy = build_fsdp_strategy(distributed_config)

    rank_zero_print(f"\n=== Distributed Config ===")
    rank_zero_print(f"  Strategy: {strategy}")
    rank_zero_print(f"  Devices: {distributed_config.devices}")
    rank_zero_print(f"  Num nodes: {distributed_config.num_nodes}")
    rank_zero_print(f"===========================\n")

    # Trainer
    trainer = L.Trainer(
        max_steps=train_config.max_steps,
        precision=train_config.precision,
        gradient_clip_val=train_config.grad_clip,
        accumulate_grad_batches=train_config.gradient_accumulation_steps,
        callbacks=callbacks,
        logger=logger,
        accelerator=hardware_config.accelerator,
        devices=distributed_config.devices,
        num_nodes=distributed_config.num_nodes,
        strategy=strategy,
        val_check_interval=val_check_interval,
        log_every_n_steps=logging_config.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=app_config.runtime.resume_from,
    )

    # Print final summary
    rank_zero_print("\n" + "=" * 60)
    rank_zero_print("  Training Complete!")
    rank_zero_print("=" * 60)
    rank_zero_print(f"\n  Output Directory: {output_dir}")
    rank_zero_print(f"\n  Files saved:")
    rank_zero_print(f"    - tokenizer.json                 (tokenizer)")
    rank_zero_print(f"    - config.resolved.yaml           (resolved configuration)")
    rank_zero_print(f"    - config.resolved.json           (resolved configuration)")
    rank_zero_print(f"    - config.meta.json               (config source/overrides)")
    rank_zero_print(f"    - model.pt                        (final model)")
    rank_zero_print(f"    - best_model.pt                   (best validation model)")
    rank_zero_print(f"    - checkpoints/                    (Lightning checkpoints)")
    rank_zero_print(f"    - logs/                           (training metrics CSV)")
    rank_zero_print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
