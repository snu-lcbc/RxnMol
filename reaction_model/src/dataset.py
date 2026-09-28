"""Unified dataset class for reaction prediction with HuggingFace caching."""

import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path


def _get_global_rank() -> int:
    """Get global rank in distributed setting."""
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return int(os.environ[key])
    return 0

try:
    from datasets import Dataset as HFDataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from .tokenizer import SMILESTokenizer


def _load_reaction_pairs(path: Path) -> List[Tuple[str, str]]:
    """Load reaction pairs in SMILES>>SMILES format (optionally with |ID suffix)."""
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ">>" not in line:
                continue
            rxn = line.rsplit("|", 1)[0] if "|" in line else line
            parts = rxn.split(">>", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
    return pairs


def _iter_reaction_pairs(path: Path):
    """Yield dicts for HF datasets from SMILES>>SMILES lines (optional |ID suffix)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ">>" not in line:
                continue
            rxn = line.rsplit("|", 1)[0] if "|" in line else line
            parts = rxn.split(">>", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                yield {"reactants": parts[0], "products": parts[1]}


class HFReactionDataset(Dataset):
    """
    Unified HuggingFace datasets-based reaction dataset with caching.

    Loads reaction pairs from SMILES>>SMILES text files.
    Tokenizes once and caches to disk. Subsequent runs load from cache.
    Uses Arrow format for memory-efficient access and proper DDP support.
    """

    def __init__(
        self,
        path: str,
        tokenizer: SMILESTokenizer,
        max_src_len: int = 256,
        max_tgt_len: int = 256,
        task: str = "forward",
        model_type: str = "transformer",
        cache_dir: str = ".cache/datasets",
        num_proc: int = 4,
        use_task_tokens: bool = True,
        use_role_tokens: bool = True,
    ):
        """
        Initialize HF-cached reaction dataset.

        Args:
            path: Path to the reaction dataset file
            tokenizer: SMILES tokenizer instance
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
            task: Task type - "forward", "retro", "elem", "mech"
            model_type: "transformer" (encoder-decoder) or "decoder_only"
            cache_dir: Directory to store cached tokenized data
            num_proc: Number of processes for parallel tokenization
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Install with: pip install datasets"
            )

        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.task = task
        self.model_type = model_type
        self.cache_dir = Path(cache_dir)
        self.num_proc = num_proc
        self.use_task_tokens = use_task_tokens
        self.use_role_tokens = use_role_tokens

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load and process dataset
        self.dataset = self._load_and_process()

    def _load_and_process(self) -> HFDataset:
        """Load raw data and tokenize with caching."""
        dataset = self._load_text()

        if _get_global_rank() == 0:
            print(f"Loaded {len(dataset)} reactions from {self.path}")

        # Create fingerprint from config + tokenizer + dataset file to ensure cache invalidation
        tokenizer_fingerprint = self.tokenizer.fingerprint()
        stat = self.path.stat()
        data_sig = f"{self.path.resolve()}_{stat.st_size}_{int(stat.st_mtime)}"
        import hashlib
        data_sig_hash = hashlib.sha1(data_sig.encode("utf-8")).hexdigest()
        fingerprint_payload = (
            f"{data_sig_hash}_{self.max_src_len}_{self.max_tgt_len}_"
            f"{self.task}_{self.model_type}_{tokenizer_fingerprint}"
        )
        config_fingerprint = hashlib.sha1(fingerprint_payload.encode("utf-8")).hexdigest()[:32]

        # Tokenize with caching (avoid duplicate work in DDP)
        def _ddp_barrier():
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except Exception:
                pass

        rank = _get_global_rank()
        if rank == 0:
            dataset = dataset.map(
                self._tokenize_fn,
                batched=True,
                batch_size=1000,
                num_proc=self.num_proc,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing {self.path.name}",
                load_from_cache_file=True,
                new_fingerprint=config_fingerprint,
            )
            _ddp_barrier()
        else:
            _ddp_barrier()
            dataset = dataset.map(
                self._tokenize_fn,
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=dataset.column_names,
                desc=f"Loading cached {self.path.name}",
                load_from_cache_file=True,
                new_fingerprint=config_fingerprint,
            )

        return dataset

    def _load_text(self) -> HFDataset:
        """Load SMILES>>SMILES text (optionally with |ID suffix)."""
        return HFDataset.from_generator(
            lambda: _iter_reaction_pairs(self.path),
            cache_dir=str(self.cache_dir),
        )

    def _tokenize_fn(self, examples: Dict) -> Dict[str, List]:
        """Batch tokenization function for HF map()."""
        if self.model_type == "decoder_only":
            results = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
        else:
            results = {
                "src": [],
                "tgt_in": [],
                "tgt_out": [],
            }

        max_length = (
            self.max_src_len if self.model_type == "transformer"
            else self.max_src_len + self.max_tgt_len
        )

        for reactants, products in zip(examples["reactants"], examples["products"]):
            encoded = self.tokenizer.encode_reaction(
                reactants=reactants,
                products=products,
                task=self.task,
                model_type=self.model_type,
                max_length=max_length,
                max_src_length=self.max_src_len,
                max_tgt_length=self.max_tgt_len,
                truncation=True,
                return_tensors=None,
                use_task_tokens=self.use_task_tokens,
                use_role_tokens=self.use_role_tokens,
            )

            for key in results:
                if key in encoded:
                    results[key].append(encoded[key])

        return results

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """Get a single sample."""
        item = self.dataset[idx]
        # Convert to regular Python lists for collate_fn compatibility
        return {k: list(v) if hasattr(v, 'tolist') else v for k, v in item.items()}


# FlowERDataset and MechUSPTODataset have been consolidated into HFReactionDataset above.
# The unified HFReactionDataset expects SMILES>>SMILES text files.


def _trim_tgt(encoded: Dict[str, List[int]], max_tgt_len: int) -> Dict[str, List[int]]:
    if max_tgt_len is None or "tgt_in" not in encoded:
        return encoded
    encoded["tgt_in"] = encoded["tgt_in"][:max_tgt_len]
    encoded["tgt_out"] = encoded["tgt_out"][:max_tgt_len]
    if "tgt_mask" in encoded:
        encoded["tgt_mask"] = encoded["tgt_mask"][:max_tgt_len]
    return encoded


def collate_fn(
    batch: List[Dict[str, List[int]]],
    pad_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function with dynamic padding.

    Pads all sequences in the batch to the maximum length in that batch.

    Args:
        batch: List of samples from dataset
        pad_id: Padding token ID

    Returns:
        Dictionary with padded tensors and attention masks
    """
    # Find max lengths in this batch
    max_src_len = max(len(s["src"]) for s in batch)
    max_tgt_len = max(len(s["tgt_in"]) for s in batch)

    batch_size = len(batch)

    # Initialize tensors
    src = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    src_mask = torch.zeros((batch_size, max_src_len), dtype=torch.bool)
    tgt_in = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    tgt_out = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    tgt_mask = torch.zeros((batch_size, max_tgt_len), dtype=torch.bool)

    # Fill tensors
    for i, sample in enumerate(batch):
        src_len = len(sample["src"])
        tgt_len = len(sample["tgt_in"])

        src[i, :src_len] = torch.tensor(sample["src"], dtype=torch.long)
        src_mask[i, :src_len] = True

        tgt_in[i, :tgt_len] = torch.tensor(sample["tgt_in"], dtype=torch.long)
        tgt_out[i, :tgt_len] = torch.tensor(sample["tgt_out"], dtype=torch.long)
        tgt_mask[i, :tgt_len] = True

    return {
        "src": src,
        "src_mask": src_mask,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "tgt_mask": tgt_mask,
    }


def create_collate_fn(pad_id: int = 0):
    """Factory function to create collate_fn with custom pad_id."""
    def _collate(batch):
        return collate_fn(batch, pad_id=pad_id)
    return _collate


def collate_causal(
    batch: List[Dict[str, List[int]]],
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for decoder-only (causal) models with label masking.

    For instruction format, the labels mask out the prompt portion
    (instruction + reactants + separator) so loss is only computed on products.

    Args:
        batch: List of samples with 'input_ids', 'attention_mask', and optionally 'labels'
        pad_id: Padding token ID
        ignore_index: Index to use for masked positions in labels (default: -100)

    Returns:
        Dictionary with padded tensors:
        - input_ids: (batch, seq_len)
        - attention_mask: (batch, seq_len)
        - labels: (batch, seq_len) with ignore_index for prompt positions
    """
    # Determine max length in batch
    max_len = max(len(s["input_ids"]) for s in batch)
    batch_size = len(batch)

    # Initialize tensors
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)

    for i, sample in enumerate(batch):
        seq_len = len(sample["input_ids"])
        input_ids[i, :seq_len] = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask[i, :seq_len] = torch.tensor(sample["attention_mask"], dtype=torch.bool)

        # Use provided labels if available, otherwise use input_ids shifted
        if "labels" in sample:
            sample_labels = sample["labels"]
            labels[i, :len(sample_labels)] = torch.tensor(sample_labels, dtype=torch.long)
        else:
            # For causal LM without explicit labels, predict next token
            labels[i, :seq_len] = torch.tensor(sample["input_ids"], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def create_collate_causal(pad_id: int = 0, ignore_index: int = -100):
    """Factory function to create collate_causal with custom pad_id."""
    def _collate(batch):
        return collate_causal(batch, pad_id=pad_id, ignore_index=ignore_index)
    return _collate


class ReactionDataset(Dataset):
    """Non-cached dataset for debugging or small datasets."""

    def __init__(
        self,
        path: str,
        tokenizer: SMILESTokenizer,
        max_src_len: int = 256,
        max_tgt_len: int = 256,
        task: str = "forward",
        model_type: str = "transformer",
        use_task_tokens: bool = True,
        use_role_tokens: bool = True,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.task = task
        self.model_type = model_type
        self.use_task_tokens = use_task_tokens
        self.use_role_tokens = use_role_tokens

        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self.pairs = _load_reaction_pairs(self.path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        reactants, products = self.pairs[idx]
        max_length = (
            self.max_src_len if self.model_type == "transformer"
            else self.max_src_len + self.max_tgt_len
        )
        encoded = self.tokenizer.encode_reaction(
            reactants=reactants,
            products=products,
            task=self.task,
            model_type=self.model_type,
            max_length=max_length,
            max_src_length=self.max_src_len,
            max_tgt_length=self.max_tgt_len,
            truncation=True,
            return_tensors=None,
            use_task_tokens=self.use_task_tokens,
            use_role_tokens=self.use_role_tokens,
        )
        return {k: list(v) if hasattr(v, "tolist") else v for k, v in encoded.items()}


class ReactionDataModule:
    """
    Simple data module for managing train/val/test dataloaders.

    Not using LightningDataModule to keep it simple and flexible.
    Supports both encoder-decoder (transformer) and decoder-only models.
    Supports HuggingFace datasets caching for efficient tokenization.
    """

    def __init__(
        self,
        tokenizer: SMILESTokenizer,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        max_src_len: int = 256,
        max_tgt_len: int = 256,
        batch_size: int = 64,
        num_workers: int = 4,
        task: str = "forward",
        model_type: str = "transformer",
        use_hf_cache: bool = True,
        cache_dir: str = ".cache/datasets",
        tokenize_num_proc: int = 4,
        use_task_tokens: bool = True,
        use_role_tokens: bool = True,
    ):
        """
        Initialize data module.

        Args:
            tokenizer: SMILES tokenizer instance
            train_path: Path to training data
            val_path: Path to validation data
            test_path: Path to test data
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
            batch_size: Batch size
            num_workers: Number of data loading workers
            task: Task type - "forward", "retro", "elem", "mech"
            model_type: "transformer" (encoder-decoder) or "decoder_only"
            use_hf_cache: Use HuggingFace datasets for caching tokenized data
            cache_dir: Directory to store cached tokenized datasets
            tokenize_num_proc: Number of processes for parallel tokenization
        """
        self.tokenizer = tokenizer
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.model_type = model_type
        self.use_hf_cache = use_hf_cache
        self.cache_dir = cache_dir
        self.tokenize_num_proc = tokenize_num_proc
        self.use_task_tokens = use_task_tokens
        self.use_role_tokens = use_role_tokens

        # Get pad_id from tokenizer
        pad_id = getattr(tokenizer, "pad_id", None) or getattr(tokenizer, "pad_token_id", 0)

        # Select appropriate collate function based on model type
        if model_type == "decoder_only":
            self._collate_fn = create_collate_causal(pad_id=pad_id)
        else:
            self._collate_fn = create_collate_fn(pad_id=pad_id)

    def _create_dataset(self, path: str) -> Dataset:
        """Create dataset (HF-cached or simple) for SMILES>>SMILES text."""
        if not HF_DATASETS_AVAILABLE and self.use_hf_cache:
            raise ImportError(
                "HuggingFace datasets required for DDP-safe data loading. "
                "Install with: pip install datasets"
            )

        if self.use_hf_cache:
            return HFReactionDataset(
                path=path,
                tokenizer=self.tokenizer,
                max_src_len=self.max_src_len,
                max_tgt_len=self.max_tgt_len,
                task=self.task,
                model_type=self.model_type,
                cache_dir=self.cache_dir,
                num_proc=self.tokenize_num_proc,
                use_task_tokens=self.use_task_tokens,
                use_role_tokens=self.use_role_tokens,
            )

        return ReactionDataset(
            path=path,
            tokenizer=self.tokenizer,
            max_src_len=self.max_src_len,
            max_tgt_len=self.max_tgt_len,
            task=self.task,
            model_type=self.model_type,
            use_task_tokens=self.use_task_tokens,
            use_role_tokens=self.use_role_tokens,
        )

    def train_dataloader(self):
        """Get training dataloader with DDP-safe settings."""
        from torch.utils.data import DataLoader

        if not self.train_path:
            raise ValueError("train_path not set")

        dataset = self._create_dataset(self.train_path)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Lightning wraps with DistributedSampler in DDP
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True,  # CRITICAL: Ensures equal batch sizes across GPUs in DDP
            persistent_workers=self.num_workers > 0,  # Avoid worker respawn overhead
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        from torch.utils.data import DataLoader

        if not self.val_path:
            raise ValueError("val_path not set")

        dataset = self._create_dataset(self.val_path)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=False,  # Don't drop for validation (want all samples)
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        from torch.utils.data import DataLoader

        if not self.test_path:
            raise ValueError("test_path not set")

        dataset = self._create_dataset(self.test_path)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=False,  # Don't drop for test (want all samples)
            persistent_workers=self.num_workers > 0,
        )
