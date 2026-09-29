"""
SMILES Tokenizer with HuggingFace compatibility.

Uses word-level tokenization where each SMILES token (atom, bond, etc.) is a word.
NOT BPE - the vocabulary is built from unique SMILES tokens.
"""

import re
import json
import os
import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, Iterator
from collections import Counter
from tqdm.auto import tqdm

try:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    PreTrainedTokenizer = object
    PreTrainedTokenizerFast = None


# =============================================================================
# SMILES Regex Pattern
# =============================================================================

# Regex pattern for SMILES tokenization
SMILES_PATTERN = re.compile(
    r"(\[[^\]]+\]"           # Bracketed atoms: [C:1], [N+:2], [OH-], [nH], etc.
    r"|Br|Cl"                # Two-letter organic atoms
    r"|[BCNOPSFIbcnops]"     # Single-letter atoms (organic + aromatic)
    r"|@@|@"                 # Stereochemistry
    r"|%[0-9]{2}"            # Ring closure >= 10
    r"|[0-9]"                # Ring closure 0-9
    r"|>>|>|\."              # Reaction arrow, agent separator, molecule separator
    r"|[=#:\-\+\\/\(\)]"     # Bonds and grouping
    r"|.)"                   # Fallback for any other character
)

def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "on"):
            return True
        if lowered in ("false", "0", "no", "n", "off"):
            return False
    return bool(value)

## Tokenization note (for datasets formatted as "reactants>>products")
#
# In the training pipeline, the dataset loader splits each line on ">>" and
# passes reactants/products separately to `encode_reaction()`. So the tokenizer
# does not see ">>" in normal training; it only tokenizes the left and right
# SMILES strings. If you call `tokenize_smiles()` directly on a raw "A>>B"
# string, ">>" will be captured as a token by the regex.
#
# How SMILES>>SMILES is processed (encoder‑decoder, transformer)
# - Input format: reactants >> products
# - Encoder input: <task> <reactants> <reactant_smiles_tokens>
# - Decoder target: <bos> <product_smiles_tokens> <eos>
#
# Example: reactants="CCO", products="CCO"
# - Encoder (source):
#   * if use_task_tokens=true: [<forward-reaction>]
#   * if use_role_tokens=true: [<reactants>]
#   * then tokenized reactants: C C O
#   * final: <forward-reaction> <reactants> C C O
#   * disabling task tokens removes <forward-reaction>
#   * disabling role tokens removes <reactants>
#   * reactant tokens always remain
# - Decoder (target):
#   * always uses BOS/EOS (non‑configurable)
#   * tgt_in  = <bos> + products
#   * tgt_out = products + <eos>
#   * so the decoder learns:
#     Input:  <bos> C C O
#     Target: C C O </s>
# - Summary: encoder sees only reactants, decoder learns to emit products.


def tokenize_smiles(smiles: str) -> List[str]:
    """
    Tokenize a SMILES string into tokens.

    Args:
        smiles: SMILES string (can include reaction arrows >>)

    Returns:
        List of SMILES tokens
    """
    return SMILES_PATTERN.findall(smiles)


def _normalize_smiles_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    if "|" in line:
        line = line.rsplit("|", 1)[0]
    return line


def _chunked_iterator(iterator: Iterator[str], chunk_size: int) -> Iterator[List[str]]:
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _count_tokens_chunk(lines: List[str]) -> Tuple[Counter, int]:
    counter = Counter()
    for line in lines:
        counter.update(tokenize_smiles(line))
    return counter, len(lines)


def _default_num_proc() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = os.environ.get(var)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def _is_slow_vocab_json(path: Path) -> bool:
    data = json.loads(path.read_text())
    return isinstance(data, dict) and ("token2id" in data or "vocab" in data)


def _build_fast_backend(
    files: List[str],
    min_frequency: int,
    special_tokens: List[str],
    show_progress: bool = True,
):
    from tokenizers import Tokenizer, Regex
    from tokenizers.models import WordLevel
    from tokenizers.trainers import WordLevelTrainer
    from tokenizers.pre_tokenizers import Split

    backend = Tokenizer(WordLevel(unk_token="<unk>"))
    backend.pre_tokenizer = Split(Regex(SMILES_PATTERN.pattern), behavior="isolated")
    trainer = WordLevelTrainer(
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=show_progress,
    )

    def iter_lines():
        for file_path in files:
            with open(file_path, "r") as handle:
                for line in handle:
                    normalized = _normalize_smiles_line(line)
                    if normalized:
                        yield normalized

    total_lines = None
    if show_progress:
        total_lines = 0
        for file_path in files:
            with open(file_path, "r") as handle:
                total_lines += sum(1 for _ in handle)

    backend.train_from_iterator(iter_lines(), trainer=trainer, length=total_lines)
    return backend


# =============================================================================
# Unified SMILES Tokenizer
# =============================================================================


class SMILESTokenizer(PreTrainedTokenizer if HF_AVAILABLE else object):
    """
    HuggingFace-compatible SMILES tokenizer with word-level tokenization.

    Each SMILES token (atom, bond, ring number, etc.) is treated as a "word".
    The vocabulary is built from unique tokens, NOT using BPE or subword methods.

    Features:
    - Regex-based SMILES tokenization
    - HuggingFace PreTrainedTokenizer interface
    - Instruction format for decoder-only models
    - Encoder-decoder format support
    - Vocabulary building from data
    - Optional plain-SMILES formatting (no task/role tokens, no BOS/EOS)

    Example:
        # Build from data
        tokenizer = SMILESTokenizer()
        tokenizer.train(files=["train.txt"])
        tokenizer.save_pretrained("tokenizer/")

        # Load and use
        tokenizer = SMILESTokenizer.from_pretrained("tokenizer/")
        encoded = tokenizer.encode_reaction("CCO.CC=O", "CCOCC", task="forward", model_type="transformer")
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    # Task tokens for chemistry-native format
    TASK_TOKENS = {
        "forward": "<forward-reaction>",      # Forward reaction prediction
        "retro": "<retrosynthesis>",      # Retrosynthesis
        "elem": "<elementary-step>",        # Elementary step
        "mech": "<mechanism>",        # Mechanism prediction
    }

    # Role tokens
    ROLE_TOKENS = {
        "react": "<reactants>",          # Reaction marker
        "reagent": "<reagents>",  # Reagents/catalysts
        "cond": "<conditions>",        # Conditions (future)
        "prod": "<products>",          # Products (future)
    }

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        sep_token: str = "<sep>",
        model_max_length: int = 512,
        use_task_tokens: bool = True,
        use_role_tokens: bool = True,
        **kwargs,
    ):
        """
        Initialize SMILES tokenizer.

        Args:
            vocab_file: Path to vocab.json file
            vocab: Vocabulary dict (token -> id)
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            unk_token: Unknown token
            pad_token: Padding token
            sep_token: Separator token (between input/output)
            model_max_length: Maximum sequence length
        """
        # Store separator and formatting flags
        self._sep = sep_token
        self.use_task_tokens = _coerce_bool(use_task_tokens)
        self.use_role_tokens = _coerce_bool(use_role_tokens)

        # Define special tokens in order (indices 0-13)
        self._special_tokens = [
            pad_token,              # 0
            bos_token,              # 1
            eos_token,              # 2
            unk_token,              # 3
            sep_token,              # 4
            # Task tokens
            "<forward-reaction>",   # 5 - forward prediction
            "<retrosynthesis>",     # 6 - retrosynthesis
            "<elementary-step>",    # 7 - elementary step
            "<mechanism>",          # 8 - mechanism
            # Role tokens
            "<reactants>",          # 9 - reactants marker
            "<reagents>",           # 10 - reagents
            "<conditions>",         # 11 - conditions
            "<products>",           # 12 - products
        ]

        # Initialize vocabulary
        if vocab is not None:
            self.token2id = dict(vocab)
            self.id2token = {v: k for k, v in vocab.items()}
        elif vocab_file is not None and Path(vocab_file).exists():
            self._load_vocab(vocab_file)
        else:
            # Start with special tokens only
            self.token2id = {tok: i for i, tok in enumerate(self._special_tokens)}
            self.id2token = {i: tok for i, tok in enumerate(self._special_tokens)}

        # Ensure all special tokens are in vocabulary
        self._add_special_tokens_to_vocab()

        # Initialize HF parent class if available
        if HF_AVAILABLE:
            # Collect task and role tokens for HF registration
            additional_tokens = list(self.TASK_TOKENS.values()) + list(self.ROLE_TOKENS.values())
            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                pad_token=pad_token,
                sep_token=sep_token,
                model_max_length=model_max_length,
                additional_special_tokens=additional_tokens,
                **kwargs,
            )

    def _add_special_tokens_to_vocab(self):
        """Ensure all special tokens are in vocabulary."""
        for tok in self._special_tokens:
            if tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

    def _load_vocab(self, path: str):
        """Load vocabulary from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Handle both formats: direct vocab or nested structure
        if "token2id" in data:
            self.token2id = data["token2id"]
        elif "vocab" in data:
            self.token2id = data["vocab"]
        else:
            self.token2id = data

        self.id2token = {int(v): k for k, v in self.token2id.items()}

        # Restore formatting flags if present
        format_cfg = data.get("format_config") if isinstance(data, dict) else None
        if isinstance(format_cfg, dict):
            self.use_task_tokens = _coerce_bool(format_cfg.get("use_task_tokens", self.use_task_tokens))
            self.use_role_tokens = _coerce_bool(format_cfg.get("use_role_tokens", self.use_role_tokens))

    # =========================================================================
    # Vocabulary Training (HuggingFace-style)
    # =========================================================================

    def train_from_iterator(
        self,
        iterator: Iterator[str],
        min_frequency: int = 1,
        show_progress: bool = True,
        length: Optional[int] = None,
    ):
        """
        Train tokenizer from an iterator of texts (HuggingFace-style).

        Args:
            iterator: Iterator yielding SMILES strings
            min_frequency: Minimum token frequency to include in vocabulary
            show_progress: Show progress bar
            length: Optional length hint for progress bar
        """
        counter = Counter()

        if show_progress:
            iterator = tqdm(iterator, desc="Training tokenizer", total=length)

        for text in iterator:
            tokens = tokenize_smiles(text)
            counter.update(tokens)

        # Add tokens meeting frequency threshold
        for token, freq in counter.items():
            if freq >= min_frequency and token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

        if show_progress:
            print(f"Vocabulary size: {len(self.token2id)}")

    def train(
        self,
        files: List[str],
        min_frequency: int = 1,
        show_progress: bool = True,
        num_proc: Optional[int] = None,
        chunk_size: int = 2000,
    ):
        """
        Train tokenizer from files (HuggingFace-style).

        Args:
            files: List of file paths containing SMILES (one per line, or reaction format)
            min_frequency: Minimum token frequency to include in vocabulary
            show_progress: Show progress bar
            num_proc: Number of worker processes for token counting
            chunk_size: Number of lines per work chunk
        """
        def file_iterator():
            for file_path in files:
                with open(file_path, "r") as f:
                    for line in f:
                        normalized = _normalize_smiles_line(line)
                        if normalized:
                            yield normalized

        if num_proc is None:
            num_proc = _default_num_proc()
        if chunk_size < 1:
            chunk_size = 1

        if num_proc <= 1:
            # Count total lines for progress bar
            total_lines = 0
            if show_progress:
                for file_path in files:
                    with open(file_path, "r") as f:
                        total_lines += sum(1 for _ in f)

            self.train_from_iterator(
                file_iterator(),
                min_frequency=min_frequency,
                show_progress=show_progress,
                length=total_lines if show_progress else None,
            )
            return

        counter = Counter()
        total_lines = None
        progress = None
        if show_progress:
            total_lines = 0
            for file_path in files:
                with open(file_path, "r") as f:
                    total_lines += sum(1 for _ in f)
            progress = tqdm(total=total_lines, desc="Training tokenizer")

        with mp.Pool(processes=num_proc) as pool:
            for chunk_counter, processed in pool.imap_unordered(
                _count_tokens_chunk,
                _chunked_iterator(file_iterator(), chunk_size),
                chunksize=1,
            ):
                counter.update(chunk_counter)
                if progress:
                    progress.update(processed)

        if progress:
            progress.close()

        for token, freq in counter.items():
            if freq >= min_frequency and token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

        if show_progress:
            print(f"Vocabulary size: {len(self.token2id)}")

    def add_tokens(
        self,
        new_tokens: Union[str, List[str]],
        special_tokens: bool = False,
    ) -> int:
        """
        Add tokens to vocabulary (HuggingFace-compatible signature).

        Args:
            new_tokens: Token or list of tokens to add
            special_tokens: Whether tokens are special tokens (ignored, for HF compat)

        Returns:
            Number of tokens added
        """
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]

        added = 0
        for token in new_tokens:
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token
                added += 1
        return added

    # =========================================================================
    # Core Tokenization (HF Interface)
    # =========================================================================

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        return len(self.token2id)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token2id)

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dict."""
        return dict(self.token2id)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using SMILES regex."""
        return tokenize_smiles(text)

    def tokenize(self, text: str) -> List[str]:
        """Public tokenize method."""
        return self._tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.token2id.get(token, self.unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self.id2token.get(index, "<unk>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string (SMILES are concatenated without spaces)."""
        return "".join(tokens)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token(s) to ID(s)."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert ID(s) to token(s)."""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(i) for i in ids]

    # =========================================================================
    # Special Token Properties
    # =========================================================================

    @property
    def pad_id(self) -> int:
        return self.token2id.get("<pad>", 0)

    @property
    def bos_id(self) -> int:
        return self.token2id.get("<s>", 1)

    @property
    def eos_id(self) -> int:
        return self.token2id.get("</s>", 2)

    @property
    def unk_id(self) -> int:
        return self.token2id.get("<unk>", 3)

    @property
    def sep_id(self) -> int:
        return self.token2id.get("<sep>", 4)

    # Task token IDs
    @property
    def forward_id(self) -> int:
        """Forward reaction prediction task token."""
        return self.token2id.get("<forward-reaction>", 5)

    @property
    def retro_id(self) -> int:
        """Retrosynthesis task token."""
        return self.token2id.get("<retrosynthesis>", 6)

    @property
    def elem_id(self) -> int:
        """Elementary step task token."""
        return self.token2id.get("<elementary-step>", 7)

    @property
    def mech_id(self) -> int:
        """Mechanism prediction task token."""
        return self.token2id.get("<mechanism>", 8)

    # Role token IDs
    @property
    def reactants_id(self) -> int:
        """Reactants marker token."""
        return self.token2id.get("<reactants>", 9)

    @property
    def reagents_id(self) -> int:
        """Reagents marker token."""
        return self.token2id.get("<reagents>", 10)

    @property
    def conditions_id(self) -> int:
        """Conditions marker token."""
        return self.token2id.get("<conditions>", 11)

    @property
    def products_id(self) -> int:
        """Products marker token."""
        return self.token2id.get("<products>", 12)

    def get_task_id(self, task: str) -> int:
        """Get token ID for a task name."""
        task_map = {
            "forward": self.forward_id,
            "fwd": self.forward_id,
            "forward-reaction": self.forward_id,
            "retro": self.retro_id,
            "retrosynthesis": self.retro_id,
            "elem": self.elem_id,
            "elementary": self.elem_id,
            "elementary-step": self.elem_id,
            "mech": self.mech_id,
            "mechanism": self.mech_id,
        }
        return task_map.get(task.lower(), self.forward_id)

    # HF compatibility
    @property
    def pad_token_id(self) -> int:
        return self.pad_id

    @property
    def bos_token_id(self) -> int:
        return self.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.eos_id

    @property
    def unk_token_id(self) -> int:
        return self.unk_id

    @property
    def sep_token_id(self) -> int:
        return self.sep_id

    # =========================================================================
    # Encoding Methods
    # =========================================================================

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Encode SMILES string to token IDs.

        Args:
            text: SMILES string
            add_bos: Add BOS token at start
            add_eos: Add EOS token at end
            max_len: Maximum length (truncate if exceeded)

        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        ids = [self._convert_token_to_id(t) for t in tokens]

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            # Ensure EOS is at end if we truncated
            if add_eos and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        """
        Decode token IDs to SMILES string.

        Args:
            ids: List of token IDs
            skip_special: Skip special tokens in output
            stop_at_eos: Stop decoding at EOS token

        Returns:
            Decoded SMILES string
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id,
                       self.sep_id, self.forward_id, self.retro_id, self.elem_id,
                       self.mech_id, self.reactants_id, self.reagents_id,
                       self.conditions_id, self.products_id}

        tokens = []
        for idx in ids:
            if stop_at_eos and idx == self.eos_id:
                break
            if skip_special and idx in special_ids:
                continue
            token = self._convert_id_to_token(idx)
            tokens.append(token)

        return "".join(tokens)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = True,
        max_len: Optional[int] = None,
        padding: bool = True,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Encode batch of SMILES strings with padding.

        Args:
            texts: List of SMILES strings
            add_bos: Add BOS token
            add_eos: Add EOS token
            max_len: Maximum length
            padding: Pad to max length in batch

        Returns:
            (input_ids, attention_masks)
        """
        all_ids = [self.encode(t, add_bos=add_bos, add_eos=add_eos, max_len=max_len) for t in texts]

        if padding:
            max_batch_len = max(len(ids) for ids in all_ids)
            attention_masks = []

            for i, ids in enumerate(all_ids):
                mask = [1] * len(ids)
                padding_len = max_batch_len - len(ids)
                if padding_len > 0:
                    all_ids[i] = ids + [self.pad_id] * padding_len
                    mask = mask + [0] * padding_len
                attention_masks.append(mask)

            return all_ids, attention_masks

        return all_ids, [[1] * len(ids) for ids in all_ids]

    # =========================================================================
    # Reaction Encoding
    # =========================================================================

    def encode_reaction(
        self,
        reactants: str,
        products: Optional[str] = None,
        task: str = "forward",
        model_type: str = "decoder_only",
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        use_task_tokens: Optional[bool] = None,
        use_role_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Encode a reaction with chemistry-native format.

        Format for decoder-only:
            <task> <rxn> reactants <sep> products <eos>
            Example: <fwd> <rxn> CC=O.N <sep> CC(N)O </s>

        Format for encoder-decoder (transformer):
            Encoder: <task> <rxn> reactants
            Decoder: <bos> products <eos>

        Args:
            reactants: Reactant SMILES string
            products: Product SMILES string (optional for inference)
            task: Task type - "forward", "retro", "elem", "mech"
            model_type: "decoder_only" or "transformer"
            add_special_tokens: Add special tokens
            max_length: Maximum sequence length (fallback if max_src_length/max_tgt_length not set)
            max_src_length: Maximum source length (encoder input)
            max_tgt_length: Maximum target length (decoder input/output)
            truncation: Truncate to max_length
            return_tensors: "pt" for PyTorch tensors, None for lists
            return_attention_mask: Include attention mask
            use_task_tokens: Override tokenizer.use_task_tokens
            use_role_tokens: Override tokenizer.use_role_tokens

        Returns:
            Dict with input_ids, attention_mask, and optionally labels/tgt_in/tgt_out
        """
        if model_type == "decoder_only":
            return self._encode_causal_format(
                reactants=reactants,
                products=products,
                task=task,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                max_src_length=max_src_length,
                max_tgt_length=max_tgt_length,
                truncation=truncation,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                use_task_tokens=use_task_tokens,
                use_role_tokens=use_role_tokens,
            )
        else:
            return self._encode_encoder_decoder_format(
                reactants=reactants,
                products=products,
                task=task,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                max_src_length=max_src_length,
                max_tgt_length=max_tgt_length,
                truncation=truncation,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                use_task_tokens=use_task_tokens,
                use_role_tokens=use_role_tokens,
            )

    def _encode_causal_format(
        self,
        reactants: str,
        products: Optional[str] = None,
        task: str = "forward",
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        use_task_tokens: Optional[bool] = None,
        use_role_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Encode in causal (decoder-only) format:
            <task> <rxn> reactants <sep> products <eos>
        """
        use_task_tokens = self.use_task_tokens if use_task_tokens is None else use_task_tokens
        use_role_tokens = self.use_role_tokens if use_role_tokens is None else use_role_tokens

        task_id = self.get_task_id(task)
        reactant_ids = [self._convert_token_to_id(t) for t in self._tokenize(reactants)]

        # Build prompt: <task> <reactants> reactants <sep>
        prompt_ids = []
        if use_task_tokens:
            prompt_ids.append(task_id)
        if use_role_tokens:
            prompt_ids.append(self.reactants_id)
        prompt_ids += reactant_ids
        if use_role_tokens:
            prompt_ids.append(self.sep_id)

        if products is not None:
            # Training: include products
            product_ids = [self._convert_token_to_id(t) for t in self._tokenize(products)]

            if add_special_tokens:
                input_ids = prompt_ids + product_ids + [self.eos_id]
            else:
                input_ids = prompt_ids + product_ids

            # Labels: mask prompt with -100, only predict products + eos
            if add_special_tokens:
                labels = [-100] * len(prompt_ids) + product_ids + [self.eos_id]
            else:
                labels = [-100] * len(prompt_ids) + product_ids
        else:
            # Inference: prompt only
            input_ids = prompt_ids
            labels = None

        # Truncation
        if max_length is None and max_src_length is not None and max_tgt_length is not None:
            max_length = max_src_length + max_tgt_length
        max_length = max_length or getattr(self, "model_max_length", 512)
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            if labels is not None:
                labels = labels[:max_length]
            # Ensure EOS at end if we truncated products
            if add_special_tokens and products is not None:
                prompt_len = len(prompt_ids)
                if prompt_len < max_length:
                    input_ids[-1] = self.eos_id
                    if labels is not None:
                        labels[-1] = self.eos_id

        result = {"input_ids": input_ids}

        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)

        if labels is not None:
            result["labels"] = labels

        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor([v]) for k, v in result.items()}

        return result

    def _encode_encoder_decoder_format(
        self,
        reactants: str,
        products: Optional[str] = None,
        task: str = "forward",
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        use_task_tokens: Optional[bool] = None,
        use_role_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Encode in encoder-decoder format.
        Source: <task> <rxn> reactants
        Target: <bos> products <eos>
        """
        use_task_tokens = self.use_task_tokens if use_task_tokens is None else use_task_tokens
        use_role_tokens = self.use_role_tokens if use_role_tokens is None else use_role_tokens

        task_id = self.get_task_id(task)
        reactant_ids = [self._convert_token_to_id(t) for t in self._tokenize(reactants)]

        # Encoder input: <task> <reactants> reactants
        input_ids = []
        if use_task_tokens:
            input_ids.append(task_id)
        if use_role_tokens:
            input_ids.append(self.reactants_id)
        input_ids += reactant_ids

        src_max_length = max_src_length or max_length or getattr(self, "model_max_length", 512)
        if truncation and len(input_ids) > src_max_length:
            input_ids = input_ids[:src_max_length]

        result = {
            "src": input_ids,
            "input_ids": input_ids,
        }

        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)
            result["src_mask"] = [1] * len(input_ids)

        if products is not None:
            product_ids = [self._convert_token_to_id(t) for t in self._tokenize(products)]

            if add_special_tokens:
                tgt_in = [self.bos_id] + product_ids
                tgt_out = product_ids + [self.eos_id]
            else:
                tgt_in = product_ids
                tgt_out = product_ids

            if truncation:
                tgt_max_length = max_tgt_length or max_length or getattr(self, "model_max_length", 512)
                tgt_in = tgt_in[:tgt_max_length]
                tgt_out = tgt_out[:tgt_max_length]
                if add_special_tokens and tgt_out:
                    tgt_out[-1] = self.eos_id

            result["tgt_in"] = tgt_in
            result["tgt_out"] = tgt_out
            result["tgt_mask"] = [1] * len(tgt_in)

        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor([v]) for k, v in result.items()}

        return result

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert keys to strings (AddedToken -> str)
        token2id_serializable = {str(k): v for k, v in self.token2id.items()}

        data = {
            "token2id": token2id_serializable,
            "special_tokens": self._special_tokens,
            "sep": self._sep,
            "task_tokens": self.TASK_TOKENS,
            "role_tokens": self.ROLE_TOKENS,
            "format_config": {
                "use_task_tokens": self.use_task_tokens,
                "use_role_tokens": self.use_role_tokens,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """Save vocabulary (HF interface)."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        self.save(vocab_file)
        return (vocab_file,)

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save tokenizer in HF format."""
        if HF_AVAILABLE:
            super().save_pretrained(save_directory, **kwargs)
        else:
            self.save_vocabulary(save_directory)

    def load(self, path: str):
        """Load vocabulary from file (backward compatibility)."""
        self._load_vocab(path)
        self._add_special_tokens_to_vocab()

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "SMILESTokenizer":
        """Load tokenizer from vocab file."""
        return cls(vocab_file=path, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        **kwargs,
    ) -> "SMILESTokenizer":
        """Load tokenizer from directory."""
        path = Path(pretrained_model_name_or_path)

        if path.is_dir():
            tokenizer_json = path / "tokenizer.json"
            vocab_json = path / "vocab.json"
            if tokenizer_json.exists() and not _is_slow_vocab_json(tokenizer_json):
                return SMILESTokenizerFast.from_file(str(tokenizer_json), **kwargs)
            if vocab_json.exists():
                return cls(vocab_file=str(vocab_json), **kwargs)
            if tokenizer_json.exists():
                return cls(vocab_file=str(tokenizer_json), **kwargs)
        elif path.is_file():
            if path.name == "tokenizer.json" and not _is_slow_vocab_json(path):
                return SMILESTokenizerFast.from_file(str(path), **kwargs)
            return cls(vocab_file=str(path), **kwargs)

        raise FileNotFoundError(f"Could not find vocabulary file in {path}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def print_info(self):
        """Print tokenizer information."""
        print(f"Vocabulary size: {len(self.token2id)}")
        print(f"Special tokens: {self._special_tokens}")
        print(f"  PAD: {self.pad_id}, BOS: {self.bos_id}, EOS: {self.eos_id}, UNK: {self.unk_id}")
        print(f"  SEP: {self.sep_id}")
        print(f"  Task tokens: FWD={self.forward_id}, RETRO={self.retro_id}, ELEM={self.elem_id}, MECH={self.mech_id}")
        print(f"  Role tokens: REACT={self.reactants_id}, REAGENT={self.reagents_id}, COND={self.conditions_id}, PROD={self.products_id}")
        print(f"  Format: use_task_tokens={self.use_task_tokens}, use_role_tokens={self.use_role_tokens}")

    def get_prompt_length(
        self,
        reactants: str,
        task: str = "forward",
    ) -> int:
        """Get prompt length for label masking in causal models."""
        reactant_len = len(self._tokenize(reactants))
        prompt_len = reactant_len
        if self.use_task_tokens:
            prompt_len += 1
        if self.use_role_tokens:
            prompt_len += 2  # <reactants> + <sep>
        return prompt_len

    def fingerprint(self) -> str:
        """Stable fingerprint for tokenizer vocab + format flags."""
        payload = {
            "token2id": self.get_vocab(),
            "use_task_tokens": self.use_task_tokens,
            "use_role_tokens": self.use_role_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class SMILESTokenizerFast(PreTrainedTokenizerFast if HF_AVAILABLE else object):
    """Fast (Rust) SMILES tokenizer using the `tokenizers` backend."""

    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    TASK_TOKENS = SMILESTokenizer.TASK_TOKENS
    ROLE_TOKENS = SMILESTokenizer.ROLE_TOKENS

    def __init__(
        self,
        tokenizer_object=None,
        tokenizer_file: Optional[str] = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        sep_token: str = "<sep>",
        model_max_length: int = 512,
        use_task_tokens: bool = True,
        use_role_tokens: bool = True,
        **kwargs,
    ):
        if PreTrainedTokenizerFast is None:
            raise ImportError("transformers is required for SMILESTokenizerFast")
        if tokenizer_object is None and tokenizer_file is None:
            raise ValueError("Provide tokenizer_object or tokenizer_file")

        self.use_task_tokens = _coerce_bool(use_task_tokens)
        self.use_role_tokens = _coerce_bool(use_role_tokens)
        self._special_tokens = [
            pad_token,
            bos_token,
            eos_token,
            unk_token,
            sep_token,
            "<forward-reaction>",
            "<retrosynthesis>",
            "<elementary-step>",
            "<mechanism>",
            "<reactants>",
            "<reagents>",
            "<conditions>",
            "<products>",
        ]
        # additional_tokens = list(self.TASK_TOKENS.values()) + list(self.ROLE_TOKENS.values())
                # Tokens you always want available
        base_additional = list(self.TASK_TOKENS.values()) + list(self.ROLE_TOKENS.values())

        # If coming from from_pretrained(dir), HF may pass additional_special_tokens via tokenizer_config.json
        cfg_additional = kwargs.pop("additional_special_tokens", None)

        if cfg_additional is None:
            additional_tokens = base_additional
        else:
            # Merge while preserving order and removing duplicates
            additional_tokens = list(dict.fromkeys(list(cfg_additional) + base_additional))


        init_kwargs = dict(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            model_max_length=model_max_length,
            additional_special_tokens=additional_tokens,
            use_task_tokens=self.use_task_tokens,
            use_role_tokens=self.use_role_tokens,
        )
        if tokenizer_object is not None:
            init_kwargs["tokenizer_object"] = tokenizer_object
        if tokenizer_file is not None:
            init_kwargs["tokenizer_file"] = tokenizer_file
        super().__init__(**init_kwargs, **kwargs)

    @property
    def pad_id(self) -> int:
        return self.pad_token_id

    @property
    def bos_id(self) -> int:
        return self.bos_token_id

    @property
    def eos_id(self) -> int:
        return self.eos_token_id

    @property
    def unk_id(self) -> int:
        return self.unk_token_id

    @property
    def sep_id(self) -> int:
        return self.convert_tokens_to_ids("<sep>")

    @property
    def forward_id(self) -> int:
        return self.convert_tokens_to_ids("<forward-reaction>")

    @property
    def retro_id(self) -> int:
        return self.convert_tokens_to_ids("<retrosynthesis>")

    @property
    def elem_id(self) -> int:
        return self.convert_tokens_to_ids("<elementary-step>")

    @property
    def mech_id(self) -> int:
        return self.convert_tokens_to_ids("<mechanism>")

    @property
    def reactants_id(self) -> int:
        return self.convert_tokens_to_ids("<reactants>")

    @property
    def reagents_id(self) -> int:
        return self.convert_tokens_to_ids("<reagents>")

    @property
    def conditions_id(self) -> int:
        return self.convert_tokens_to_ids("<conditions>")

    @property
    def products_id(self) -> int:
        return self.convert_tokens_to_ids("<products>")

    def get_task_id(self, task: str) -> int:
        task_map = {
            "forward": self.forward_id,
            "fwd": self.forward_id,
            "forward-reaction": self.forward_id,
            "retro": self.retro_id,
            "retrosynthesis": self.retro_id,
            "elem": self.elem_id,
            "elementary": self.elem_id,
            "elementary-step": self.elem_id,
            "mech": self.mech_id,
            "mechanism": self.mech_id,
        }
        return task_map.get(task.lower(), self.forward_id)

    def _convert_token_to_id(self, token: str) -> int:
        return self.convert_tokens_to_ids(token)

    def _encode_smiles_ids(self, text: str) -> List[int]:
        if not text:
            return []
        return self.backend_tokenizer.encode(text).ids

    def encode_reaction(
        self,
        reactants: str,
        products: Optional[str] = None,
        task: str = "forward",
        model_type: str = "transformer",
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        use_task_tokens: Optional[bool] = None,
        use_role_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if model_type == "decoder_only":
            return self._encode_causal_format(
                reactants=reactants,
                products=products,
                task=task,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                max_src_length=max_src_length,
                max_tgt_length=max_tgt_length,
                truncation=truncation,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                use_task_tokens=use_task_tokens,
                use_role_tokens=use_role_tokens,
            )
        return self._encode_encoder_decoder_format(
            reactants=reactants,
            products=products,
            task=task,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            max_src_length=max_src_length,
            max_tgt_length=max_tgt_length,
            truncation=truncation,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            use_task_tokens=use_task_tokens,
            use_role_tokens=use_role_tokens,
        )

    def _encode_causal_format(
        self,
        reactants: str,
        products: Optional[str] = None,
        task: str = "forward",
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        use_task_tokens: Optional[bool] = None,
        use_role_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        use_task_tokens = self.use_task_tokens if use_task_tokens is None else use_task_tokens
        use_role_tokens = self.use_role_tokens if use_role_tokens is None else use_role_tokens

        task_id = self.get_task_id(task)
        reactant_ids = self._encode_smiles_ids(reactants)

        prompt_ids = []
        if use_task_tokens:
            prompt_ids.append(task_id)
        if use_role_tokens:
            prompt_ids.append(self.reactants_id)
        prompt_ids += reactant_ids
        if use_role_tokens:
            prompt_ids.append(self.sep_id)

        if products is not None:
            product_ids = self._encode_smiles_ids(products)
            if add_special_tokens:
                input_ids = prompt_ids + product_ids + [self.eos_id]
            else:
                input_ids = prompt_ids + product_ids

            if add_special_tokens:
                labels = [-100] * len(prompt_ids) + product_ids + [self.eos_id]
            else:
                labels = [-100] * len(prompt_ids) + product_ids
        else:
            input_ids = prompt_ids
            labels = None

        if max_length is None and max_src_length is not None and max_tgt_length is not None:
            max_length = max_src_length + max_tgt_length
        max_length = max_length or getattr(self, "model_max_length", 512)
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            if labels is not None:
                labels = labels[:max_length]
            if add_special_tokens and products is not None:
                prompt_len = len(prompt_ids)
                if prompt_len < max_length:
                    input_ids[-1] = self.eos_id
                    if labels is not None:
                        labels[-1] = self.eos_id

        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)
        if labels is not None:
            result["labels"] = labels
        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor([v]) for k, v in result.items()}
        return result

    def _encode_encoder_decoder_format(
        self,
        reactants: str,
        products: Optional[str] = None,
        task: str = "forward",
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        max_src_length: Optional[int] = None,
        max_tgt_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        use_task_tokens: Optional[bool] = None,
        use_role_tokens: Optional[bool] = None,
    ) -> Dict[str, Any]:
        use_task_tokens = self.use_task_tokens if use_task_tokens is None else use_task_tokens
        use_role_tokens = self.use_role_tokens if use_role_tokens is None else use_role_tokens

        task_id = self.get_task_id(task)
        reactant_ids = self._encode_smiles_ids(reactants)

        input_ids = []
        if use_task_tokens:
            input_ids.append(task_id)
        if use_role_tokens:
            input_ids.append(self.reactants_id)
        input_ids += reactant_ids

        src_max_length = max_src_length or max_length or getattr(self, "model_max_length", 512)
        if truncation and len(input_ids) > src_max_length:
            input_ids = input_ids[:src_max_length]

        result = {"src": input_ids, "input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)
            result["src_mask"] = [1] * len(input_ids)

        if products is not None:
            product_ids = self._encode_smiles_ids(products)
            if add_special_tokens:
                tgt_in = [self.bos_id] + product_ids
                tgt_out = product_ids + [self.eos_id]
            else:
                tgt_in = product_ids
                tgt_out = product_ids

            if truncation:
                tgt_max_length = max_tgt_length or max_length or getattr(self, "model_max_length", 512)
                tgt_in = tgt_in[:tgt_max_length]
                tgt_out = tgt_out[:tgt_max_length]
                if add_special_tokens and tgt_out:
                    tgt_out[-1] = self.eos_id

            result["tgt_in"] = tgt_in
            result["tgt_out"] = tgt_out
            result["tgt_mask"] = [1] * len(tgt_in)

        if return_tensors == "pt":
            import torch
            result = {k: torch.tensor([v]) for k, v in result.items()}
        return result

    def get_prompt_length(self, reactants: str, task: str = "forward") -> int:
        reactant_len = len(self._encode_smiles_ids(reactants))
        prompt_len = reactant_len
        if self.use_task_tokens:
            prompt_len += 1
        if self.use_role_tokens:
            prompt_len += 2
        return prompt_len

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
        stop_at_eos: bool = True,
        **kwargs,
    ) -> str:
        special_ids = {
            self.pad_id,
            self.bos_id,
            self.eos_id,
            self.unk_id,
            self.sep_id,
            self.forward_id,
            self.retro_id,
            self.elem_id,
            self.mech_id,
            self.reactants_id,
            self.reagents_id,
            self.conditions_id,
            self.products_id,
        }
        tokens = []
        for idx in ids:
            if stop_at_eos and idx == self.eos_id:
                break
            if skip_special and idx in special_ids:
                continue
            tokens.append(self.convert_ids_to_tokens(idx))
        return "".join(tokens)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "init_kwargs"):
            self.init_kwargs["use_task_tokens"] = self.use_task_tokens
            self.init_kwargs["use_role_tokens"] = self.use_role_tokens
        if path.is_dir():
            self.save_pretrained(str(path))
            return
        if not str(path).endswith(".json"):
            self.save_pretrained(str(path))
            return
        self.backend_tokenizer.save(str(path))
        # Persist tokenizer_config.json so formatting flags reload correctly.
        super().save_pretrained(str(path.parent))

    def save_pretrained(self, save_directory: str, **kwargs):
        if hasattr(self, "init_kwargs"):
            self.init_kwargs["use_task_tokens"] = self.use_task_tokens
            self.init_kwargs["use_role_tokens"] = self.use_role_tokens
        return super().save_pretrained(save_directory, **kwargs)

    def print_info(self):
        print(f"Vocabulary size: {len(self.get_vocab())}")
        print(f"Special tokens: {self._special_tokens}")
        print(f"  PAD: {self.pad_id}, BOS: {self.bos_id}, EOS: {self.eos_id}, UNK: {self.unk_id}")
        print(f"  SEP: {self.sep_id}")
        print(f"  Task tokens: FWD={self.forward_id}, RETRO={self.retro_id}, ELEM={self.elem_id}, MECH={self.mech_id}")
        print(f"  Role tokens: REACT={self.reactants_id}, REAGENT={self.reagents_id}, COND={self.conditions_id}, PROD={self.products_id}")
        print(f"  Format: use_task_tokens={self.use_task_tokens}, use_role_tokens={self.use_role_tokens}")

    def fingerprint(self) -> str:
        payload = {
            "token2id": self.get_vocab(),
            "use_task_tokens": self.use_task_tokens,
            "use_role_tokens": self.use_role_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "SMILESTokenizerFast":
        path = Path(path)
        if path.is_file():
            config_path = path.parent / "tokenizer_config.json"
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text())
                    for key in ("use_task_tokens", "use_role_tokens"):
                        if key not in kwargs and key in cfg:
                            kwargs[key] = cfg[key]
                except (OSError, json.JSONDecodeError):
                    pass
        return cls(tokenizer_file=str(path), **kwargs)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for building SMILES tokenizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Build SMILES tokenizer (fast by default) from reaction files")
    parser.add_argument(
        "files",
        nargs="+",
        help="Input reaction files (can pass multiple, e.g., *_clean_*.txt)",
    )
    parser.add_argument("-o", "--output", required=True, help="Output path for vocabulary")
    parser.add_argument("--min-frequency", type=int, default=1, help="Minimum token frequency")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=_default_num_proc(),
        help="Number of worker processes for token counting",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Lines per chunk sent to each worker",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Use the Python tokenizer and save vocab.json (default is fast tokenizer.json)",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    if not args.slow:
        print("Building fast SMILES tokenizer...")
        if PreTrainedTokenizerFast is None:
            raise ImportError("transformers with tokenizers backend is required for fast tokenization")
        special_tokens = SMILESTokenizer()._special_tokens
        backend = _build_fast_backend(
            files=args.files,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )
        tokenizer = SMILESTokenizerFast(tokenizer_object=backend)
        if output_path.is_dir():
            tokenizer.save_pretrained(str(output_path))
        else:
            tokenizer.save(str(output_path))
    else:
        tokenizer = SMILESTokenizer()
        tokenizer.train(
            files=args.files,
            min_frequency=args.min_frequency,
            num_proc=args.num_proc,
            chunk_size=args.chunk_size,
        )
        if output_path.is_dir():
            tokenizer.save_pretrained(str(output_path))
        else:
            tokenizer.save(str(output_path))

    print(f"Tokenizer saved to: {output_path}")


if __name__ == "__main__":
    main()
