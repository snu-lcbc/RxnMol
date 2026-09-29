# Reported forward-reaction model

This directory provides the training and evaluation procedure for
`ft_synth_to_uspto_mix90_10`, the forward-reaction model selected for the RxnMol
manuscript. It includes Synthetic-RXN pretraining and subsequent fine-tuning on
USPTO-MIT with Synthetic-RXN replay. Other transfer directions and fine-tuning
strategies are outside this package.

## Files and data

- `configs/pretrain.yaml` and `configs/finetune.yaml`: portable training recipes.
- `reference/`: retained training settings, evaluation records, checkpoint identities,
  and download hashes. Original records are preserved separately from portable recipes.
- `tokenizer/`: the saved 233-token vocabulary and tokenizer settings.
- `train.py`, `evaluate.py`, `predict.py`, `src/`: training, evaluation, and inference code.
- `scripts/prepare_data.py`: exact record splitting and replay preparation.
- `data/manifest.json`: counts and SHA-256 hashes for inputs and prepared files.
- `environment/`: CPU installation requirements and the tested environment freeze.
- `validation/`: preparation, vocabulary-coverage, and software-test results.

The asset manifest identifies the two raw reaction corpora and two full
checkpoints by filename and SHA-256 hash. Large files are distributed separately
from the Git source. From this directory, run:

```bash
python scripts/download_assets.py
```

With no arguments, the script downloads both the datasets and checkpoints from
[release v1.1.0](https://github.com/snu-lcbc/RxnMol/releases/tag/v1.1.0), verifies
their SHA-256 hashes, extracts the reaction records into `data/raw/`, and places
the checkpoints and tokenizer in `weights/`. Existing files are verified and
reused. Then reproduce the training splits and replay file:

```bash
python scripts/prepare_data.py --verify-reported
```

Optional download settings:

```bash
# Prepare only one group.
python scripts/download_assets.py --datasets
python scripts/download_assets.py --weights

# Use files already downloaded to a local directory.
python scripts/download_assets.py --local-dir /path/to/downloads

# Select another release containing the same checksum-matched assets.
python scripts/download_assets.py --release-tag TAG
```

The local directory must contain the requested files:
`reaction_datasets.tar.gz`, `synthetic_pretrain.ckpt`, and
`reported_finetune.ckpt`. The script verifies SHA-256 hashes for either source.
The preparation script reproduces
all five saved training/validation/replay files byte for byte. It refuses to
overwrite existing outputs. To use another dataset, provide `--synthetic` and
`--uspto` paths and omit `--verify-reported`. Inputs must contain one
`reactants>>product` record per line. That mode still reports hashes and counts;
it does not imply that a new corpus is equivalent to the reported datasets.

## Training recipe

| Setting | Synthetic-RXN pretraining | Replay fine-tuning |
|---|---:|---:|
| Training records | 2,619,767 | 526,964 |
| Validation records | 13,100 synthetic | 4,767 USPTO |
| GPUs / DDP processes | 4 | 2 |
| Batch size per process | 128 | 128 |
| Gradient accumulation | 1 | 1 |
| Effective batch size | 512 | 256 |
| Peak learning rate | 0.0002 | 0.0001 |
| Warmup optimizer steps | 2,000 | 1,000 |
| Maximum optimizer steps | 120,000 | 30,000 |
| Validation interval, training batches | 2,000 | 1,000 |

Both stages use AdamW (betas 0.9/0.98, epsilon 1e-8, weight decay 0.01),
cosine scheduling with two restart cycles, bfloat16 mixed precision,
label smoothing 0.1, gradient clipping 1.0, and seed 42. Early stopping monitors
validation loss with patience 50 validation checks. The model has eight encoder
and eight decoder layers, dimension 512, eight query heads, two key/value heads,
feed-forward dimension 1,408, dropout 0.1, and configured sequence length 200.
The tokenizer metadata's generic length 512 is not the training sequence limit.

The retained reaction-model training environment reports Python 3.12.2,
PyTorch 2.6.0+cu124, Lightning 2.6.0, Transformers 4.57.3, datasets 4.4.2,
NumPy 1.26.4, pandas 2.2.3, and RDKit 2023.09.6. These are recorded versions,
not a freshly tested GPU installation. Molecular-generation versions are listed
separately in the repository's main README.

Install GPU PyTorch appropriate for the receiving machine and the dependencies
listed under `environment/`. The clean CPU installation below has its own tested
dependency list. Machine-specific environment snapshots are not distributed as
installation files.

```bash
# Inspect settings without starting training.
python train.py --config-path configs --config-name pretrain --cfg job
python train.py --config-path configs --config-name finetune --cfg job

# Synthetic-RXN training from the saved vocabulary.
python train.py --config-path configs --config-name pretrain

# Fine-tuning starts from the released Synthetic-RXN checkpoint.
python train.py --config-path configs --config-name finetune
```

The pretraining configuration was reconstructed from the saved hyperparameters
and training record; the fine-tuning configuration comes from its resolved run
configuration. Portable changes are limited to paths, output handling, and mapping
legacy field names to the supplied training interface. The original scheduler
wrappers are not required. The source includes shared model utilities needed by
imports, but no additional model recipes or checkpoints are distributed.

To fine-tune a newly trained base model, override `runtime.init_from` with its
checkpoint. Checkpoint names and exact weights can differ after retraining.
Changing the GPU count changes the effective batch size unless compensated by
batch size or gradient accumulation. This release does not claim bit-identical
GPU retraining.

## Dataset splits and interpretation

The Synthetic-RXN corpus contains 2,632,867 individual reaction steps, flattened
from generated routes before splitting. A per-record Bernoulli draw using
`random.Random(42)` and validation probability 0.005 produces 2,619,767 training
and 13,100 validation records. Steps of a route are not grouped. The USPTO-MIT
corpus contains 479,035 records and uses the same procedure with probability
0.01, giving 474,268 training and 4,767 validation records. This is an internal
split of the supplied corpus, not the official USPTO-MIT test partition.
Neither corpus was deduplicated before splitting. Shared reactions, products,
or synthetic routes can therefore occur across training and validation.

Replay preparation keeps all 474,268 USPTO training records and appends 52,696
synthetic training records selected by reservoir sampling with seed 42. Synthetic
records comprise approximately 10% of the resulting 526,964-record file; this
is not a 10% sample of the entire synthetic corpus. Validation records are not
intentionally sampled into the replay file, although duplicate records can occur
across the original split.

The author recalls constructing the vocabulary from Synthetic-RXN and USPTO.
The original input list and vocabulary-building log were not recovered. The
saved vocabulary is supplied directly. An independent scan found no unknown
SMILES tokens in either complete input corpus (`validation/tokenizer_coverage.json`).
This establishes coverage of those files, not vocabulary-construction provenance
or coverage of arbitrary future chemistry. Training-only vocabulary construction
cannot be established from the retained records.

## Checkpoints and reported accuracy

| Released filename | Original identity | Purpose |
|---|---|---|
| `synthetic_pretrain.ckpt` | epoch 20, step 104,320, validation loss 0.8727 | Required initialization for fine-tuning |
| `reported_finetune.ckpt` | epoch 14, step 29,812, validation loss 0.8856 | Checkpoint used for reported accuracy |

Fine-tuning uses weights-only initialization, not optimizer-state continuation.
Greedy top-1 accuracy after canonicalizing product SMILES was 73.8931% on
Synthetic-RXN and 84.7913% on USPTO internal validation. The retained evaluation
JSON files identify the checkpoint and sample counts. These are internal
validation results, not an independent test of unseen routes or a comparison
against published official USPTO-MIT test results.

```bash
python evaluate.py --checkpoint weights/reported_finetune.ckpt \
  --data data/synthetic_rxn/val_rxn.txt --beam-size 1 --device cuda \
  --output evaluation_synthetic.json
python evaluate.py --checkpoint weights/reported_finetune.ckpt \
  --data data/uspto-mit/val_rxn.txt --beam-size 1 --device cuda \
  --output evaluation_uspto.json
```

The existing [Zenodo weights archive](https://doi.org/10.5281/zenodo.20444600)
contains `reaction/model.pt`, the final step-30,000 weights of the same run.
That file and the reported step-29,812 checkpoint are distinct artifacts;
`reference/checkpoints.json` records their identities and hashes. The archive's
older description reverses the transfer direction: the correct direction is
Synthetic-RXN to USPTO-MIT with synthetic replay.

## CPU installation check

```bash
python3.11 -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environment/requirements-cpu.txt
python scripts/smoke_test.py --report cpu_smoke.json
```

The smoke test checks dependencies, both training configurations, two optimizer
steps on a small model, checkpoint reload, and two-sample evaluation. It uses
one RDKit distribution. These checks do not reproduce full training, the reported
accuracy, GPU performance, or the full RxnMol objective stack.

## Data sources

USPTO-MIT is attributed to Jin et al. (2017), *Predicting Organic Reaction
Outcomes with Weisfeiler-Lehman Network* (the manuscript's USPTO-MIT reference).
Synthetic-RXN was generated from Enamine building blocks using the 115-template
library attributed in the manuscript to Gao et al. (2022); 56 distinct templates
occur in the retained generated records. The release distributes the exact flat
reaction corpora used for training. It does not claim to regenerate the original
synthetic route population, and does not include the vendor SDF or unrelated
training datasets.
