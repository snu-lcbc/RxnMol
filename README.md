# RxnMol: Fragment-Based Synthesis Path Optimization

**Synthesis-Aware Molecular Optimization: A Data-Driven Approach to de novo Molecular Generation**

RxnMol is a novel molecular optimization framework that operates on **synthesis routes** rather than molecular structures directly. By representing candidates as ordered sequences of commercially available building blocks (fragments), RxnMol ensures that every generated molecule comes with a valid, step-by-step synthesis pathway predicted by a neural reaction model.

## Reproducing the reported experiments

The [reaction-model package](reaction_model/README.md) contains the selected
Synthetic-RXN to USPTO-MIT replay model's training recipes, saved tokenizer,
dataset preparation, evaluation code, and versioned dataset/checkpoint links.
The [reproducibility documentation](reproducibility/README.md) maps saved
generation configurations to effective settings, distinguishes recorded
environments from the tested CPU installation, and documents building-block
preparation. Large training files are supplied through the
[reproducibility release](https://github.com/snu-lcbc/RxnMol/releases/tag/r2-reproducibility-2026-09-29).

## Research Overview

### The Core Idea
Traditional de novo design methods often generate "unmakeable" molecules. RxnMol solves this by optimizing the **recipe** instead of the **result**.

- **Genotype**: Ordered list of fragments `[F₁, F₂, ..., Fₙ]`
- **Phenotype**: The molecule resulting from reacting these fragments sequentially `F₁ + F₂ → I₁ + F₃ → ... → Final Molecule`
- **Optimization**: Conformational Space Annealing (CSA) searches the combinatorial space of fragment sequences.

### Key Features
- **Synthesizability**: All intermediates and final products are validated by a forward reaction prediction model.
- **Explicit Pathways**: Output includes the exact recipe (reactants and order) to make the molecule.
- **Controllable Complexity**: Route length constraints (e.g., 2-5 steps) directly control synthetic complexity.
- **High Performance**: GPU-accelerated batch reaction prediction and caching.

## Installation
### - Prerequisite: Install PyTorch & TensorRT
The reaction predictor relies on **Torch-TensorRT** for fast inference code. You must install the version of PyTorch and TensorRT compatible with your specific CUDA version (e.g., CUDA 12.x).

Recommend to setup a new environment before installing.
```bash
# Clone the repository
git clone https://github.com/snu-lcbc/RxnMol.git
cd RxnMol

# Install dependencies
pip install -r requirements.txt
# Or if you need torch=2.4.0 wit cuda=12.1
# pip install -r requirements_full.txt

# Install the package in development mode
pip install -e .
```
> **Note**: TDC requires `numpy<2.0` and `scikit-learn==1.2.2`. These are pinned to avoid compatibility issues with TDC's pre-trained models.

### Manual Installations:
If you need to install a specific PyTorch and CUDA version (`torch=2.4.0` `cuda=12.1`): 
```bash
pip install torch==2.4.0 torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
```
> **Note**: Install `torch` and torch-related packages with compatible versions matching your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) for details.


#### Extra GPU stacks for other methods used in this study.
- **CSearch** needs DGL libs compatible with the same torch/CUDA:
```bash
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
```
- **SynFlowNet** needs PyG libs matching torch 2.4.0 + CUDA 12.1:
```bash
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```
> These GPU libraries are ABI‑tied to specific CUDA/torch builds. If your CUDA version differs, use the corresponding wheel URLs/versions so everything stays compatible.

## Architecture

The codebase is organized in `src/rxnmol`:

```
src/rxnmol/
├── core/
│   ├── engine.py          # CSAEngine: The optimization loop
│   ├── data_models.py     # Candidate, RunContext
│   └── config.py          # Configuration management
├── solutions/
│   ├── fragments_route.py # FragmentRouteSpec: Synthesis logic
│   ├── smiles_based.py    # SmilesDirectSpec: Raw SMILES optimization
│   └── reaction_model.py  # RxnPredictor: Transformer model
├── objectives/            # Objective functions (GuacaMol, TDC, etc.)
├── runtime/               # Metrics and artifact storage
└── utils/                 # Helper utilities
```

## Usage

### Running an Experiment

```bash
python run.py --config configs.yaml --objective seh
```

### Configuration Example

```yaml
# configs.yaml
csa:
  bank_size: 100
  seed_size: 60
  max_iter: 100

solution:
  spec_type: fragment_route  
  min_fragments: 2
  max_fragments: 5

objective:
  name: seh # name of objective functions 

runtime:
  device: cuda    # Use GPU for reaction prediction
```

### Code Example

```python
from rxnmol.core import MasterConfig, RunContext, CSAEngine
from rxnmol.solutions import FragmentRouteSpec
import numpy as np

# 1. Load Configuration
config = MasterConfig.from_yaml("configs.yaml")

# 2. Initialize Context
rng = np.random.default_rng(42)
context = RunContext(config=config, rng=rng)

# 3. Initialize Solution Specification
spec = FragmentRouteSpec(context)

# 4. Initialize Engine
engine = CSAEngine(spec, context)

# 5. Run Optimization
best_candidate = engine.run()

print(f"Best Score: {best_candidate.objective_value}")
print(f"Molecule SMILES: {best_candidate.smiles}")
print(f"Synthesis Route: {best_candidate.genotype}")
```

### Available Objectives

| Category | Objectives |
|----------|-----------|
| Standard | `qed`, `sa_score`, `logp`, `lipinski` |
| GuacaMol | `zaleplon`, `osimertinib`, `fexofenadine`, `ranolazine`, ... |
| TDC | `jnk3`, `gsk3b`, `drd2`, `seh`, ... |
| Custom | Define your own via `ObjectiveFunction` base class |

Check all available objective functions:
```bash
python src/rxnmol/objectives/registry.py
```

## Model Weights

The reaction prediction model weights are required for fragment-based optimization. Download from:

- [Deployment, proxy, and docking weights](https://doi.org/10.5281/zenodo.20444600)
- [Reported reaction-model checkpoint, pretraining checkpoint, and datasets](https://github.com/snu-lcbc/RxnMol/releases/tag/r2-reproducibility-2026-09-29)

The evaluated checkpoint and the final deployment weights are identified separately
in [reaction_model/README.md](reaction_model/README.md).


## License

MIT License

## Citation

If you use RxnMol in your research, please cite:

```bibtex
@article{rxnmol2025,
  title={Synthesis-Aware Molecular Optimization: A Data-Driven Approach to de novo Molecular Generation},
  author={Ashyrmamatov, Islambek and Ucak, Umit V. and Lee, Juyong},
  year={2025}
}
```
