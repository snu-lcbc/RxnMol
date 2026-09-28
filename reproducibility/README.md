# Reproducibility resources

The configuration documentation describes the fragment-route implementation
used in the reported RxnMol experiments. The manuscript uses 19 tasks and a
primary bank size of 1,000 with routes containing two to five building blocks.

## Configurations and effective settings

`configurations/run_manifest.csv` maps the available saved configurations to
the repaired-results provenance for the 19 manuscript tasks and the retained
route-length configurations. Missing original configurations are explicitly
marked; they are not replaced with invented historical files. Configurations
retain their original paths and must be adapted before execution. The three
`example_bank*.yaml` files are unedited saved ALK configurations illustrating the
bank-size settings, not universal configurations for every objective.

| Configuration key | Bank 100 | Bank 500 | Bank 1,000 |
|---|---:|---:|---:|
| `csa.bank_size` | 100 | 500 | 1,000 |
| `csa.seed_size` | 60 | 300 | 600 |
| `csa.max_iter` | 100 | 500 | 500 |
| `csa.early_stop_patience` | 50 | 50 | 50 |
| `csa.d_init` | 0.6 | 0.6 | 0.6 |
| `csa.gamma` | 0.99 | 0.99 | 0.99 |
| `csa.d_min` | 0.1 | 0.1 | 0.1 |
| Maximum attempted offspring per generation | 1,500 | 7,500 | 15,000 |

Each seed undergoes `csa.n_cross_op1=5` crossover attempts against the current
bank, `csa.n_cross_op2=5` against the initial bank, and `csa.n_mut_op=5` mutation
attempts. Each crossover can return two children and each mutation one. Thus
25 is an upper bound on attempted children per seed, not an evaluation count.
Invalid, rejected, and repeated candidates affect the relationship between
attempts, evaluations, and unique products.

The fragment-route implementation supplies two crossover operators and three
mutation operators. `CSAEngine` samples uniformly from each list: crossover
probabilities are 1/2 and mutation probabilities are 1/3. The generic
`operators.crossover_prob`, `operators.mutation_*_prob`, `operators.*_weights`,
and configured operator-name lists do not determine this fragment-route selection.
They must not be interpreted as the effective probabilities of these runs.
Mutation positions use weight 0.2 for the first building block and 1.0 for other
positions. This position weighting is separate from selection of mutation type.

The initial bank receives one unconditional local-refinement pass, with
`csa.local_optimization_max_steps=3` and `csa.local_optimization_neighbors=5`.
`csa.local_optimization_enabled=false` disables refinement in the main loop;
it does not disable the initial pass. `csa.local_optimization_frequency` is
inactive when in-loop refinement is disabled.

Search stops at the maximum generation count, after 50 generations without an
improvement in the best objective, or when no unused bank members remain as
seeds. Used seed markers are not globally reset to continue a converged run.
Accepted new bank members are marked unused. Normal termination writes the final
bank. Generation limits are upper bounds, not the number every run completed.
Matched-budget analyses instead use the saved bank selected by their declared
evaluation-count rule.

`solution.min_fragments=2` and `solution.max_fragments=5` define the primary
route length in building blocks (one to four forward steps). The route-length
study also uses maximum lengths four and six. `solution.num_step` is not a
replacement for these fragment-route bounds. Original search configurations
set `runtime.random_seed=null`; repair and additional ablation runs can have
recorded seeds and must be identified separately.

The selected predictor is `transformer_v2`, with greedy decoding
(`reaction_model.beam_size=1`). The saved generation setting `max_len=256`
is the inference source/decode limit passed by the encoder-decoder adapter;
it is distinct from the training source and target limits of 200.
`objective.validity_penalty=999.0` is the internal minimization
penalty for invalid candidates, not a molecular objective value to report.

Source anchors: `src/rxnmol/core/engine.py` (`run`, `_generate_initial_bank`,
`_apply_crossover`, `_apply_mutation`), `src/rxnmol/solutions/fragments_route.py`
(operator lists and position weighting), and the saved configurations.

## Software environments

`environments/*_pip_freeze.txt` are unedited snapshots captured on 2026-09-29
from the retained `rxnmol` and `torch-cu12.4` environments. The accompanying
metadata files identify Python and installed distributions. They document the
retained environments, not proof that every package was unchanged since the
original runs. Conda build paths and duplicate distributions are preserved.

The RxnMol environment contains both `rdkit==2023.9.6` and
`rdkit-pypi==2022.9.5`; the imported RDKit reports 2023.09.6. Do not install
both distributions in a new environment. Principal recorded generation versions
are guacamol 0.5.5, PyTDC 1.1.15, PyTorch 2.4.0+cu121, NumPy 1.26.4,
SciPy 1.16.3, scikit-learn 1.2.2, and pandas 2.3.3.

A separate clean CPU installation and integration test is supplied in
`reaction_model/environment/` and `reaction_model/validation/`. Its tested
dependency freeze is distinct from these historical-environment snapshots.
It validates the released training interface and building-block preparation,
not GPU retraining or all generation objectives.

## Building-block preparation

The reported search used the released `data/enamine_fragments.smi`, without
additional molecular-weight, formal-charge, or reactive-substructure filtering.
The new `scripts/prepare_building_blocks.py` supports SDF and SMILES inputs,
RDKit parsing/sanitization, hydrogen removal, canonical isomeric SMILES output,
and explicit per-record reporting. Default settings preserve salts, charges,
stereochemistry, and duplicate valid entries. Optional standardization and
filters are for preparing alternative libraries; they were not applied to the
reported search library.

```bash
python scripts/prepare_building_blocks.py --input data/enamine_fragments.smi \
  --output prepared/enamine_fragments.smi --report prepared/report.json \
  --records prepared/records.csv
```

Reprocessing the released SMILES retains 259,380 entries and reproduces its bytes.
Converting the retained vendor SDF also retains 259,380 entries, including 15,278
multicomponent entries. It does not reproduce file order or byte identity.
Two SMILES representations differ but agree after reparsing and canonicalization;
the other 259,378 strings already match as a set. This verifies structural
agreement under that comparison, not recovery of the original conversion or
shuffle procedure. The supplied reports keep these checks distinct.
