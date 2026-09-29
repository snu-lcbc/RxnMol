"""
Fragment-based synthesis route solution specification.

This module implements the FragmentRouteSpec, which represents molecules as
ordered lists of fragments that are sequentially reacted to produce the final molecule.
"""

from typing import List, Tuple, Callable, Optional, Dict, Any
import logging
from pathlib import Path
import copy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from .base import SolutionSpec, tanimoto_distance
from ..core.data_models import Candidate, RunContext
from .cache import PersistentReactionCache, compute_reaction_model_cache_id
from .reaction_models import load_reaction_model
from ..utils.building_blocks import load_building_blocks

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# Substrings marking a CUDA/device fault that corrupts the context for the rest of
# the process. Retrying after one of these always fails, so they must abort the run
# rather than be absorbed into per-candidate "FAILED" results. Plain OOM is excluded
# on purpose: RxnPredictor.predict_batch recovers from it by shrinking the batch.
_FATAL_DEVICE_ERROR_MARKERS = (
    "cuda error",
    "cuda driver error",
    "device-side assert",
    "illegal memory access",
    "initialization error",
    "no kernel image",
    "cublas",
    "cudnn",
    "nccl",
)


def _is_fatal_device_error(exc: BaseException) -> bool:
    """True if `exc` signals an unrecoverable GPU/device fault."""
    message = str(exc).lower()
    if "out of memory" in message:
        return False
    return any(marker in message for marker in _FATAL_DEVICE_ERROR_MARKERS)


class FragmentRouteSpec(SolutionSpec):
    """
    Solution specification for fragment-based synthesis routes.

    Genotype: List of fragment SMILES strings
    Phenotype: Final product molecule from sequential reactions
    """

    def __init__(self, context: RunContext):
        """
        Initialize fragment route specification.

        Args:
            context: Shared runtime context
        """
        super().__init__(context)

        # Load fragment library
        self.building_blocks_file = self.config.data.building_blocks_file
        self.building_blocks = self._load_building_blocks(self.building_blocks_file)

        # Persistent cache with multi-process support
        # - Reads from shared DB (read-only, immutable - no WAL/SHM access)
        # - Writes to process-local DB (for post-run merge)
        # - If local DB exists (resume), loads into memory at startup
        # - After all experiments complete, run: python -m rxnmol.solutions.cache merge
        # Use just the last directory name from output_dir (e.g., "seh_mf2-5_r1")
        model_cache_id = compute_reaction_model_cache_id(self.config.reaction_model)
        run_name = Path(self.config.output_dir).name if self.config.output_dir else None
        self.reaction_cache = PersistentReactionCache(
            shared_db_path="./cache/reaction_cache_shared.db",
            local_db_dir="./cache/local",
            run_name=run_name,
            copy_shared_to_local=False,  # Disabled - local copies caused Bus errors
            model_id=model_cache_id,
            auto_merge_on_close=self.config.persistence.cache_auto_merge,
            auto_merge_delete=self.config.persistence.cache_auto_merge_delete,
            cache_max_entries=self.config.persistence.cache_max_entries,
        )

        # Load reaction predictor (shared via context)
        self.rxn_predictor = context.reaction_model
        if self.rxn_predictor is None:
            logger.info("Loading reaction prediction model...")
            self.rxn_predictor = load_reaction_model(
                self.config.reaction_model,
                device=self.config.runtime.device,
            )
            self.context.reaction_model = self.rxn_predictor
        logger.info("Reaction prediction model ready")
        
        # Genetic operators
        # Both crossover operators produce 2 children each (swap both halves)
        self.crossover_ops = [
            self._crossover_one_point,  # Swaps suffix portions
            self._crossover_two_point,  # Swaps middle segments, preserves scaffold and terminal
        ]
        # All mutation operators use scaffold-aware position weighting
        self.mutation_ops = [
            self._mutate_replace,  # Replace fragment at weighted position
            self._mutate_add,      # Insert fragment at weighted position
            self._mutate_remove,   # Remove fragment from weighted position
        ]

        # Create objective instance ONCE
        self.objective = self._create_objective()

    # Reaction predictor is now loaded immediately in __init__, no lazy loading needed

    def _create_objective(self):
        from ..objectives import get_objective
        return get_objective(
            name=self.context.config.objective.name,
            validity_penalty=self.context.config.objective.validity_penalty
        )

    # ========================================================================
    # SolutionSpec Abstract Methods
    # ========================================================================

    def random_candidates(self, n: int, generation: int = 0) -> List[Candidate]:
        """Generate n random candidates efficiently with zero waste."""
        # 1. Sample fragment counts
        lengths = self.rng.integers(
            self.config.solution.min_fragments,
            self.config.solution.max_fragments + 1,
            size=n
        )

        # 2. Sample indices (faster than sampling objects)
        total_frags = lengths.sum()
        indices = self.rng.integers(0, len(self.building_blocks), size=total_frags)

        # 3. Index into building blocks array
        all_fragments = self.building_blocks[indices]

        # 4. Split into variable-length genotypes using cumsum
        split_points = np.cumsum(lengths)[:-1]
        genotypes = np.split(all_fragments, split_points)

        # 5. Build candidates
        return [
            Candidate(
                genotype=g.tolist(),
                genotype_type='fragment_route',
                metadata={'generation': generation, 'is_seed': True}
            )
            for g in genotypes
        ]

    def build(self, candidate: Candidate) -> Candidate:
        """Build phenotype for a single candidate (wrapper for batch)."""
        return self.build_batch([candidate])[0]

    def build_batch(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Build phenotypes for a batch of candidates using GPU-accelerated batch prediction.

        OPTIMIZED: Greedy cache resolution with batched GPU prediction.

        Algorithm:
        1. Each candidate advances independently through cached reactions
        2. When a candidate hits a cache miss, it's marked as 'blocked'
        3. After all candidates are either complete, failed, or blocked:
           - Collect ALL blocked reactions (deduplicated)
           - Predict them in ONE GPU batch
           - Update blocked candidates and continue

        This maximizes GPU batch size and cache utilization by:
        - Advancing cache-hit candidates immediately (no waiting)
        - Batching ALL cache misses across ALL levels together
        - Deduplicating identical reactions (predicted once, shared by multiple candidates)
        """
        if not candidates:
            return []

        # Initialize state for each candidate
        # State dict is local to this method only
        states = []
        for i, cand in enumerate(candidates):
            frags = cand.genotype
            if not frags:
                states.append({
                    'idx': i,
                    'fragments': [],
                    'current_product': None,
                    'next_frag_idx': 0,
                    'intermediates': [],
                    'reactions': [],
                    'status': 'failed',
                    'reason': 'Empty fragment list'
                })
            else:
                states.append({
                    'idx': i,
                    'fragments': frags,
                    'current_product': frags[0],  # First fragment is the scaffold
                    'next_frag_idx': 1,           # Next fragment to react with
                    'intermediates': [frags[0]],
                    'reactions': [],
                    'status': 'active',
                    'reason': None
                })

        # Main loop: greedy cache resolution + batched GPU prediction
        iteration = 0
        while any(s['status'] in ('active', 'blocked') for s in states):
            iteration += 1

            # =================================================================
            # PHASE 1: Greedy cache resolution
            # Advance each candidate as far as cache allows
            # =================================================================
            blocked = {}  # reaction_key -> [state_indices waiting for this reaction]

            progress = True
            while progress:
                progress = False
                keys_to_check = []
                state_indices_for_keys = []

                for i, s in enumerate(states):
                    if s['status'] != 'active':
                        continue

                    # Check if candidate is complete
                    if s['next_frag_idx'] >= len(s['fragments']):
                        s['status'] = 'complete'
                        continue

                    # Build cache key for next reaction
                    r1 = s['current_product']
                    r2 = s['fragments'][s['next_frag_idx']]
                    key = f"{r1}.{r2}"
                    keys_to_check.append(key)
                    state_indices_for_keys.append(i)

                if not keys_to_check:
                    break

                # Batch cache lookup
                products = self.reaction_cache.get_batch(keys_to_check)

                for state_idx, key, product in zip(state_indices_for_keys, keys_to_check, products):
                    s = states[state_idx]

                    if product is not None:
                        # Cache hit - validate and advance
                        clean = self._validate_and_clean_product(product)
                        if clean:
                            s['current_product'] = clean
                            s['intermediates'].append(clean)
                            s['reactions'].append(f"{key}>>{clean}")
                            s['next_frag_idx'] += 1
                            progress = True  # Made progress, continue loop
                        else:
                            s['status'] = 'failed'
                            s['reason'] = f"Invalid cached product: {product}"
                    else:
                        # Cache miss - mark as blocked
                        s['status'] = 'blocked'
                        blocked.setdefault(key, []).append(state_idx)

            # =================================================================
            # PHASE 2: Batch GPU prediction for ALL blocked reactions
            # =================================================================
            if not blocked:
                break  # All candidates are complete or failed

            # Deduplicate: each unique reaction is predicted only once
            unique_keys = list(blocked.keys())

            logger.debug(
                f"Iteration {iteration}: {len(unique_keys)} unique reactions to predict "
                f"(from {sum(len(v) for v in blocked.values())} blocked candidates)"
            )

            # Single GPU batch for all unique reactions
            predicted_products = self._predict_missing_with_fallback(unique_keys)

            # Cache the results
            self.reaction_cache.set_batch(unique_keys, predicted_products)

            # Update all blocked candidates
            for key, product in zip(unique_keys, predicted_products):
                clean = self._validate_and_clean_product(product)

                for state_idx in blocked[key]:
                    s = states[state_idx]
                    if clean:
                        s['current_product'] = clean
                        s['intermediates'].append(clean)
                        s['reactions'].append(f"{key}>>{clean}")
                        s['next_frag_idx'] += 1
                        s['status'] = 'active'  # Back to active for next iteration
                    else:
                        s['status'] = 'failed'
                        s['reason'] = f"Prediction failed: {key} -> {product}"

        # =================================================================
        # Finalize candidates from states
        # =================================================================
        for s in states:
            cand = candidates[s['idx']]

            if s['status'] == 'complete':
                final_smi = s['current_product']
                mol = Chem.MolFromSmiles(final_smi)
                if mol:
                    cand.smiles = final_smi
                    cand.phenotype_mol = mol
                    cand.intermediates = s['intermediates']
                    cand.metadata['reactions'] = s['reactions']
                    cand.is_valid = True
                    cand.fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius=2, nBits=2048, useChirality=True
                    )
                else:
                    cand.is_valid = False
                    cand.metadata['failure_reason'] = f"Invalid final SMILES: {final_smi}"
                    cand.phenotype_mol = None
                    cand.smiles = None
            else:
                cand.is_valid = False
                cand.metadata['failure_reason'] = s['reason'] or 'Unknown failure'
                cand.phenotype_mol = None
                cand.smiles = None

        valid_count = sum(1 for c in candidates if c.is_valid)
        invalid_count = len(candidates) - valid_count
        logger.debug(
            f"✓ Build complete: {valid_count}/{len(candidates)} valid "
            f"({valid_count/len(candidates)*100:.1f}%), {invalid_count} failed, "
            f"{iteration} iteration(s)"
        )

        return candidates

    def _batch_predict_with_cache(self, reaction_keys: List[str], description: str = "") -> List[str]:
        """
        Batch predict reactions with cache lookup.

        Args:
            reaction_keys: List of "reactant1.reactant2" strings
            description: Description for logging

        Returns:
            List of product SMILES (or "FAILED" for failed predictions)
        """
        if not reaction_keys:
            return []

        # 1. Check cache
        products = self.reaction_cache.get_batch(reaction_keys)

        # 2. Identify missing
        missing_indices = [i for i, p in enumerate(products) if p is None]
        missing_inputs = [reaction_keys[i] for i in missing_indices]

        cache_hits = len(reaction_keys) - len(missing_inputs)
        hit_rate = (cache_hits / len(reaction_keys) * 100) if reaction_keys else 0

        logger.debug(f"📊 {description}: cache {cache_hits}/{len(reaction_keys)} ({hit_rate:.1f}%), GPU {len(missing_inputs)}")

        # 3. Predict missing via RxnPredictor (which already retries on OOM)
        if missing_inputs:
            predicted_products = self._predict_missing_with_fallback(missing_inputs)

            # Never persist the "FAILED" sentinel: a cached failure is served to
            # every later run as if it were a real prediction, so one bad job
            # poisons the shared cache permanently.
            cacheable = [
                (key, pred)
                for key, pred in zip(missing_inputs, predicted_products)
                if pred != "FAILED"
            ]
            if cacheable:
                self.reaction_cache.set_batch(
                    [k for k, _ in cacheable], [p for _, p in cacheable]
                )

            for idx, pred in zip(missing_indices, predicted_products):
                products[idx] = pred

        return products

    def _predict_missing_with_fallback(self, inputs: List[str]) -> List[str]:
        """
        Predict missing reactions, retrying input-level failures individually.

        Propagate device failures and an entirely failed fallback so an unusable
        reaction model cannot silently produce a normal final bank.
        """
        try:
            return self.rxn_predictor.predict_batch(inputs)
        except Exception as e:
            if _is_fatal_device_error(e):
                logger.critical(
                    "Unrecoverable device fault during batch prediction of %d reactions: %s. "
                    "Aborting the run instead of degrading silently.",
                    len(inputs),
                    e,
                )
                raise RuntimeError(
                    f"Unrecoverable device fault in reaction model: {e}"
                ) from e

            logger.error(
                f"Batch prediction failed for {len(inputs)} reactions ({e}); falling back to sequential"
            )
            results = []
            n_failed = 0
            for inp in inputs:
                try:
                    results.append(self.rxn_predictor.predict(inp))
                except Exception as single_err:
                    if _is_fatal_device_error(single_err):
                        logger.critical(
                            "Unrecoverable device fault during sequential fallback: %s. "
                            "Aborting the run.",
                            single_err,
                        )
                        raise RuntimeError(
                            f"Unrecoverable device fault in reaction model: {single_err}"
                        ) from single_err
                    logger.debug("Sequential prediction failed for %s: %s", inp, single_err)
                    results.append("FAILED")
                    n_failed += 1

            if n_failed == len(inputs):
                raise RuntimeError(
                    f"All {len(inputs)} reaction predictions failed in the sequential "
                    "fallback; the reaction model is not usable. Aborting the run."
                )
            if n_failed:
                logger.warning(
                    "%d/%d reaction predictions failed in sequential fallback",
                    n_failed,
                    len(inputs),
                )
            return results

    def compute_fingerprint(self, candidate: Candidate) -> Optional[Any]:
        """Return fingerprint."""
        return candidate.fingerprint

    def distance(self, cand1: Candidate, cand2: Candidate) -> float:
        """Compute distance."""
        if cand1.fingerprint is None or cand2.fingerprint is None:
            return 1.0
        return tanimoto_distance(cand1.fingerprint, cand2.fingerprint)

    def get_operators(self) -> Tuple[List[Callable], List[Callable]]:
        return (self.crossover_ops, self.mutation_ops)

    def get_genotype_type(self) -> str:
        return "fragment_route"

    # ========================================================================
    # Helpers
    # ========================================================================

    def _load_building_blocks(self, building_blocks_file: str) -> List[str]:
        logger.debug(f"Loading building blocks from: {building_blocks_file}")
        return load_building_blocks(building_blocks_file)

    def _validate_and_clean_product(self, product_smiles: str) -> Optional[str]:
        if not product_smiles or product_smiles == "FAILED":
            return None

        try:
            mol = Chem.MolFromSmiles(product_smiles)
            if mol is None:
                return None

            if '.' in product_smiles:
                products = product_smiles.split('.')
                mols = [Chem.MolFromSmiles(p) for p in products]
                mols = [m for m in mols if m is not None]
                if not mols:
                    return None
                mol = max(mols, key=lambda m: m.GetNumAtoms())

            mol = self._neutralize_atoms(mol)

            # Check minimum size: must have at least one bond (no single atoms)
            # Single atoms like [Mg], Cl, [Na] are byproducts, not valid products
            if mol.GetNumBonds() == 0:
                return None

            # Check for unsupported bond types (allow only organic/covalent bonds)
            # DATIVE bonds (metal-ligand) and other exotic types will crash the GNN
            SUPPORTED_BOND_TYPES = {
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            }
            for bond in mol.GetBonds():
                if bond.GetBondType() not in SUPPORTED_BOND_TYPES:
                    return None

            # Check size constraint
            if self.config.solution.max_mol_size is not None:
                if mol.GetNumAtoms() > self.config.solution.max_mol_size:
                    return None

            return Chem.MolToSmiles(mol)

        except Exception:
            return None

    @staticmethod
    def _neutralize_atoms(mol):
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]

        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol

    # ========================================================================
    # Genetic Operators
    # ========================================================================


    def _crossover_one_point(self, parent1: Candidate, parent2: Candidate) -> List[Candidate]:
        """
        One-point crossover producing TWO children by swapping both halves.

        Given parents:
            Parent1: [A, B, C, D]  cut at position 2
            Parent2: [X, Y, Z]    cut at position 1

        Produces:
            Child1: [A, B] + [Y, Z] = [A, B, Y, Z]  (prefix from P1 + suffix from P2)
            Child2: [X] + [C, D]   = [X, C, D]      (prefix from P2 + suffix from P1)

        Args:
            parent1: First parent candidate
            parent2: Second parent candidate

        Returns:
            List of 0-2 valid child candidates
        """
        frags1 = parent1.genotype
        frags2 = parent2.genotype

        if not frags1 or not frags2:
            return []

        # Select cut points (cut1 >= 1 to always include at least one fragment from prefix)
        cut1 = self.rng.integers(1, len(frags1))  # [1, len-1] to ensure both parts non-empty
        cut2 = self.rng.integers(1, len(frags2))  # [1, len-1] to ensure both parts non-empty

        # Child 1: prefix from parent1 + suffix from parent2
        child1_geno = list(frags1[:cut1]) + list(frags2[cut2:])

        # Child 2: prefix from parent2 + suffix from parent1
        child2_geno = list(frags2[:cut2]) + list(frags1[cut1:])

        children = []
        generation = max(
            parent1.metadata.get('generation', 0),
            parent2.metadata.get('generation', 0)
        ) + 1

        for i, geno in enumerate([child1_geno, child2_geno]):
            # Validate length constraints
            if len(geno) > self.config.solution.max_fragments:
                geno = geno[:self.config.solution.max_fragments]

            if self.config.solution.min_fragments <= len(geno) <= self.config.solution.max_fragments:
                children.append(Candidate(
                    genotype=geno,
                    genotype_type='fragment_route',
                    metadata={
                        'generation': generation,
                        'parents': [parent1.smiles, parent2.smiles],
                        'operator': 'crossover_one_point',
                        'child_index': i + 1,  # 1 or 2
                        'cut_points': (cut1, cut2),
                        'is_seed': False
                    }
                ))

        return children

    def _crossover_two_point(self, parent1: Candidate, parent2: Candidate) -> List[Candidate]:
        """
        Two-point crossover producing TWO children by swapping middle segments.

        Given parents:
            Parent1: [A, B, C, D, E]  cuts at positions 1 and 3
            Parent2: [X, Y, Z, W]     cuts at positions 1 and 3

        Produces:
            Child1: [A] + [Y, Z] + [D, E] = [A, Y, Z, D, E]  (middle from P2)
            Child2: [X] + [B, C] + [W]    = [X, B, C, W]     (middle from P1)

        This preserves both the scaffold (first fragment) and terminal chemistry
        while mixing the middle building blocks.

        Args:
            parent1: First parent candidate
            parent2: Second parent candidate

        Returns:
            List of 0-2 valid child candidates
        """
        frags1 = parent1.genotype
        frags2 = parent2.genotype

        if not frags1 or not frags2:
            return []

        # Need at least 3 fragments for meaningful two-point crossover
        if len(frags1) < 3 or len(frags2) < 3:
            # Fall back to one-point crossover for short routes
            return self._crossover_one_point(parent1, parent2)

        # Select two cut points for parent1: 0 < cut1_a < cut1_b < len(frags1)
        cut1_a = self.rng.integers(1, len(frags1) - 1)  # At least 1, at most len-2
        cut1_b = self.rng.integers(cut1_a + 1, len(frags1))  # At least cut1_a+1, at most len-1

        # Select two cut points for parent2: 0 < cut2_a < cut2_b < len(frags2)
        cut2_a = self.rng.integers(1, len(frags2) - 1)
        cut2_b = self.rng.integers(cut2_a + 1, len(frags2))

        # Child 1: prefix1 + middle2 + suffix1
        # [A] + [Y, Z] + [D, E] where parent1=[A,B,C,D,E], parent2=[X,Y,Z,W]
        child1_geno = list(frags1[:cut1_a]) + list(frags2[cut2_a:cut2_b]) + list(frags1[cut1_b:])

        # Child 2: prefix2 + middle1 + suffix2
        # [X] + [B, C] + [W]
        child2_geno = list(frags2[:cut2_a]) + list(frags1[cut1_a:cut1_b]) + list(frags2[cut2_b:])

        children = []
        generation = max(
            parent1.metadata.get('generation', 0),
            parent2.metadata.get('generation', 0)
        ) + 1

        for i, geno in enumerate([child1_geno, child2_geno]):
            # Validate length constraints
            if len(geno) > self.config.solution.max_fragments:
                geno = geno[:self.config.solution.max_fragments]

            if self.config.solution.min_fragments <= len(geno) <= self.config.solution.max_fragments:
                children.append(Candidate(
                    genotype=geno,
                    genotype_type='fragment_route',
                    metadata={
                        'generation': generation,
                        'parents': [parent1.smiles, parent2.smiles],
                        'operator': 'crossover_two_point',
                        'child_index': i + 1,
                        'cut_points_p1': (cut1_a, cut1_b),
                        'cut_points_p2': (cut2_a, cut2_b),
                        'is_seed': False
                    }
                ))

        return children

    # Scaffold mutation probability factor (lower = less likely to mutate scaffold)
    # Value of 0.2 means scaffold (idx=0) is 5x less likely to be mutated than other positions
    SCAFFOLD_MUTATION_WEIGHT = 0.2
    def _get_mutation_position_weights(self, n_frags: int) -> np.ndarray:
        """
        Get position weights for scaffold-aware mutation.

        The first fragment (scaffold) has lower probability of being mutated
        because it defines the core structure and all subsequent reactions
        build upon it.

        Args:
            n_frags: Number of fragments in the route

        Returns:
            Normalized probability weights for each position
        """
        # First position (scaffold) gets reduced weight, others get weight 1.0
        weights = np.array([self.SCAFFOLD_MUTATION_WEIGHT] + [1.0] * (n_frags - 1))
        # Normalize to sum to 1
        return weights / weights.sum()

    def _mutate_replace(self, parent: Candidate) -> Optional[Candidate]:
        """
        Replace a fragment with a random building block.

        Uses scaffold-aware position selection: the first fragment (scaffold)
        has lower probability of being replaced since it defines the core
        structure and all subsequent reactions build upon it.

        Args:
            parent: Parent candidate to mutate

        Returns:
            Mutated child candidate or None if mutation fails
        """
        frags = list(parent.genotype)
        if not frags:
            return None

        # Use scaffold-aware position weights
        weights = self._get_mutation_position_weights(len(frags))
        idx = self.rng.choice(len(frags), p=weights)

        frags[idx] = self.rng.choice(self.building_blocks)

        return Candidate(
            genotype=frags,
            genotype_type='fragment_route',
            metadata={
                'generation': parent.metadata.get('generation', 0) + 1,
                'parents': [parent.smiles],
                'operator': 'mutate_replace',
                'mutated_position': idx,
                'is_scaffold_mutation': idx == 0,
                'is_seed': False
            }
        )

    def _mutate_add(self, parent: Candidate) -> Optional[Candidate]:
        """
        Add a random building block at a random position.

        Insertion positions are weighted to discourage inserting at position 0
        (before the scaffold), as this would make the new fragment the scaffold
        and demote the original scaffold to a building block.

        Args:
            parent: Parent candidate to mutate

        Returns:
            Mutated child candidate or None if at max length
        """
        frags = list(parent.genotype)
        if len(frags) >= self.config.solution.max_fragments:
            return None

        # Weight positions: position 0 (insert before scaffold) is discouraged
        # Positions 1 to len(frags) are equally likely
        n_positions = len(frags) + 1  # Can insert at 0, 1, ..., len(frags)
        weights = np.array([self.SCAFFOLD_MUTATION_WEIGHT] + [1.0] * len(frags))
        weights = weights / weights.sum()

        idx = self.rng.choice(n_positions, p=weights)
        frags.insert(idx, self.rng.choice(self.building_blocks))

        return Candidate(
            genotype=frags,
            genotype_type='fragment_route',
            metadata={
                'generation': parent.metadata.get('generation', 0) + 1,
                'parents': [parent.smiles],
                'operator': 'mutate_add',
                'insertion_position': idx,
                'is_seed': False
            }
        )

    def _mutate_remove(self, parent: Candidate) -> Optional[Candidate]:
        """
        Remove a fragment from a random position.

        Uses scaffold-aware position selection: the first fragment (scaffold)
        has lower probability of being removed since removing it would promote
        the second fragment to scaffold, fundamentally changing the molecule's
        core structure.

        Args:
            parent: Parent candidate to mutate

        Returns:
            Mutated child candidate or None if at min length
        """
        frags = list(parent.genotype)
        if len(frags) <= self.config.solution.min_fragments:
            return None

        # Use scaffold-aware position weights
        weights = self._get_mutation_position_weights(len(frags))
        idx = self.rng.choice(len(frags), p=weights)

        removed_frag = frags.pop(idx)

        return Candidate(
            genotype=frags,
            genotype_type='fragment_route',
            metadata={
                'generation': parent.metadata.get('generation', 0) + 1,
                'parents': [parent.smiles],
                'operator': 'mutate_remove',
                'removed_position': idx,
                'removed_fragment': removed_frag,
                'is_scaffold_removal': idx == 0,
                'is_seed': False
            }
        )

    def local_optimize(self, candidate: Candidate, max_steps: int = 10) -> Tuple[Candidate, int]:
        """A non-batch wrapper for local_optimize_batch for compatibility."""
        optimized_candidates, n_evals = self.local_optimize_batch([candidate], max_steps=max_steps)
        return optimized_candidates[0], n_evals

    def local_optimize_batch(
        self,
        candidates: List[Candidate],
        max_steps: int = 3,
        neighbors_per_candidate: int = 5,
    ) -> Tuple[List[Candidate], int]:
        """
        Batch local optimization (greedy hill climbing) for multiple candidates.

        CSA Local Optimization Algorithm:
        ----------------------------------
        For each candidate, perform greedy hill climbing:
          Step 0: Generate N neighbors (mutations of current best)
                  Build & evaluate all neighbors
                  If best neighbor improves → replace current with it
          Step 1: Generate N neighbors from NEW current best
                  Build & evaluate all neighbors
                  If best neighbor improves → replace current with it
          ...repeat until max_steps or convergence...

        Key: Step N+1 neighbors come from Step N's result (sequential dependency).

        GPU Batching Strategy:
        ----------------------
        All candidates are processed together within each step:
        - Generate neighbors for ALL active candidates at once
        - Build ALL neighbors in ONE build_batch() call
        - Evaluate ALL valid neighbors in ONE evaluate_batch() call
        This maximizes GPU batch size within each step.

        Note: Input candidates are assumed to be already built and evaluated
        (this is guaranteed by CSAEngine which calls build_batch/evaluate_batch
        before local_optimize_batch).

        Args:
            candidates: List of candidates (already built and evaluated)
            max_steps: Maximum hill climbing steps (default 3)
            neighbors_per_candidate: Neighbors per candidate per step (default 5)

        Returns:
            Tuple of (optimized candidates, total evaluations)
        """
        if not candidates:
            return [], 0

        best_candidates = list(candidates)
        total_evals = 0
        n_candidates = len(candidates)

        # Track convergence: stop a candidate after N steps without improvement
        no_improvement_count = [0] * n_candidates
        early_stop_threshold = 3

        for step in range(max_steps):
            # Find active candidates (not yet converged)
            active_indices = [i for i in range(n_candidates)
                            if no_improvement_count[i] < early_stop_threshold]

            if not active_indices:
                logger.debug(f"Local opt step {step}: all {n_candidates} candidates converged")
                break

            # =================================================================
            # STEP 1: Generate neighbors for all active candidates
            # =================================================================
            all_neighbors = []      # Neighbor candidates
            parent_indices = []     # Which candidate each neighbor came from

            for i in active_indices:
                parent = best_candidates[i]  # Current best for this candidate
                for _ in range(neighbors_per_candidate):
                    op = self.rng.choice(self.mutation_ops)
                    neighbor = op(parent)
                    if neighbor:
                        all_neighbors.append(neighbor)
                        parent_indices.append(i)

            if not all_neighbors:
                logger.debug(f"Local opt step {step}: no neighbors generated")
                break

            # =================================================================
            # STEP 2: Build all neighbors in one batch
            # =================================================================
            logger.debug(
                f"Local opt step {step}: {len(all_neighbors)} neighbors "
                f"for {len(active_indices)} active candidates"
            )
            built_neighbors = self.build_batch(all_neighbors)

            # =================================================================
            # STEP 3: Filter valid and evaluate
            # =================================================================
            valid_neighbors = []
            valid_parent_indices = []

            for i, neighbor in enumerate(built_neighbors):
                if neighbor.is_valid:
                    valid_neighbors.append(neighbor)
                    valid_parent_indices.append(parent_indices[i])

            if not valid_neighbors:
                # No valid neighbors - increment no-improvement for all active
                for i in active_indices:
                    no_improvement_count[i] += 1
                continue

            evaluated_neighbors = self.evaluate_batch(valid_neighbors)
            total_evals += len(evaluated_neighbors)

            # =================================================================
            # STEP 4: Greedy selection - keep best neighbor if it improves
            # =================================================================
            # Group by parent, find best neighbor for each
            best_per_parent: Dict[int, Candidate] = {}

            for neighbor, parent_idx in zip(evaluated_neighbors, valid_parent_indices):
                if parent_idx not in best_per_parent:
                    best_per_parent[parent_idx] = neighbor
                elif neighbor.objective_value < best_per_parent[parent_idx].objective_value:
                    best_per_parent[parent_idx] = neighbor

            # Update if improved
            improved_count = 0
            for parent_idx, best_neighbor in best_per_parent.items():
                current = best_candidates[parent_idx]
                if best_neighbor.objective_value < current.objective_value:
                    best_candidates[parent_idx] = best_neighbor
                    no_improvement_count[parent_idx] = 0
                    improved_count += 1
                else:
                    no_improvement_count[parent_idx] += 1

            # Update no-improvement for candidates with no valid neighbors
            for i in active_indices:
                if i not in best_per_parent:
                    no_improvement_count[i] += 1

            converged = sum(1 for c in no_improvement_count if c >= early_stop_threshold)
            logger.debug(
                f"Local opt step {step}: {improved_count}/{len(active_indices)} improved, "
                f"{converged}/{n_candidates} converged"
            )

        return best_candidates, total_evals
