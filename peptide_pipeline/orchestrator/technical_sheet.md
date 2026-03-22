# Orchestrator Technical Sheet

## Purpose

The Orchestrator module coordinates generation, chemical evaluation, biological scoring, and iterative ranking.

Primary responsibilities:

- run iterative optimization rounds,
- choose exploration vs exploitation parent strategy,
- merge chemist and biologist scores,
- maintain a global best pool and return final Top-K.

Base contract: `BaseOrchestrator` in `peptide_pipeline/orchestrator/base.py`.

---

## Base Contract

`BaseOrchestrator` stores references to three agents:

- `generator: BaseGenerator`
- `chemist: BaseChemist`
- `biologist: BaseBiologist`

It defines a single required orchestration method:

### `run(...) -> List[Dict[str, Any]]`

Current abstract signature:

- `nb_iterations: int`
- `nb_peptides: int`
- `top_k: int`
- `exploration_rate: float = 0.1`
- `initial_peptide: Optional[str] = None`
- `final_target: Optional[Dict[str, Any]] = None`
- `random_parent_count: int = 4`

---

## Inherited Capabilities

By subclassing `BaseOrchestrator`, you inherit:

- class-level logger named `"peptide_pipeline.orchestrator"`,
- stored typed agent references (`generator`, `chemist`, `biologist`).

---

## Current Implementations In This Repository

### `Orchestrator` (`peptide_pipeline/orchestrator/orchestrator.py`)

Core behavior:

1. Builds base generation constraints:
   - uses explicit `final_target` when provided,
   - otherwise infers target values from chemist config (`length`, `molecular_weight`, `net_charge`, `isoelectric_point`, `hydrophobicity`, `cathionicity`, `logp`) and maps to generator keys (`size`, `net_charge_pH5_5`, etc.).
2. Iterates for `nb_iterations` rounds.
3. Generation mode:
   - iteration 1: target-only generation,
   - later iterations: exploitation from top global pool or exploration random parents.
4. Runs chemist evaluation via `chemist.evaluate_peptides(candidates)`.
5. If no in-limit candidates, falls back to all chemist results for biological scoring.
6. Scores remaining sequences using `biologist.score_peptides(...)`.
7. Computes combined score:
   - `combined_score = (chemist_score + biologist_score) / 2.0`
8. Updates global best per unique sequence and sorts final ranking.
9. Returns final `top_k` rows.

Returned row schema:

- `peptide`
- `score` (same value as `combined_score`)
- `combined_score`
- `chemist_score`
- `biologist_score`
- `in_limits`
- `properties`
- `iteration`

### Important Behavior Notes (As Implemented)

- `initial_peptide` is accepted but currently not used as a generation seed (only logged).
- The orchestrator uses `evaluate_peptides` directly, not `get_top_filtered_peptides`.
- `peptide_pipeline/orchestrator/__init__.py` currently exports only `BaseOrchestrator` and not `Orchestrator`.
- Constraint inference includes `cathionicity` mapping, but current `ChemistConfig` does not define `cathionicity`.

---

## How To Add A New Orchestrator

1. Create a class inheriting from `BaseOrchestrator`.
2. Keep dependencies typed as `BaseGenerator`, `BaseChemist`, `BaseBiologist`.
3. Implement `run(...)` with explicit iteration loop and ranking strategy.
4. Define a stable output schema for each ranked row.
5. Handle empty/failed intermediate stages (no candidates, no scores) safely.
6. Add logs for traceability at each stage.
7. Add tests for deterministic ranking and edge cases.

---

## Minimal Skeleton

```python
from typing import Any, Dict, List, Optional
from peptide_pipeline.orchestrator.base import BaseOrchestrator


class MyOrchestrator(BaseOrchestrator):
    def run(
        self,
        nb_iterations: int,
        nb_peptides: int,
        top_k: int,
        exploration_rate: float = 0.1,
        initial_peptide: Optional[str] = None,
        final_target: Optional[Dict[str, Any]] = None,
        random_parent_count: int = 4,
    ) -> List[Dict[str, Any]]:
        global_rows: List[Dict[str, Any]] = []

        for iteration in range(1, nb_iterations + 1):
            peptides = self.generator.generate_peptides(nb_peptides, constraints=final_target)
            chem = self.chemist.evaluate_peptides(peptides)
            seqs = [row["sequence"] for row in chem if row.get("sequence")]
            bio = self.biologist.score_peptides(seqs)

            for c, b in zip(chem, bio):
                c_score = float(c.get("score", 0.0))
                global_rows.append(
                    {
                        "peptide": c["sequence"],
                        "combined_score": (c_score + float(b)) / 2.0,
                        "iteration": iteration,
                    }
                )

        ranked = sorted(global_rows, key=lambda x: x["combined_score"], reverse=True)
        return ranked[:top_k]
```

---

## Developer Checklist

- Iteration loop is explicit and traceable.
- Exploration/exploitation policy is documented.
- Combined-score formula is documented and test-covered.
- Output schema is stable across runs.
- Empty-stage fallbacks are implemented and tested.
