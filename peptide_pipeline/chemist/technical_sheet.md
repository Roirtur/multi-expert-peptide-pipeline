# Chemist Technical Sheet

## Purpose

The Chemist module evaluates peptide candidates against chemical constraints and ranks them for downstream selection.

Primary responsibilities:

- validate amino-acid sequence compliance,
- compute configured physicochemical properties,
- score candidates from distance to target constraints,
- return ranked candidates with in-limit status.

Base contract: `BaseChemist` in `peptide_pipeline/chemist/base.py`.

---

## Base Contract

Every chemist implementation must inherit from `BaseChemist` and implement the following methods.

### `get_top_filtered_peptides(self, peptides: List[str], topK: int) -> List[str]`

- Returns top candidates after filtering and ranking.
- Accepts a peptide list plus a requested `topK` count.

### `evaluate_peptides(self, peptides: List[str]) -> List[Dict[str, Any]]`

- Computes chemical properties and scores for each peptide.
- Returns a structured per-peptide evaluation list.

---

## Inherited Capabilities

By subclassing `BaseChemist`, you inherit:

- `basic_aa` set with 20 standard amino acids,
- `logger` configured under `"peptide_pipeline.chemist"`,
- `validate_sequence(sequence: str) -> bool` utility,
- `self.config` storage initialized from a Pydantic model.

---

## Current Implementations In This Repository

### `ChemistAgent` (`peptide_pipeline/chemist/agent_v1/chemist_agent.py`)

- Uses `ChemistConfig` and `RangeTarget` from `config_chemist.py`.
- Computes configured properties using `PROPERTY_REGISTRY`.
- For each peptide, derives:
  - `properties`,
  - `distance_from_target`,
  - `in_limits` boolean,
  - aggregate normalized `score`.
- `get_top_filtered_peptides` sorts by in-limit first, then by score.

### Configurable Property Families

Current config supports optional constraints for:

- `length`,
- `molecular_weight`,
- `logp`,
- `net_charge`,
- `isoelectric_point`,
- `hydrophobicity`,
- `cathionicity`,
- plus global `ph`.

### Important Behavior Notes

- `BaseChemist.get_top_filtered_peptides` is typed as `-> List[str]`, while current `ChemistAgent` returns a list of dictionaries from `evaluate_peptides` truncated to `topK`.
- In `BaseChemist.validate_sequence`, a warning log is emitted before the boolean check, including valid sequences. This may produce noisy logs.

---

## How To Add A New Chemist Agent

1. Create a class inheriting from `BaseChemist`.
2. Define a configuration schema (Pydantic model) for constraints.
3. Implement `evaluate_peptides` with stable output keys and score semantics.
4. Implement `get_top_filtered_peptides` with deterministic ranking logic.
5. Keep invalid sequence handling explicit and non-crashing.
6. Add tests for valid, invalid, and out-of-range peptide scenarios.

---

## Minimal Skeleton

```python
from typing import Any, Dict, List
from pydantic import BaseModel
from peptide_pipeline.chemist.base import BaseChemist


class MyChemistConfig(BaseModel):
    pass


class MyChemist(BaseChemist):
    def __init__(self, config: MyChemistConfig):
        super().__init__(config)

    def evaluate_peptides(self, peptides: List[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for peptide in peptides:
            if not self.validate_sequence(peptide):
                continue
            results.append(
                {
                    "sequence": peptide,
                    "properties": {},
                    "score": 0.0,
                    "in_limits": True,
                }
            )
        return results

    def get_top_filtered_peptides(self, peptides: List[str], topK: int):
        ranked = self.evaluate_peptides(peptides)
        return ranked[:topK]
```

---

## Developer Checklist

- `evaluate_peptides` returns stable, documented keys.
- Score computation and normalization are documented.
- `get_top_filtered_peptides` ranking logic is deterministic.
- Invalid sequences are handled safely.
- Tests cover filtering, ranking, and boundary conditions.
