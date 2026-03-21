# Chemist Technical Sheet

## Purpose

The Chemist module evaluates peptide candidates against configurable physicochemical constraints and produces ranked outputs.

Primary responsibilities:

- validate amino-acid sequence format,
- compute chemical properties,
- score candidates against configured ranges/targets,
- expose ranked candidate lists for downstream selection.

Base contract: `BaseChemist` in `peptide_pipeline/chemist/base.py`.

---

## Base Contract

Every implementation must inherit `BaseChemist` and implement:

### `get_top_filtered_peptides(self, peptides: List[str], topK: int) -> List[str]`

- Returns top-ranked peptides after evaluation/filtering.

### `evaluate_peptides(self, peptides: List[str]) -> List[Dict[str, Any]]`

- Returns per-peptide evaluation records with scores/properties.

---

## Inherited Capabilities

By subclassing `BaseChemist`, you inherit:

- `basic_aa`: set of the 20 standard amino acids,
- class-level logger named `"peptide_pipeline.chemist"`,
- `validate_sequence(sequence)` helper,
- `self.config` set from constructor argument.

---

## Current Implementations In This Repository

### `ChemistAgent` (`peptide_pipeline/chemist/agent_v1/chemist_agent.py`)

- Uses `ChemistConfig` + `RangeTarget` from `config_chemist.py`.
- Computes configured properties from `PROPERTY_REGISTRY`:
  - `length`
  - `molecular_weight`
  - `logp`
  - `net_charge` (pH-dependent)
  - `isoelectric_point`
  - `hydrophobicity`
- For each peptide, computes:
  - `properties`
  - per-property score map `property_scores`
  - aggregate score `score`
  - boolean `in_limits`
- Ranking in `get_top_filtered_peptides` is deterministic:
  - in-limit first,
  - then descending score.

### Configuration Schema (As Implemented)

`ChemistConfig` supports:

- global `ph: float = 7.0`
- optional constraints: `length`, `molecular_weight`, `logp`, `net_charge`, `isoelectric_point`, `hydrophobicity`

Each constraint is a `RangeTarget` with:

- `min: float`
- `max: float`
- `target: float`
- optional `weight: float = 1.0`

### Important Behavior Notes (As Implemented)

- `BaseChemist.get_top_filtered_peptides` is typed as `List[str]`, but `ChemistAgent.get_top_filtered_peptides` currently returns a list of dictionaries (truncated evaluated rows).
- `evaluate_peptides` currently mutates the input list while iterating when removing invalid peptides.
- Internal `distance_from_target` is computed in analysis helpers but is not exposed in final `evaluate_peptides` output.
- Overall score uses weighted sum divided by number of scored properties (not by sum of weights).

---

## How To Add A New Chemist Agent

1. Create a class inheriting from `BaseChemist`.
2. Define/validate a clear configuration schema.
3. Implement `evaluate_peptides` with stable output keys.
4. Implement deterministic `get_top_filtered_peptides` ranking.
5. Handle invalid sequences without mutating caller-owned inputs in-place.
6. Add tests for boundary cases and ranking consistency.

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
        rows: List[Dict[str, Any]] = []
        for sequence in peptides:
            if not self.validate_sequence(sequence):
                continue
            rows.append(
                {
                    "sequence": sequence,
                    "properties": {},
                    "score": 0.0,
                    "in_limits": True,
                }
            )
        return rows

    def get_top_filtered_peptides(self, peptides: List[str], topK: int):
        ranked = self.evaluate_peptides(peptides)
        return ranked[:topK]
```

---

## Developer Checklist

- Output schema keys are documented and stable.
- Score formula and normalization are explicit.
- Ranking behavior is deterministic and test-covered.
- Invalid sequences are handled safely.
- Contract mismatch (`List[str]` vs list of dicts) is resolved or documented.
