# Biologist Technical Sheet

## Purpose

The Biologist module evaluates peptide candidates from a biological relevance perspective.

Primary responsibilities:

- score candidate peptides in a normalized way,
- estimate biological activity for a peptide batch,
- optionally incorporate contextual biological information,
- provide stable outputs for orchestrator ranking and filtering.

Base contract: `BaseBiologist` in `peptide_pipeline/biologist/base.py`.

---

## Base Contract

Every biologist implementation must inherit from `BaseBiologist` and implement the following methods.

### `score_peptides(self, peptides: List[str]) -> List[float]`

- Returns one scalar score per peptide.
- Preserves input order in output alignment.
- Base contract specifies normalized scores in `[0, 1]`.

### `predict_activity(self, peptides: List[str], context: Optional[Any] = None) -> List[float]`

- Returns one activity estimate per peptide.
- Accepts optional `context` to adjust prediction behavior.
- Preserves input-to-output mapping.

---

## Inherited Capabilities

By subclassing `BaseBiologist`, you inherit:

- `logger` configured under `"peptide_pipeline.biologist"`.

---

## Current Implementations In This Repository

### `ESMBiologistCos` (`peptide_pipeline/biologist/esm_biologist_cos.py`)

- Uses ESM-2 embeddings with cosine similarity against a reference peptide.
- Converts cosine similarity from `[-1, 1]` to `[0, 1]`.
- Supports optional context by temporarily swapping the reference embedding.

### `ESMBiologistZscore` (`peptide_pipeline/biologist/esm_biologist_zscore.py`)

- Uses ESM-2 embeddings and L2 distance to a reference embedding.
- Applies z-score normalization across batch distances, then sigmoid scaling.
- Supports optional context with temporary reference replacement.

### Important Behavior Notes

- Both implementations return empty lists for empty peptide inputs.
- Both implementations depend on `transformers` model loading and may require access to Hugging Face model files.

---

## How To Add A New Biologist

1. Create a class inheriting from `BaseBiologist`.
2. Define model dependencies and runtime configuration in `__init__`.
3. Implement `score_peptides` with deterministic, documented behavior.
4. Implement `predict_activity` and define accepted context schema.
5. Validate peptide and context inputs.
6. Add tests for normal behavior, invalid inputs, and edge cases.

---

## Minimal Skeleton

```python
from typing import Any, List, Optional
from peptide_pipeline.biologist.base import BaseBiologist


class MyBiologist(BaseBiologist):
    def __init__(self):
        self._is_ready = True

    def score_peptides(self, peptides: List[str]) -> List[float]:
        if not peptides:
            return []
        return [0.5 for _ in peptides]

    def predict_activity(
        self,
        peptides: List[str],
        context: Optional[Any] = None,
    ) -> List[float]:
        if not peptides:
            return []
        return [0.5 for _ in peptides]
```

---

## Developer Checklist

- `score_peptides` returns one score per peptide in stable order.
- Score range and scaling are explicitly documented.
- `predict_activity` context format is documented and validated.
- Error handling is explicit and actionable.
- Tests cover expected behavior and failure modes.
