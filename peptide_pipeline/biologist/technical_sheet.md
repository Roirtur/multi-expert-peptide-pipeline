# Biologist Technical Sheet

## Purpose

The Biologist module scores peptide candidates for biological relevance/activity proxies.

Primary responsibilities:

- generate one biological score per peptide,
- support optional contextual scoring,
- provide stable outputs for orchestrator ranking.

Base contract: `BaseBiologist` in `peptide_pipeline/biologist/base.py`.

---

## Base Contract

Every implementation must inherit `BaseBiologist` and implement:

### `score_peptides(self, peptides: List[str]) -> List[float]`

- Returns one score per peptide.

### `predict_activity(self, peptides: List[str], context: Optional[Any] = None) -> List[float]`

- Returns one activity estimate per peptide.
- Accepts optional context.

---

## Inherited Capabilities

By subclassing `BaseBiologist`, you inherit:

- class-level logger named `"peptide_pipeline.biologist"`.

---

## Current Implementations In This Repository

### `ESMBiologistCos` (`peptide_pipeline/biologist/esm_biologist_cos.py`)

- Loads an ESM-2 model/tokenizer from Hugging Face (`transformers`).
- Computes mean-pooled sequence embeddings.
- Scores with cosine similarity to a stored reference peptide embedding.
- Maps cosine score from `[-1, 1]` to `[0, 1]`.
- `predict_activity(..., context=...)` temporarily replaces reference embedding when context is a non-empty string.

### `ESMBiologistGlobalL2` (`peptide_pipeline/biologist/esm_biologist_global_l2.py`)

- Loads ESM-2 and uses hidden layer index 6 embeddings (with `output_hidden_states=True`).
- Computes mean-pooled embeddings and L2 distance to reference embedding.
- Scores with `exp(-distance / score_temperature)`.
- `predict_activity` context behavior matches the cosine variant (temporary reference replacement).

### Important Behavior Notes (As Implemented)

- Both implementations return `[]` for empty peptide input.
- Neither implementation validates sequence characters before model inference.
- Base interface accepts `Optional[Any]` context, but both concrete implementations effectively handle string context.
- Runtime requires model download/access (or local cache) for the configured Hugging Face model.

---

## How To Add A New Biologist

1. Create a class inheriting from `BaseBiologist`.
2. Initialize model/dependencies in `__init__`.
3. Implement deterministic `score_peptides` mapping input list to score list.
4. Implement `predict_activity` and document context schema.
5. Validate inputs and return explicit errors for unsupported context types.
6. Add tests for empty input, normal input, and context-driven behavior.

---

## Minimal Skeleton

```python
from typing import Any, List, Optional
from peptide_pipeline.biologist.base import BaseBiologist


class MyBiologist(BaseBiologist):
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
        return self.score_peptides(peptides)
```

---

## Developer Checklist

- One output score per input peptide.
- Score range/meaning is clearly documented.
- Context schema and fallback behavior are explicit.
- External model dependencies and offline behavior are documented.
- Tests cover deterministic behavior and edge cases.
