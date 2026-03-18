# Biologist Technical Sheet

## Purpose

In this project, a Biologist is the component that evaluates peptide candidates from a biological perspective.

Its role is to:

- score peptide quality in a normalized way,
- estimate biological activity for peptide batches,
- optionally use contextual biological information during prediction,
- expose a stable interface to the rest of the pipeline.

The contract is defined by `BaseBiologist` in `peptide_pipeline/biologist/base.py`.

---

## Base Interface (What Every New Biologist Must Implement)

Any new biologist must inherit from `BaseBiologist` and implement exactly these abstract methods:

### 1) `score_peptides(self, peptides: List[str]) -> List[float]`

This method is responsible for assigning a normalized score to each peptide.

Expected behavior:

- accept a batch of peptide sequences as `List[str]`,
- return one scalar score per input peptide,
- return a `List[float]` aligned with input order,
- keep scores in the `[0, 1]` range as defined by the base contract.

Implementation notes:

- validate peptide inputs early (empty strings, invalid characters, malformed entries),
- document your scoring logic and assumptions,
- keep scoring deterministic when possible for reproducibility,
- fail with clear and actionable errors when input is invalid.

### 2) `predict_activity(self, peptides: List[str], context: Optional[Any] = None) -> List[float]`

This method is responsible for predicting functional activity for each peptide.

Expected behavior:

- accept a peptide batch as `List[str]`,
- optionally consume `context` to enrich predictions (for example target metadata or reference information),
- return a `List[float]` with one prediction per peptide,
- preserve a predictable mapping between inputs and outputs (same order and same length).

Implementation notes:

- define and document what `context` format you support,
- handle missing or unsupported context explicitly,
- avoid hidden side effects during prediction,
- make error messages explicit about what failed and how to fix it.

---

## What You Inherit From BaseBiologist

When you subclass `BaseBiologist`, you can directly use:

- `logger`: class logger configured under `"peptide_pipeline.biologist"` for consistent logging.

---

## How To Add a New Biologist

Use this workflow when creating a new biologist implementation.

1. Create a new class that inherits from `BaseBiologist`.
2. Add an initializer (`__init__`) to define model/state dependencies.
3. Implement `score_peptides(peptides)`.
4. Implement `predict_activity(peptides, context=None)`.
5. Validate inputs and context with clear error handling.
6. Keep output structure stable and documented.
7. Add tests for success cases and failure cases for both methods.

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

		# Replace with real scoring logic.
		return [0.5 for _ in peptides]

	def predict_activity(
		self,
		peptides: List[str],
		context: Optional[Any] = None,
	) -> List[float]:
		if not peptides:
			return []

		# Replace with real activity prediction logic.
		return [0.5 for _ in peptides]
```

---

## Design Guidelines For New Developers

- Keep one biologist focused on one biological evaluation strategy.
- Define clear contracts for inputs, outputs, and accepted context.
- Keep scoring and activity prediction behavior consistent across runs.
- Ensure output list length always matches input list length.
- Use logging to expose key decisions and troubleshooting details.

---

## Quick Validation Checklist

Before opening a PR for a new biologist:

- `score_peptides` validates peptide inputs correctly.
- `score_peptides` returns one `[0, 1]` score per peptide.
- `predict_activity` supports documented context behavior.
- `predict_activity` returns one value per peptide in the same order.
- Both methods fail clearly on invalid inputs.
- Tests cover success and failure paths for both methods.
