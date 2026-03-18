# Generator Technical Sheet

## Purpose

In this project, a Generator is the component that proposes peptide candidates and improves them over time.

Its role is to:

- create new peptide sequences,
- update existing sequences from feedback,
- learn from data so proposals become better,
- expose a stable interface to the rest of the pipeline.

The contract is defined by `BaseGenerator` in `peptide_pipeline/generator/base.py`.

---

## Base Interface (What Every New Generator Must Implement)

Any new generator must inherit from `BaseGenerator` and implement exactly these abstract methods:

### 1) `generate_peptides(self, count: int, constraints: Optional[Dict[str, Any]] = None) -> List[str]`

This method is responsible for creating new peptide sequences.

Expected behavior:

- return exactly `count` peptide sequences,
- return a `List[str]` where each item is one peptide sequence,
- optionally use `constraints` to guide generation (length, motif rules, residue bans, property targets, etc.),
- ensure outputs are valid and usable by downstream components.

Implementation notes:

- validate `count` and constraints early,
- fail with clear errors for impossible or invalid constraints,
- keep generation behavior reproducible when randomness is involved,
- document which keys are accepted in `constraints`.

### 2) `modify_peptides(self, peptides: List[str], feedback: Optional[Any] = None) -> List[str]`

This method is responsible for refining or evolving an existing batch of peptides.

Expected behavior:

- accept a list of input peptides to modify,
- optionally use `feedback` to guide modifications,
- return a `List[str]` of updated peptide sequences,
- preserve a predictable relation between input and output (for example, same order or same output size), and document it.

Implementation notes:

- validate peptide inputs before applying modifications,
- define how feedback is interpreted and document expected format,
- avoid hidden side effects outside method scope,
- ensure modified outputs remain valid peptide strings.

### 3) `train_model(self, data: Any, **kwargs) -> None`

This method is responsible for training or updating the generator model.

Expected behavior:

- consume training data from `data`,
- support optional training options through `**kwargs` (epochs, learning rate, batch size, output paths, etc.),
- update internal model state so generation quality can improve,
- not return data (returns `None`).

Implementation notes:

- validate training inputs and required options,
- document accepted `**kwargs` keys and defaults,
- keep training workflow explicit (setup, optimization, checkpointing),
- raise actionable errors when configuration is incomplete.

---

## What You Inherit From BaseGenerator

When you subclass `BaseGenerator`, you can directly use:

- `self.device`: automatically set to `"cuda"` if available, otherwise `"cpu"`,
- `logger`: class logger configured under `"peptide_pipeline.generator"`,
- `nn.Module` behavior: because `BaseGenerator` inherits from `torch.nn.Module`, your generator can use standard PyTorch module patterns.

---

## How To Add a New Generator

Use this workflow when creating a new generator implementation.

1. Create a new class that inherits from `BaseGenerator`.
2. Call `super().__init__()` in your initializer to setup inherited state (including `self.device`).
3. Add your model components and internal state in `__init__`.
4. Implement `generate_peptides(count, constraints=None)`.
5. Implement `modify_peptides(peptides, feedback=None)`.
6. Implement `train_model(data, **kwargs)`.
7. Validate inputs and emit clear, actionable errors.
8. Add tests for normal behavior and failure cases for all three methods.

---

## Minimal Skeleton

```python
from typing import Any, Dict, List, Optional
import torch.nn as nn
from peptide_pipeline.generator.base import BaseGenerator


class MyGenerator(BaseGenerator):
	def __init__(self):
		super().__init__()
		self.model = nn.Identity()

	def generate_peptides(
		self,
		count: int,
		constraints: Optional[Dict[str, Any]] = None,
	) -> List[str]:
		if count <= 0:
			raise ValueError("count must be > 0")

		# Replace with real generation logic.
		peptides = ["ACDE" for _ in range(count)]
		return peptides

	def modify_peptides(
		self,
		peptides: List[str],
		feedback: Optional[Any] = None,
	) -> List[str]:
		if not peptides:
			return []

		# Replace with real refinement logic.
		return peptides

	def train_model(self, data: Any, **kwargs) -> None:
		if data is None:
			raise ValueError("data must not be None")

		# Replace with real training loop.
		return None
```

---

## Design Guidelines For New Developers

- Keep one generator focused on one generation strategy.
- Define and document your expected constraint and feedback schemas.
- Keep method contracts stable so orchestrator and evaluators can rely on them.
- Make failures explicit and easy to debug.
- Ensure generated and modified sequences are always valid for downstream use.

---

## Quick Validation Checklist

Before opening a PR for a new generator:

- `generate_peptides` returns the requested number of valid strings.
- `generate_peptides` handles/validates constraints as documented.
- `modify_peptides` handles empty and invalid inputs clearly.
- `modify_peptides` applies feedback consistently and predictably.
- `train_model` validates `data` and required training options.
- All three methods are covered by tests (success + failure cases).
