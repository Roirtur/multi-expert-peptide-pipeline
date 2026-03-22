# Generator Technical Sheet

## Purpose

The Generator module proposes peptide candidates and provides trainable generative models used by the orchestrator.

Primary responsibilities:

- generate peptide sequences,
- optionally condition generation on target properties,
- retrain/update generative model weights,
- support iterative refinement entry points.

Base contract: `BaseGenerator` in `peptide_pipeline/generator/base.py`.

---

## Base Contract

Every implementation must inherit `BaseGenerator` (which extends `torch.nn.Module`) and implement:

### `generate_peptides(self, count: int, constraints: Optional[Dict[str, Any]] = None) -> List[str]`

- Generates peptide strings.
- Accepts optional constraints.

### `modify_peptides(self, peptides: List[str], feedback: Optional[Any] = None) -> List[str]`

- Refines/replaces an input peptide list.
- May use optional feedback.

### `train_model(self, data: Any, **kwargs) -> None`

- Trains model parameters in place.

---

## Inherited Capabilities

By subclassing `BaseGenerator`, you inherit:

- class-level logger named `"peptide_pipeline.generator"`,
- `nn.Module` behavior (parameters, `.to(...)`, `.train()`, `.eval()`),
- default `self.device` string (`"cuda"` or `"cpu"`) initialized in base class.

---

## Current Implementations In This Repository

### `VAEGenerator` (`peptide_pipeline/generator/vae_generator.py`)

- Unconditional VAE-like model.
- Uses one-hot flattened sequence representation with 20 amino-acid channels.
- `generate_peptides` samples latent vectors and decodes with temperature sampling.
- `constraints` argument exists but is currently ignored.
- `modify_peptides` regenerates a fresh batch (same size by default, or overridden by `feedback['count']`).
- `train_model` uses cross-entropy reconstruction + KL term + cosine LR schedule.

### `CVAEGenerator` (`peptide_pipeline/generator/cvae_generator.py`)

- Conditional VAE with condition tensor (`condition_dim`, default 32).
- Uses 21-token vocabulary internally (20 amino acids + PAD), with `max_len` fixed at model init.
- `generate_peptides` enforces fixed generated length derived from scalar constraint `size`.
- Constraints are scalar-only per property. Min/max dictionaries are rejected.
- `modify_peptides` currently ignores feedback and regenerates same-size batch.
- `train_model` expects three tensors: `data`, `conditions`, `lengths`.

### Important Behavior Notes (As Implemented)

- The two concrete `train_model` signatures are different (`VAEGenerator` vs `CVAEGenerator`) even though both satisfy the abstract method name.
- `CVAEGenerator.__init__` overwrites `input_dim` using `max_len * vocab_size`.
- Package export file `peptide_pipeline/generator/__init__.py` currently exposes only `BaseGenerator`.

---

## How To Add A New Generator

1. Create a class inheriting from `BaseGenerator`.
2. Call `super().__init__()` in `__init__`.
3. Define architecture, tokenization/encoding strategy, and device behavior.
4. Implement `generate_peptides`, `modify_peptides`, and `train_model`.
5. Clearly document accepted `constraints` and `feedback` schemas.
6. Validate inputs and raise explicit exceptions for invalid usage.
7. Add save/load helpers if persistent weights are needed.
8. Add tests for generation count, output validity, and training-step behavior.

---

## Minimal Skeleton

```python
from typing import Any, Dict, List, Optional
import torch.nn as nn
from peptide_pipeline.generator.base import BaseGenerator


class MyGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()

    def generate_peptides(
        self,
        count: int,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if count <= 0:
            raise ValueError("count must be > 0")
        return ["ACDE" for _ in range(count)]

    def modify_peptides(
        self,
        peptides: List[str],
        feedback: Optional[Any] = None,
    ) -> List[str]:
        return self.generate_peptides(len(peptides), constraints=None)

    def train_model(self, data: Any, **kwargs) -> None:
        if data is None:
            raise ValueError("data must not be None")
        return None
```

---

## Developer Checklist

- `generate_peptides` returns exactly the requested count.
- Output sequences use valid amino-acid alphabet and expected lengths.
- `constraints`/`feedback` schema is explicit and validated.
- `train_model` input contract is documented and test-covered.
- Save/load behavior is reproducible across devices.
