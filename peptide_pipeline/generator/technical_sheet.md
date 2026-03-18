# Generator Technical Sheet

## Purpose

The Generator module is responsible for proposing peptide candidates and refining them across optimization cycles.

Primary responsibilities:

- generate new peptide sequences,
- modify existing candidates from feedback,
- train or update an internal generative model,
- expose a stable API for orchestrator integration.

Base contract: `BaseGenerator` in `peptide_pipeline/generator/base.py`.

---

## Base Contract

Every generator implementation must inherit from `BaseGenerator` and implement the following methods.

### `generate_peptides(self, count: int, constraints: Optional[Dict[str, Any]] = None) -> List[str]`

- Produces exactly `count` peptide sequences.
- Accepts optional `constraints` to guide generation.
- Returns one peptide string per list item.

### `modify_peptides(self, peptides: List[str], feedback: Optional[Any] = None) -> List[str]`

- Refines an input peptide batch.
- May use `feedback` as an optional control signal.
- Returns a predictable output mapping (for example, same list length and order) documented by the implementation.

### `train_model(self, data: Any, **kwargs) -> None`

- Trains or updates the underlying generator model.
- Uses `data` plus optional training parameters in `**kwargs`.
- Updates model state in place and returns `None`.

---

## Inherited Capabilities

By subclassing `BaseGenerator`, you inherit:

- `self.device` initialized as `"cuda"` when available, otherwise `"cpu"`,
- `logger` configured under `"peptide_pipeline.generator"`,
- full `torch.nn.Module` behavior.

---

## Current Implementations In This Repository

### `VAEGenerator` (`peptide_pipeline/generator/vae_generator.py`)

- Unconditional VAE peptide generator.
- Generates sequences by sampling latent vectors and decoding token logits.
- `modify_peptides` currently regenerates a new batch of equal size.
- `train_model` uses reconstruction loss + KL regularization with cosine LR scheduling.

### `CVAEGenerator` (`peptide_pipeline/generator/cvae_generator.py`)

- Conditional VAE supporting property-based constraints.
- Encodes constraints into a conditioning tensor and generates peptides with sampled lengths.
- Supports property ranges and scalar constraints.
- `train_model` consumes encoded sequences, condition vectors, and sequence lengths.

---

## How To Add A New Generator

1. Create a class inheriting from `BaseGenerator`.
2. Call `super().__init__()` in `__init__`.
3. Define model layers and internal state.
4. Implement `generate_peptides`, `modify_peptides`, and `train_model`.
5. Validate method inputs and raise explicit errors for invalid usage.
6. Document accepted schemas for `constraints`, `feedback`, and `**kwargs`.
7. Add tests for success and failure paths.

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
        return ["ACDE" for _ in range(count)]

    def modify_peptides(
        self,
        peptides: List[str],
        feedback: Optional[Any] = None,
    ) -> List[str]:
        if not peptides:
            return []
        return peptides

    def train_model(self, data: Any, **kwargs) -> None:
        if data is None:
            raise ValueError("data must not be None")
        return None
```

---

## Developer Checklist

- `generate_peptides` returns exactly the requested count.
- All generated and modified sequences are valid peptide strings.
- `constraints` and `feedback` formats are documented.
- `train_model` required inputs are validated.
- Public behavior is deterministic where reproducibility is required.
- Tests cover normal and failure behavior for all abstract methods.
