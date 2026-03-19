# DataLoader Technical Sheet

## Purpose

The DataLoader module converts raw peptide datasets into validated in-memory objects that can be consumed by training and inference components.

Primary responsibilities:

- read data from a source path,
- parse and normalize tabular or JSON content,
- validate schema expectations,
- expose loaded data through a stable retrieval API.

Base contract: `BaseDataLoader` in `peptide_pipeline/dataloader/base.py`.

---

## Base Contract

Every dataloader implementation must inherit from `BaseDataLoader` and implement the following methods.

### `load_data(self, source: str, **kwargs) -> Any`

- Loads and preprocesses dataset content from `source`.
- Accepts loader-specific options through `**kwargs`.
- Stores processed data internally.
- Returns a loaded object (contract-level expectation).

### `get_data(self) -> Any`

- Returns the currently loaded data object.
- Must fail explicitly when called before loading.
- Must avoid hidden side effects.

---

## Inherited Capabilities

By subclassing `BaseDataLoader`, you inherit:

- `logger` configured under `"peptide_pipeline.dataloader"`.

---

## Current Implementations In This Repository

### `dataloader.py::DataLoader`

- CSV-oriented loader using `pandas.read_csv`.
- Validates presence of `NAME` and `SEQUENCE` columns.
- Supports selecting a subset of columns via `columns`.
- Stores loaded data in `self.data` and exposes it via `get_data`.

### `dataloader_json.py::DataLoader`

- JSON-oriented loader supporting standard JSON and JSON Lines.
- Supports optional `required_columns`, `rename_map`, and `auto_create_column`.
- Supports optional projection through `columns`.
- Stores loaded data in `self.data` and exposes it via `get_data`.

### Important Behavior Notes

- Both current implementations terminate the process with `sys.exit(1)` on error.
- Both current `load_data` implementations are annotated to return `None`, even though the abstract contract uses `-> Any`.
- For new implementations, prefer raising explicit exceptions instead of process termination.

---

## How To Add A New DataLoader

1. Create a class inheriting from `BaseDataLoader`.
2. Initialize internal storage (for example, `self.data = None`).
3. Implement `load_data(source, **kwargs)` for one source format or one strategy.
4. Implement `get_data()` with a clear pre-load failure mode.
5. Document accepted kwargs and output schema.
6. Add tests for valid input, invalid input, and pre-load access.

---

## Minimal Skeleton

```python
from typing import Any
from peptide_pipeline.dataloader.base import BaseDataLoader


class MyDataLoader(BaseDataLoader):
    def __init__(self):
        self.data = None

    def load_data(self, source: str, **kwargs) -> Any:
        # 1) Validate source/options
        # 2) Read raw content
        # 3) Normalize schema
        # 4) Store in self.data
        self.data = ...
        return self.data

    def get_data(self) -> Any:
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data(...) first.")
        return self.data
```

---

## Developer Checklist

- `load_data` validates path and expected schema.
- Errors are explicit and actionable.
- `get_data` fails clearly if loading did not run.
- Output shape and key columns are documented.
- Tests cover success and failure paths.

