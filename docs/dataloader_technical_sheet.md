# DataLoader Technical Sheet

## Purpose

The DataLoader module reads raw peptide datasets and exposes them as in-memory tables for downstream training/evaluation.

Primary responsibilities:

- load dataset files from disk,
- validate or normalize schema,
- optionally filter/project rows and columns,
- return loaded data through a retrieval API.

Base contract: `BaseDataLoader` in `peptide_pipeline/dataloader/base.py`.

---

## Base Contract

Every implementation must inherit `BaseDataLoader` and implement:

### `load_data(self, source: str, **kwargs) -> Any`

- Loads and preprocesses data from `source`.
- Stores loaded data in internal state.
- Contract type is `Any`.

### `get_data(self) -> Any`

- Returns the currently loaded dataset.
- Must fail clearly if data was never loaded.

---

## Inherited Capabilities

By subclassing `BaseDataLoader`, you inherit:

- class-level logger named `"peptide_pipeline.dataloader"`.

---

## Current Implementations In This Repository

### `dataloader.py::DataLoader` (CSV loader)

- Reads CSV with `pandas.read_csv(source)`.
- Requires columns `SEQUENCE` and `NAME`.
- If `columns` is `None`, defaults to `['NAME', 'SEQUENCE']`.
- Validates requested `columns` and keeps only those columns.
- Stores result in `self.data` (`pandas.DataFrame`).

### `dataloader_json.py::DataLoader` (JSON/JSONL loader)

- Resolves relative `source` path against project root.
- Supports:
  - `json_lines` mode (`True`, `False`, or auto-detect),
  - `rename_map`,
  - `required_columns`,
  - `auto_create_column` with `auto_create_pattern`,
  - column projection via `columns`,
  - `fillna_defaults`,
  - sequence normalization (`normalize_sequence`, `sequence_column`),
  - standard amino-acid filtering (`keep_standard_amino_acids_only`).
- Resets index at end with `reset_index(drop=True)`.

### Important Behavior Notes (As Implemented)

- Both loaders currently return `None` from `load_data(...)` even though the abstract contract says `-> Any`.
- Both loaders call `sys.exit(1)` on failures and when `get_data()` is called before load.
- `dataloader.py` imports `BaseDataLoader` via `from base import BaseDataLoader` (script-style import), unlike package-style imports elsewhere.

---

## How To Add A New DataLoader

1. Create a class inheriting from `BaseDataLoader`.
2. Initialize internal storage in `__init__` (for example `self.data = None`).
3. Implement `load_data(source, **kwargs)` for one source type/strategy.
4. Implement `get_data()` with explicit pre-load failure behavior.
5. Document accepted kwargs and output schema.
6. Add tests for success, malformed input, missing input, and pre-load access.

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
        # 2) Read content
        # 3) Normalize schema
        self.data = ...
        return self.data

    def get_data(self) -> Any:
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_data(...) first.")
        return self.data
```

---

## Developer Checklist

- `load_data` validates path and schema assumptions.
- Errors are explicit and actionable (prefer exceptions over process exit).
- `get_data` has a clear pre-load failure mode.
- Output shape is documented (column names/types).
- Tests cover normal and failure paths.

