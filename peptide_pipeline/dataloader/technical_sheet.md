# DataLoader Technical Sheet

## Purpose

In this project, a DataLoader is the component that turns a raw data source into a clean, usable in-memory object for the rest of the pipeline.

Its role is to:

- connect to a source (file, URL, API, database export, etc.),
- parse and normalize the content,
- perform validation and lightweight preprocessing,
- expose ready-to-use data through a stable interface.

The contract is defined by `BaseDataLoader` in `peptide_pipeline/dataloader/base.py`.

---

## Base Interface (What Every New DataLoader Must Implement)

Any new dataloader must inherit from `BaseDataLoader` and implement exactly these abstract methods:

### 1) `load_data(self, source: str, **kwargs) -> Any`

This method is responsible for reading the source and preparing data.

Expected behavior:

- accept `source` as the primary location/identifier of the dataset,
- use `**kwargs` for optional loader options (format-specific settings, filters, split choices, credentials, etc.),
- parse and preprocess data,
- store the processed result in an internal attribute (for example `self._data`),
- return the loaded data object as well.

Implementation notes:

- validate inputs early (missing file, unsupported format, wrong parameters),
- fail with explicit errors instead of silent fallbacks,
- keep heavy transformations explicit and documented,
- keep I/O logic and cleaning logic readable and separated when possible.

### 2) `get_data(self) -> Any`

This method returns the current processed dataset already loaded by `load_data`.

Expected behavior:

- return the internal data object in a stable structure,
- raise a clear error if called before `load_data`,
- avoid hidden reloading or side effects.

Implementation notes:

- `get_data` should be predictable and cheap,
- do not mutate data when serving it,
- if defensive copying is needed, do it intentionally and document it.

---

## How To Add a New DataLoader

Use this workflow when creating a new loader implementation.

1. Create a new class that inherits from `BaseDataLoader`.
2. Add an initializer (`__init__`) to define internal state (for example `self._data = None`).
3. Implement `load_data(source, **kwargs)` with source-specific parsing and preprocessing.
4. Implement `get_data()` to return the prepared data.
5. Add clear validation and error messages for invalid source/options.
6. Ensure output shape/type is documented and consistent.
7. Add usage examples and tests for normal and failure cases.

---

## Minimal Skeleton

```python
from typing import Any
from peptide_pipeline.dataloader.base import BaseDataLoader


class MyDataLoader(BaseDataLoader):
	def __init__(self):
		self._data = None

	def load_data(self, source: str, **kwargs) -> Any:
		# 1) Validate source/options
		# 2) Read raw content
		# 3) Clean/normalize
		# 4) Store in self._data
		self._data = ...
		return self._data

	def get_data(self) -> Any:
		if self._data is None:
			raise RuntimeError("Data not loaded. Call load_data(...) first.")
		return self._data
```

---

## Design Guidelines For New Developers

- Keep one loader focused on one source format or one loading strategy.
- Prefer explicit options in `**kwargs` and document accepted keys.
- Always make error messages actionable (what failed and how to fix it).
- Keep preprocessing deterministic so training results are reproducible.
- If you change output structure, communicate it clearly to downstream users.

---

## Quick Validation Checklist

Before opening a PR for a new dataloader:

- `load_data` handles valid input correctly.
- `load_data` fails clearly on bad input.
- `get_data` returns expected object after loading.
- `get_data` fails clearly if data is not loaded yet.
- Method behavior is documented (inputs, output shape/type, key options).

