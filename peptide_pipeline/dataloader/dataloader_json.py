import sys
import os
import json
import re
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from . import BaseDataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class DataLoader(BaseDataLoader):
    """
    Data Loader Agent for peptide JSON data.
    Responsible for loading and preprocessing peptide data for training or evaluation.
    """

    def __init__(self):
        super().__init__()
        self.data = None
        self.logger.info("DataLoader initialized.")


    def load_data(
        self,
        source: str = "ai_training_peptides.json",
        columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        rename_map: Optional[Dict[str, str]] = None,
        auto_create_column: Optional[str] = None,
        auto_create_pattern: str = "row_{i}",
        json_lines: Optional[bool] = None,
        fillna_defaults: Optional[Dict[str, Any]] = None,
        normalize_sequence: bool = False,
        sequence_column: str = "sequence",
        keep_standard_amino_acids_only: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Loads data from a JSON file.

        Parameters are schema-driven to avoid hardcoded column names:
        - columns: subset of columns to keep (None => keep all)
        - required_columns: columns that must exist
        - rename_map: optional renaming map {old_name: new_name}
        - auto_create_column: create this column if missing
        - auto_create_pattern: format pattern for generated values (uses i)
        - json_lines:
            * True  => parse as JSONL
            * False => parse as standard JSON
            * None  => try both + manual fallback
                - fillna_defaults: dictionary of default values used to replace NaN/null values.
                - normalize_sequence: if True, strips spaces and uppercases `sequence_column`.
                - keep_standard_amino_acids_only: if True, keep only rows where `sequence_column`
                    contains standard amino acids.
        """
        try:
            if not os.path.isabs(source):
                source = os.path.join(PROJECT_ROOT, source)

            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")

            # Load JSON using requested mode (or auto-detect)
            if json_lines is True:
                self.data = pd.read_json(source, lines=True)
            elif json_lines is False:
                self.data = pd.read_json(source)
            else:
                try:
                    self.data = pd.read_json(source)
                except ValueError:
                    try:
                        self.data = pd.read_json(source, lines=True)
                    except ValueError:
                        with open(source, "r", encoding="utf-8") as f:
                            raw = json.load(f)
                        self.data = pd.DataFrame(raw)

            # Optional renaming
            if rename_map:
                self.data.rename(columns=rename_map, inplace=True)

            # Optional required-column validation
            required_columns = required_columns or []
            missing_required = [col for col in required_columns if col not in self.data.columns]
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")

            # Optional auto-create column
            if auto_create_column and auto_create_column not in self.data.columns:
                self.data[auto_create_column] = [
                    auto_create_pattern.format(i=i) for i in range(len(self.data))
                ]

            # Optional projection
            if columns is not None:
                missing_requested = [col for col in columns if col not in self.data.columns]
                if missing_requested:
                    raise ValueError(f"Requested columns not found: {missing_requested}")
                self.data = self.data[columns]

            if fillna_defaults:
                for col, value in fillna_defaults.items():
                    if col in self.data.columns:
                        self.data[col] = self.data[col].fillna(value)

            # Optional sequence normalization/cleanup.
            if normalize_sequence and sequence_column in self.data.columns:
                self.data[sequence_column] = (
                    self.data[sequence_column]
                    .astype(str)
                    .str.replace(" ", "", regex=False)
                    .str.upper()
                )

            if keep_standard_amino_acids_only and sequence_column in self.data.columns:
                self.data = self.data[
                    self.data[sequence_column].str.fullmatch(
                        rf"[{re.escape(AMINO_ACIDS)}]+",
                        na=False,
                    )
                ]

            self.data = self.data.reset_index(drop=True)

            self.logger.info(f"Data loaded successfully from {source}. Total records: {len(self.data)}")

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            sys.exit(1)


    def get_data(self) -> pd.DataFrame:
        """
        Returns the loaded and processed data as a pandas DataFrame.
        """
        if self.data is None:
            self.logger.error("Data not loaded. Call load_data() first.")
            sys.exit(1)
        self.logger.info("Data retrieval successful.")
        return self.data


