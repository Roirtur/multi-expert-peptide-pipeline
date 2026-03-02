import csv
import sys
import os
import pandas as pd
import torch
from typing import List, Any, Optional
from base import BaseDataLoader
import re

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class VAEDataLoader(BaseDataLoader):
    """
    Data loader for the VAE generator.
    Loads and preprocesses peptide sequences from a CSV file.
    """

    def __init__(self, seq_length: int = 6):
        """
        Args:
            seq_length: Only keep sequences of this length.
        """
        super().__init__()
        self.seq_length = seq_length
        self._peptides: List[str] = []
        self._one_hot: Optional[torch.Tensor] = None

    def load_data(self, source: str, **kwargs) -> List[str]:
        """
        Loads peptide sequences from a CSV file, filtering by length
        and removing sequences with non-standard amino acids.
        """
        # Try reading header with csv to find the sequence column index
        try:
            with open(source, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
        except Exception:
            # try with latin-1 if utf-8 fails
            with open(source, newline='', encoding='latin-1') as f:
                reader = csv.reader(f)
                header = next(reader)

        def clean(h: str) -> str:
            return re.sub(r'[^a-z0-9]', '', str(h).lower())

        cleaned = [clean(h) for h in header]
        seq_idx = None
        for i, ch in enumerate(cleaned):
            if 'sequence' in ch:
                seq_idx = i
                print( "i ch", i, ch)

                break
        if seq_idx is None:
            raise KeyError(f"Could not find a 'sequence' column in CSV header. Found columns: {header}")

        # Read full CSV into DataFrame (let pandas handle quoting/delimiters)
        try:
            df = pd.read_csv(source, dtype=str, skipinitialspace=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV with pandas: {e}")

        # Use positional index to extract the sequence column (robust to weird column names)
        if seq_idx >= len(df.columns):
            raise KeyError(f"Detected sequence column index {seq_idx} but DataFrame has columns {list(df.columns)}")

        original_col_name = df.columns[seq_idx]
        seq_values = df[original_col_name].astype(str).str.strip().str.replace('"', "", regex=False)

        # Keep a stable column name
        df = df.copy()
        df["sequence"] = seq_values

        # Filter by length
        df = df[df["sequence"].str.len() == self.seq_length]

        # Filter out sequences with non-standard amino acids
        valid_aa_set = set(AMINO_ACIDS)
        df = df[df["sequence"].apply(lambda s: all(c in valid_aa_set for c in s))]

        self._peptides = df["sequence"].tolist()
        self._one_hot = self._encode(self._peptides)

        print(f"Loaded {len(self._peptides)} valid sequences of length {self.seq_length}")
        return self._peptides

    def get_data(self) -> torch.Tensor:
        """
        Returns the one-hot encoded tensor of loaded sequences.

        Returns:
            Tensor of shape [n_sequences, seq_length * 20]
        """
        if self._one_hot is None:
            raise RuntimeError("No data loaded. Call load_data() first.")
        return self._one_hot

    def get_peptides(self) -> List[str]:
        """
        Returns the raw peptide sequence strings.
        """
        if not self._peptides:
            raise RuntimeError("No data loaded. Call load_data() first.")
        return self._peptides

    def _encode(self, peptides: List[str]) -> torch.Tensor:
        """
        Converts peptide sequences to one-hot encoded tensors.

        Returns:
            Tensor of shape [n_sequences, seq_length * 20]
        """
        input_dim = self.seq_length * 20
        one_hot = torch.zeros(len(peptides), input_dim)

        for i, peptide in enumerate(peptides):
            for j, aa in enumerate(peptide):
                if aa in AMINO_ACIDS:
                    idx = AMINO_ACIDS.index(aa)
                    one_hot[i, j * 20 + idx] = 1.0

        return one_hot


# --- Convenience functions (backwards compatible) ---

def load_peptides_from_csv(csv_path: str, seq_length: int = 6) -> List[str]:
    loader = VAEDataLoader(seq_length=seq_length)
    return loader.load_data(csv_path)


def peptides_to_one_hot(peptides: List[str], seq_length: int = 6) -> torch.Tensor:
    loader = VAEDataLoader(seq_length=seq_length)
    return loader._encode(peptides)