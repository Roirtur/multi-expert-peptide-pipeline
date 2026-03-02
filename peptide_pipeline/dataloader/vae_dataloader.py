import sys
import os
import pandas as pd
import torch
from typing import List, Any, Optional
from peptide_pipeline.dataloader.base import BaseDataLoader

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

        Args:
            source: Path to the CSV file.

        Returns:
            List of valid peptide sequences.
        """
        df = pd.read_csv(source)

        # Strip whitespace from sequence column
        df["SEQUENCE"] = df["SEQUENCE"].astype(str).str.strip()

        # Filter by length
        df = df[df["SEQUENCE"].str.len() == self.seq_length]

        # Filter out sequences with non-standard amino acids
        valid_aa_set = set(AMINO_ACIDS)
        df = df[df["SEQUENCE"].apply(lambda s: all(c in valid_aa_set for c in s))]

        self._peptides = df["SEQUENCE"].tolist()
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