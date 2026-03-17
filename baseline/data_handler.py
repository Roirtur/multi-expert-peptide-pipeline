from typing import Any, Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Define the vocabulary for standard amino acids plus special tokens
SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
VOCAB = SPECIAL_TOKENS + AMINO_ACIDS

# Create mapping dictionaries
AA_TO_IDX = {aa: idx for idx, aa in enumerate(VOCAB)}
IDX_TO_AA = {idx: aa for idx, aa in enumerate(VOCAB)}

CONDITION_DEFAULTS = {
    "length": 10,
    "ph": 7.0,
    "molecular_weight": 1500.0,
    "logp": -0.2,
    "net_charge": 5.0,
    "isoelectric_point": 10.0,
    "hydrophobicity": 1.0,
    "cathionicity": 5,
}

class PeptideDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        max_len: int = 12,
        scaler_path: str = 'scaler.pkl',
        is_train: bool = True,
        scaler: Optional[StandardScaler] = None,
    ):
        """
        Args:
            records (List[Dict[str, Any]]): In-memory peptide records.
            max_len (int): Maximum length of the peptide sequence (default 12).
            scaler_path (str): Path to save/load the StandardScaler.
            is_train (bool): If True, fits the scaler. If False, loads the scaler.
            scaler (StandardScaler): Optional pre-fitted scaler to reuse.
        """
        self.max_len = max_len
        self.scaler_path = scaler_path
        self.is_train = is_train
        self.scaler = scaler if scaler is not None else StandardScaler()
        if not records:
            raise ValueError("records cannot be empty.")
            
        self.sequences: List[str] = []
        self.conditions: List[List[float]] = []
        
        # handle missing values
        for item in records:
            seq = str(item.get("sequence", "")).upper()
            if len(seq) > self.max_len:
                raise ValueError(
                    f"Found sequence longer than max_len={self.max_len}. "
                    "Filter records before creating PeptideDataset."
                )
                
            self.sequences.append(seq)

            cond = self._extract_conditions(item)
            self.conditions.append(cond)

        if not self.conditions:
            raise ValueError(
                "No valid peptides were loaded. Check max_len and dataset content."
            )

        self.conditions = np.array(self.conditions, dtype=np.float32)
        
        if self.is_train:
            self.conditions = self.scaler.fit_transform(self.conditions)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to {self.scaler_path}")
        else:
            if scaler is not None:
                self.conditions = self.scaler.transform(self.conditions)
            elif os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.conditions = self.scaler.transform(self.conditions)
            else:
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Train the dataset first.")

    def _extract_conditions(self, item: Dict[str, Any]) -> List[float]:
        """Extract condition vector in a fixed, model-compatible order."""
        defaults = dict(CONDITION_DEFAULTS)

        values: List[float] = []
        for key in CONDITION_DEFAULTS:
            raw_value = item.get(key)
            value = defaults[key] if raw_value is None else raw_value
            values.append(float(value))
        return values

    def __len__(self):
        return len(self.sequences)

    def encode_sequence(self, seq):
        """Converts a string sequence into a padded list of integers."""
        # Start with <SOS>
        encoded = [AA_TO_IDX['<SOS>']]
        
        # Add amino acids
        for aa in seq:
            encoded.append(AA_TO_IDX.get(aa, AA_TO_IDX['<UNK>']))
            
        # Add <EOS>
        encoded.append(AA_TO_IDX['<EOS>'])
        
        # max_len + 2 (for SOS and EOS)
        target_len = self.max_len + 2
        padding_length = target_len - len(encoded)
        encoded.extend([AA_TO_IDX['<PAD>']] * padding_length)
        
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        seq_tensor = self.encode_sequence(self.sequences[idx])
        cond_tensor = torch.tensor(self.conditions[idx], dtype=torch.float32)
        
        return seq_tensor, cond_tensor

def get_dataloader(
    records,
    batch_size=32,
    max_len=12,
    is_train=True,
    shuffle=True,
    scaler_path='scaler.pkl',
    scaler=None,
):
    """Helper function to create a ready-to-use dataloader."""
    dataset = PeptideDataset(
        records=records,
        max_len=max_len,
        is_train=is_train,
        scaler_path=scaler_path,
        scaler=scaler,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    vocab_size = len(VOCAB)
    condition_dim = dataset.conditions.shape[1]
    
    return dataloader, vocab_size, condition_dim