import json
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

class PeptideDataset(Dataset):
    def __init__(self, json_file, max_len=12, default_ph=7.0, scaler_path='scaler.pkl', is_train=True):
        """
        Args:
            json_file (str): Path to the dataset.json file.
            max_len (int): Maximum length of the peptide sequence (default 12).
            default_ph (float): Value to replace null pH values with.
            scaler_path (str): Path to save/load the StandardScaler.
            is_train (bool): If True, fits the scaler. If False, loads the scaler.
        """
        self.max_len = max_len
        self.default_ph = default_ph
        self.scaler_path = scaler_path
        self.is_train = is_train
        
        # Load the JSON data
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
            
        self.sequences = []
        self.conditions = []
        
        # Extract data and handle missing values
        for item in raw_data:
            seq = item.get("sequence", "")
            if len(seq) > self.max_len:
                continue # Skip sequences longer than the defined max_length
                
            self.sequences.append(seq)
            
            # Extract conditions in a fixed order
            ph = item.get("ph")
            ph = self.default_ph if ph is None else ph
            
            cond = [
                item.get("length", 0),
                ph,
                item.get("molecular_weight", 0.0),
                item.get("logp", 0.0),
                item.get("net_charge", 0.0),
                item.get("isoelectric_point", 0.0),
                item.get("hydrophobicity", 0.0),
                item.get("cathionicity", 0)
            ]
            self.conditions.append(cond)
            
        self.conditions = np.array(self.conditions)
        
        # Scale the conditions (crucial for neural networks)
        self.scaler = StandardScaler()
        if self.is_train:
            self.conditions = self.scaler.fit_transform(self.conditions)
            # Save the scaler so we can use the exact same one for generation later
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to {self.scaler_path}")
        else:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.conditions = self.scaler.transform(self.conditions)
            else:
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Train the dataset first.")

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
        
        # Pad to max length: max_len + 2 (for SOS and EOS)
        target_len = self.max_len + 2
        padding_length = target_len - len(encoded)
        encoded.extend([AA_TO_IDX['<PAD>']] * padding_length)
        
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        seq_tensor = self.encode_sequence(self.sequences[idx])
        cond_tensor = torch.tensor(self.conditions[idx], dtype=torch.float32)
        
        return seq_tensor, cond_tensor

def get_dataloader(json_file, batch_size=32, max_len=12, is_train=True, shuffle=True):
    """Helper function to create a ready-to-use dataloader."""
    dataset = PeptideDataset(json_file=json_file, max_len=max_len, is_train=is_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    # Also return VOCAB size and condition size for the model architecture
    vocab_size = len(VOCAB)
    condition_dim = dataset.conditions.shape[1]
    
    return dataloader, vocab_size, condition_dim