import os
import sys
import tempfile
import pandas as pd
import torch

# Ensure repo root is importable
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from peptide_pipeline.dataloader.vae_dataloader import VAEDataLoader
def find_peptides_csv():
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "peptides.csv"),
        os.path.join(repo_root, "peptides.csv"),
        os.path.join(repo_root, "data", "peptides.csv"),
        os.path.join(repo_root, "peptide_pipeline", "generator", "peptides.csv"),
        os.path.join(repo_root, "peptide_pipeline", "peptides.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Could not find peptides.csv in expected locations. Looked at:\n" + "\n".join(candidates))

def test_vae_dataloader_loads_real_csv():
    seq_length = 6
    csv_path = find_peptides_csv()

    loader = VAEDataLoader(seq_length=seq_length)
    peptides = loader.load_data(csv_path)

    assert isinstance(peptides, list)
    assert len(peptides) > 0, f"No valid sequences of length {seq_length} found in {csv_path}"

    one_hot = loader.get_data()
    assert isinstance(one_hot, torch.Tensor)
    assert one_hot.shape[0] == len(peptides)
    assert one_hot.shape[1] == seq_length * 20

    # Values must be 0/1 and each position must have exactly one 1
    assert torch.logical_or(one_hot == 0.0, one_hot == 1.0).all()
    for row in range(one_hot.size(0)):
        for pos in range(seq_length):
            start = pos * 20
            end = start + 20
            s = one_hot[row, start:end].sum().item()
            assert abs(s - 1.0) < 1e-6

if __name__ == "__main__":
    test_vae_dataloader_loads_real_csv()
    print("VAE dataloader test passed (loaded peptides.csv).")