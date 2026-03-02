#### TEMPORARY FILE FOR TESTING VAE GENERATOR ####

import sys
import os
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from vae_generator import VAEGenerator
from  peptide_pipeline.dataloader.vae_dataloader import VAEDataLoader

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peptides.csv")
SEQ_LENGTH = 6
INPUT_DIM = SEQ_LENGTH * 20  # 120

def main():
    print("=== VAE Generator - Hexapeptide Training Test ===\n")

    # 1. Load sequences from CSV
    print("1. Loading sequences from peptides.csv...")
    loader = VAEDataLoader(seq_length=SEQ_LENGTH)
    peptides = loader.load_data(CSV_PATH)
    if not peptides:
        print("   ERROR: No valid sequences found. Exiting.")
        return
    print(f"   Example sequences: {peptides[:5]}\n")

    # 2. Convert to one-hot
    print("2. Encoding sequences to one-hot tensors...")
    one_hot = loader.get_data()
    print(f"   Tensor shape: {one_hot.shape}\n")  # Expected: [n, 120]

    # 3. Initialize model
    print("3. Initializing VAEGenerator...")
    vae = VAEGenerator(input_dim=INPUT_DIM, latent_dim=16, hidden_dim=64)
    print(f"   Device:     {vae.device}")
    print(f"   Input dim:  {vae.input_dim}")
    print(f"   Latent dim: {vae.latent_dim}")
    print(f"   Hidden dim: {vae.hidden_dim}\n")

    # 4. Train the model
    print("4. Training VAE...")
    one_hot = one_hot.to(vae.device)
    vae.train_model(one_hot, epochs=100, batch_size=16, lr=1e-3)
    print("   Training complete!\n")

    # 5. Generate new peptides
    print("5. Generating 10 peptides from latent space...")
    generated = vae.generate_peptides(count=10)
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    for i, pep in enumerate(generated):
        valid = all(c in amino_acids for c in pep) and len(pep) == SEQ_LENGTH
        status = "✓" if valid else "✗ INVALID"
        print(f"   Peptide {i+1:02d}: {pep}  {status}")

    # 6. Novelty check — how many are not in training set
    print("\n6. Novelty check...")
    training_set = set(peptides)
    novel = [p for p in generated if p not in training_set]
    print(f"   Novel peptides (not in training data): {len(novel)}/{len(generated)}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()