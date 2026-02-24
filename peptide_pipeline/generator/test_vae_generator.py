#### TEMPORARY FILE FOR TESTING VAE GENERATOR ####

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vae_generator import VAEGenerator

def main():
    print("=== VAE Generator Basic Test ===\n")

    # 1. Initialize the model
    print("1. Initializing VAEGenerator...")
    vae = VAEGenerator(input_dim=420, latent_dim=64, hidden_dim=256)
    print(f"   Device: {vae.device}")
    print(f"   Input dim: {vae.input_dim}, Latent dim: {vae.latent_dim}, Hidden dim: {vae.hidden_dim}\n")

    # 2. Create some dummy training peptides (length 21)
    print("2. Creating dummy training data...")
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    import random
    random.seed(42)
    dummy_peptides = [
        "".join(random.choices(amino_acids, k=21)) for _ in range(100)
    ]
    print(f"   Created {len(dummy_peptides)} peptides")
    print(f"   Example: {dummy_peptides[0]}\n")

    # 3. Train the model
    print("3. Training the VAE for 10 epochs...")
    vae.fit(dummy_peptides, epochs=10, batch_size=16, lr=1e-3)
    print("   Training complete!\n")

    # 4. Generate peptides
    print("4. Generating 5 peptides from latent space...")
    generated = vae.generate_peptides(count=5)
    for i, pep in enumerate(generated):
        print(f"   Peptide {i+1}: {pep}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()