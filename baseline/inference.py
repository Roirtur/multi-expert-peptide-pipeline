import torch
import numpy as np
import pickle
import os

from data_handler import IDX_TO_AA, VOCAB
from model import PeptideCVAE

def generate_peptides(model_path='cvae_peptide_model.pth', scaler_path='scaler.pkl', num_samples=5, properties=None):
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file '{scaler_path}' not found. Please train the dataset first to generate the scaler.")
        return

    # 3. Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 4. Initialize the model
    vocab_size = len(VOCAB)
    condition_dim = scaler.mean_.shape[0]  # Typically 8 properties
    max_seq_len = 14  # MAX_LEN (12) + 2 (SOS, EOS) from training settings
    latent_dim = 32   # LATENT_DIM from training settings

    model = PeptideCVAE(
        vocab_size=vocab_size,
        condition_dim=condition_dim,
        max_seq_len=max_seq_len,
        latent_dim=latent_dim
    ).to(device)

    # 5. Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return
        
    model.eval()
    print("Model loaded successfully.")

    # 6. Define desired properties for generation
    # Target Features: [length, ph, molecular_weight, logp, net_charge, isoelectric_point, hydrophobicity, cathionicity]
    if properties is None:
        properties = [10, 7.0, 1200.0, 2.5, 3.0, 10.0, -0.5, 3]
        
    desired_properties = np.array([properties])
    
    print("\nTarget Properties:")
    print(f"- Length: {desired_properties[0][0]}")
    print(f"- pH: {desired_properties[0][1]}")
    print(f"- Molecular Weight: {desired_properties[0][2]}")
    print(f"- LogP: {desired_properties[0][3]}")
    print(f"- Net Charge: {desired_properties[0][4]}")
    print(f"- Isoelectric Point: {desired_properties[0][5]}")
    print(f"- Hydrophobicity: {desired_properties[0][6]}")
    print(f"- Cathionicity: {desired_properties[0][7]}")
    
    # 7. Scale the properties to match the training distributions
    scaled_properties = scaler.transform(desired_properties)
    condition_tensor = torch.tensor(scaled_properties, dtype=torch.float32).to(device)

    print(f"\n--- Generating {num_samples} novel peptide(s) ---")
    
    # 8. Generate sequence integers
    with torch.no_grad():
        generated_integers = model.generate(condition_tensor, num_samples=num_samples)

    # 9. Decode integers back to Amino Acid characters
    for i, seq_tensor in enumerate(generated_integers):
        seq_list = seq_tensor.tolist()
        
        peptide = ""
        for idx in seq_list:
            token = IDX_TO_AA.get(idx, '<UNK>')
            if token == '<SOS>': 
                continue
            if token == '<EOS>' or token == '<PAD>': 
                break
            peptide += token
            
        print(f"Sample {i+1}: {peptide}")

if __name__ == "__main__":
    # Example usage with custom properties:
    # [length, ph, MW, logp, net_charge, isoelectric_point, hydrophobicity, cathionicity]
    custom_props = [11, 7, 1475.81, -1.80, 3.0, 14.0, -0.17, 3]
    generate_peptides(properties=custom_props)
