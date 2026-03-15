import torch
import numpy as np

from data_handler import IDX_TO_AA

# Target Features: [length, ph, molecular_weight, logp, net_charge, isoelectric_point, hydrophobicity, cathionicity]
DEFAULT_TARGET_PROPERTIES = [10, 7.0, 1200.0, 2.5, 3.0, 10.0, -0.5, 3]

def generate_peptides(
    model,
    scaler,
    num_samples=5,
    properties=DEFAULT_TARGET_PROPERTIES,
    temperature=1.0,
    top_k=5,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if model is None:
        print("Error: A trained PeptideCVAE model instance is required.")
        return

    if scaler is None:
        print("Error: A fitted scaler instance is required.")
        return

    model = model.to(device)

    if not hasattr(scaler, "transform"):
        print("Error: Provided scaler must implement a transform(...) method.")
        return

    model.eval()
    print("Model loaded successfully.")
    
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
    
    # scale the properties to match the training distributions
    scaled_properties = scaler.transform(desired_properties)
    condition_tensor = torch.tensor(scaled_properties, dtype=torch.float32).to(device)

    print(f"\n--- Generating {num_samples} novel peptide(s) ---")
    
    with torch.no_grad():
        generated_integers = model.generate(
            condition_tensor,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
        )

    # decode back to Amino Acid characters
    outputs = []

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
        
        outputs.append(peptide)
        print(f"Sample {i+1}: {peptide}")

    return outputs

