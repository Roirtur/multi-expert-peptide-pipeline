import torch
import torch.optim as optim
import numpy as np

# Import from the files we just created
from data_handler import get_dataloader, IDX_TO_AA
from model import PeptideCVAE, cvae_loss_function

def train_model(dataset_file):
    # --- Hyperparameters ---
    BATCH_SIZE = 32
    MAX_LEN = 12
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    LATENT_DIM = 32
    
    # Device configuration (use GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    # Make sure 'dataset.json' is in the same directory
    dataloader, vocab_size, condition_dim = get_dataloader(
        json_file=dataset_file, 
        batch_size=BATCH_SIZE, 
        max_len=MAX_LEN,
        is_train=True
    )
    
    # Max sequence length in the model is MAX_LEN + 2 (for <SOS> and <EOS>)
    max_seq_len = MAX_LEN + 2

    # --- 2. Initialize Model and Optimizer ---
    model = PeptideCVAE(
        vocab_size=vocab_size, 
        condition_dim=condition_dim, 
        max_seq_len=max_seq_len,
        latent_dim=LATENT_DIM
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop with KL Annealing ---
    model.train()
    for epoch in range(EPOCHS):
        total_loss_val = 0
        total_recon_val = 0
        total_kl_val = 0
        
        # KL Annealing: Slowly increase beta from 0 to 1 over the first half of training
        # This prevents "posterior collapse" where the model ignores the latent space
        beta = min(1.0, epoch / (EPOCHS * 0.5))
        
        for batch_idx, (sequences, conditions) in enumerate(dataloader):
            sequences = sequences.to(device)
            conditions = conditions.to(device)
            
            # Forward pass
            recon_x, mu, logvar = model(sequences, conditions)
            
            # Calculate loss
            loss, recon_loss, kl_loss = cvae_loss_function(recon_x, sequences, mu, logvar, beta=beta)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss_val += loss.item()
            total_recon_val += recon_loss.item()
            total_kl_val += kl_loss.item()
            
        # Print epoch summary
        avg_loss = total_loss_val / len(dataloader.dataset)
        avg_recon = total_recon_val / len(dataloader.dataset)
        avg_kl = total_kl_val / len(dataloader.dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Beta: {beta:.2f} | "
                  f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    # --- 4. Save the Model ---
    torch.save(model.state_dict(), 'cvae_peptide_model.pth')
    print("Model saved to 'cvae_peptide_model.pth'")
    
    return model, device

def generate_sample(model, device):
    """Demonstrates how to generate a peptide and convert integers back to strings."""
    import pickle
    
    print("\n--- Generating a novel peptide ---")
    
    # 1. Load the exact scaler used during training
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # 2. Define desired properties (MUST be in the exact order as training!)
    # [length, ph, molecular_weight, logp, net_charge, isoelectric_point, hydrophobicity, cathionicity]
    desired_properties = np.array([[10, 7.0, 1200.0, 2.5, 3.0, 10.0, -0.5, 3]])
    
    # 3. Scale the properties
    scaled_properties = scaler.transform(desired_properties)
    condition_tensor = torch.tensor(scaled_properties, dtype=torch.float32).to(device)
    
    # 4. Generate sequences (e.g., generate 3 variations for these exact conditions)
    model.eval()
    generated_integers = model.generate(condition_tensor, num_samples=3)
    
    # 5. Decode integers back to Amino Acid strings
    for i, seq_tensor in enumerate(generated_integers):
        seq_list = seq_tensor.tolist()
        
        peptide = ""
        for idx in seq_list:
            token = IDX_TO_AA[idx]
            if token == '<SOS>': continue
            if token == '<EOS>' or token == '<PAD>': break
            peptide += token
            
        print(f"Sample {i+1}: {peptide}")

if __name__ == "__main__":
    trained_model, compute_device = train_model(dataset_file="../database/test.json")
    generate_sample(trained_model, compute_device)