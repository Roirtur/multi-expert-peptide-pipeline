import argparse
import torch
import torch.optim as optim

from data_handler import get_dataloader
from model import PeptideCVAE, cvae_loss_function

# --- Default Hyperparameters ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LEN = 12
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_LATENT_DIM = 32
DEFAULT_MODEL_PATH = 'cvae_peptide_model.pth'

def train_model(
    dataset_file,
    batch_size=DEFAULT_BATCH_SIZE,
    max_len=DEFAULT_MAX_LEN,
    epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    latent_dim=DEFAULT_LATENT_DIM,
    model_path=DEFAULT_MODEL_PATH,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    dataloader, vocab_size, condition_dim = get_dataloader(
        json_file=dataset_file, 
        batch_size=batch_size, 
        max_len=max_len,
        is_train=True
    )
    
    # max_len + 2 for <SOS> and <EOS>
    max_seq_len = max_len + 2

    # --- 2. Initialize Model and Optimizer ---
    model = PeptideCVAE(
        vocab_size=vocab_size, 
        condition_dim=condition_dim, 
        max_seq_len=max_seq_len,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 3. Training Loop with KL Annealing ---
    model.train()
    for epoch in range(epochs):
        total_loss_val = 0
        total_recon_val = 0
        total_kl_val = 0
        seen_samples = 0
        
        # KL Annealing: Slowly increase beta from 0 to 1 over the first half of training
        # This prevents "posterior collapse" where the model ignores the latent space
        beta = min(1.0, epoch / (epochs * 0.5))
        
        for _, (sequences, conditions) in enumerate(dataloader):
            sequences = sequences.to(device)
            conditions = conditions.to(device)
            
            recon_x, mu, logvar = model(sequences, conditions)

            targets = sequences[:, 1:]
            loss, recon_loss, kl_loss = cvae_loss_function(recon_x, targets, mu, logvar, beta=beta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_curr = sequences.size(0)
            seen_samples += batch_size_curr
            
            total_loss_val += loss.item()
            total_recon_val += recon_loss.item()
            total_kl_val += kl_loss.item()
            
        avg_loss = total_loss_val / max(seen_samples, 1)
        avg_recon = total_recon_val / max(seen_samples, 1)
        avg_kl = total_kl_val / max(seen_samples, 1)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Beta: {beta:.2f} | "
                  f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    # --- 4. Save the Model ---
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")
    
    return model, device