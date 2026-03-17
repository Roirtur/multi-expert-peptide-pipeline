import json
import torch
import torch.optim as optim

from data_handler import get_dataloader
from model import PeptideCVAE, cvae_loss_function

# default hyper params
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LEN = 12
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_LATENT_DIM = 32
DEFAULT_MODEL_PATH = 'cvae_peptide_model.pth'

def train_model(
    dataset_file,
    scaler_path='scaler.pkl',
    batch_size=DEFAULT_BATCH_SIZE,
    max_len=DEFAULT_MAX_LEN,
    epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    latent_dim=DEFAULT_LATENT_DIM,
    model_path=DEFAULT_MODEL_PATH,
    val_split=0.1,
    random_seed=42,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(dataset_file, 'r') as f:
        raw_data = json.load(f)

    # Filter before splitting to avoid empty splits after max_len filtering.
    filtered_data = []
    for item in raw_data:
        seq = str(item.get("sequence", "")).upper()
        if len(seq) <= max_len:
            filtered_data.append(item)

    if not filtered_data:
        raise ValueError("No valid peptides after filtering by max_len.")

    train_data = filtered_data
    val_data = []
    if 0.0 < val_split < 1.0 and len(filtered_data) >= 2:
        generator = torch.Generator().manual_seed(random_seed)
        perm = torch.randperm(len(filtered_data), generator=generator).tolist()
        val_size = max(1, int(len(filtered_data) * val_split))
        if len(filtered_data) - val_size <= 0:
            val_size = 1
        val_indices = set(perm[:val_size])
        train_data = [item for i, item in enumerate(filtered_data) if i not in val_indices]
        val_data = [item for i, item in enumerate(filtered_data) if i in val_indices]

    train_loader, vocab_size, condition_dim = get_dataloader(
        records=train_data,
        batch_size=batch_size,
        max_len=max_len,
        is_train=True,
        shuffle=True,
        scaler_path=scaler_path,
    )

    val_loader = None
    if val_data:
        val_loader, _, _ = get_dataloader(
            records=val_data,
            batch_size=batch_size,
            max_len=max_len,
            is_train=False,
            shuffle=False,
            scaler_path=scaler_path,
            scaler=train_loader.dataset.scaler,
        )
    
    # max_len + 2 for <SOS> and <EOS>
    max_seq_len = max_len + 2

    model = PeptideCVAE(
        vocab_size=vocab_size, 
        condition_dim=condition_dim, 
        max_seq_len=max_seq_len,
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        train_seen_samples = 0
        
        beta = min(1.0, epoch / (epochs * 0.5))
        
        for _, (sequences, conditions) in enumerate(train_loader):
            sequences = sequences.to(device)
            conditions = conditions.to(device)
            
            recon_x, mu, logvar = model(sequences, conditions)

            targets = sequences[:, 1:]
            loss, recon_loss, kl_loss = cvae_loss_function(recon_x, targets, mu, logvar, beta=beta)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_curr = sequences.size(0)
            train_seen_samples += batch_size_curr

            train_loss_sum += loss.item() * batch_size_curr
            train_recon_sum += recon_loss.item() * batch_size_curr
            train_kl_sum += kl_loss.item() * batch_size_curr

        avg_train_loss = train_loss_sum / max(train_seen_samples, 1)
        avg_train_recon = train_recon_sum / max(train_seen_samples, 1)
        avg_train_kl = train_kl_sum / max(train_seen_samples, 1)

        avg_val_loss = None
        avg_val_recon = None
        avg_val_kl = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_recon_sum = 0.0
            val_kl_sum = 0.0
            val_seen_samples = 0

            with torch.no_grad():
                for sequences, conditions in val_loader:
                    sequences = sequences.to(device)
                    conditions = conditions.to(device)

                    recon_x, mu, logvar = model(sequences, conditions)
                    targets = sequences[:, 1:]
                    val_loss, val_recon, val_kl = cvae_loss_function(
                        recon_x,
                        targets,
                        mu,
                        logvar,
                        beta=beta,
                    )

                    batch_size_curr = sequences.size(0)
                    val_seen_samples += batch_size_curr
                    val_loss_sum += val_loss.item() * batch_size_curr
                    val_recon_sum += val_recon.item() * batch_size_curr
                    val_kl_sum += val_kl.item() * batch_size_curr

            avg_val_loss = val_loss_sum / max(val_seen_samples, 1)
            avg_val_recon = val_recon_sum / max(val_seen_samples, 1)
            avg_val_kl = val_kl_sum / max(val_seen_samples, 1)
            model.train()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = (
                f"Epoch [{epoch+1}/{epochs}] | Beta: {beta:.2f} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Recon: {avg_train_recon:.4f} | "
                f"Train KL: {avg_train_kl:.4f}"
            )
            if avg_val_loss is not None:
                msg += (
                    f" | Val Loss: {avg_val_loss:.4f}"
                    f" | Val Recon: {avg_val_recon:.4f}"
                    f" | Val KL: {avg_val_kl:.4f}"
                )
            print(msg)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")
    
    return model, device