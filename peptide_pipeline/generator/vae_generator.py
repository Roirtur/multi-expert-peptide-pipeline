import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Dict, Any
import numpy as np
from abc import ABC
from peptide_pipeline.generator.base import BaseGenerator

class VAEGenerator(BaseGenerator):

    def __init__(self, input_dim: int = 120, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.logger.debug(f"Initialized VAEGenerator with input_dim={input_dim}, latent_dim={latent_dim}, hidden_dim={hidden_dim}")

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)
        )

        # Decoder — NO Sigmoid, raw logits for cross_entropy
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
            # No Sigmoid — raw logits fed to cross_entropy
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        self.logger.debug(f"Reparameterization: mu shape {mu.shape}, log_var shape {log_var.shape}, std shape {std.shape}, eps shape {eps.shape}")
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self._reparameterize(mu, log_var)
        self.logger.debug(f"Forward pass: input shape {x.shape}, encoded shape {h.shape}, mu shape {mu.shape}, log_var shape {log_var.shape}, z shape {z.shape}")
        return self.decoder(z), mu, log_var

    def generate_peptides(self, count: int, constraints: Optional[Dict[str, Any]] = None, temperature: float = 1.0) -> List[str]:
        self.eval()
        with torch.no_grad():
            z = torch.randn(count, self.latent_dim, device=self.device)
            logits = self.decoder(z)
            num_positions = self.input_dim // 20
            logits = logits.view(count, num_positions, 20)
            # Temperature sampling for diversity — higher = more random
            probs = F.softmax(logits / temperature, dim=-1)
            amino_acid_indices = torch.multinomial(
                probs.view(-1, 20), num_samples=1
            ).view(count, num_positions).cpu().numpy()
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            peptides = ["".join([amino_acids[idx] for idx in seq]) for seq in amino_acid_indices]
        self.train()
        return peptides

    def modify_peptides(self, peptides: List[str], feedback: Optional[Any] = None) -> List[str]:
        count = len(peptides)
        if feedback and isinstance(feedback, dict) and "count" in feedback:
            count = feedback["count"]
        return self.generate_peptides(count)

    def train_model(self, data: torch.Tensor, epochs: int = 300, batch_size: int = 64, lr: float = 1e-3, kl_anneal_epochs: int = 1000) -> None:
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num_positions = self.input_dim // 20
        self.logger.info(f"Starting training for {epochs} epochs with batch size {batch_size} and learning rate {lr}")
        for epoch in range(epochs):
            kl_weight = min(1.0, epoch / max(1, kl_anneal_epochs))
            epoch_recon = 0.0
            epoch_kl = 0.0

            for batch in dataloader:
                x = batch[0].to(self.device)
                x_recon, mu, log_var = self.forward(x)

                # Reshape for cross entropy
                recon_logits = x_recon.view(-1, num_positions, 20)
                targets = x.view(-1, num_positions, 20).argmax(dim=-1)  # [batch, positions]

                recon_loss = F.cross_entropy(
                    recon_logits.reshape(-1, 20),
                    targets.reshape(-1)
                )
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

                loss = recon_loss + kl_weight * kl_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            scheduler.step()

            if (epoch + 1) % 50 == 0:
                avg_recon = epoch_recon / len(dataloader)
                avg_kl = epoch_kl / len(dataloader)
                self.logger.info(f"  Epoch {epoch+1:03d}/{epochs} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | KL weight: {kl_weight:.2f}")

    def _peptides_to_one_hot(self, peptides: List[str]) -> torch.Tensor:
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        one_hot = torch.zeros(len(peptides), self.input_dim, device=self.device)
        for i, peptide in enumerate(peptides):
            for j, aa in enumerate(peptide):
                if aa in amino_acids:
                    idx = amino_acids.index(aa)
                    one_hot[i, j * 20 + idx] = 1
        return one_hot

    def save_model(self, path: str) -> None:
        self.logger.info(f"Saving model to {path}")
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.logger.info(f"Loading model from {path}")
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)