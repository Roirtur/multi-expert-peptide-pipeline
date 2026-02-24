import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Dict, Any
import numpy as np
from abc import ABC
from base import BaseGenerator

class VAEGenerator(BaseGenerator):
    """
    VAE-based peptide generator.
    """

    def __init__(self, input_dim: int = 420, latent_dim: int = 64, hidden_dim: int = 256):
        """
        Args:
            input_dim: Dimension of the input (one-hot encoded peptides).
                       Default: 420 (20 amino acids * 21 positions).
            latent_dim: Dimension of the latent space.
            hidden_dim: Dimension of the hidden layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mu and LogVar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)  # Output probabilities for each amino acid
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the VAE.
        """
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self._reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def generate_peptides(self, count: int, constraints: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generates peptide sequences by sampling from the latent space.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(count, self.latent_dim, device=self.device)
            one_hot = self.decoder(z)
            # Convert one-hot to amino acid indices
            amino_acid_indices = torch.argmax(one_hot, dim=-1).cpu().numpy()
            # Map indices to amino acid characters (0-19 -> A, C, D, ..., Y)
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            peptides = ["".join([amino_acids[idx] for idx in seq]) for seq in amino_acid_indices]
        return peptides

    def modify_peptides(self, peptides: List[str], feedback: Optional[Any] = None) -> List[str]:
        """
        Modifies peptides based on feedback (e.g., gradient-based optimization).
        For simplicity, this implementation just regenerates peptides.
        """
        return self.generate_peptides(len(peptides))

    def train(self, data: torch.Tensor, epochs: int = 50, batch_size: int = 32, lr: float = 1e-3) -> None:
        """
        Trains the VAE on peptide data.
        Args:
            data: One-hot encoded peptide sequences (shape: [n_sequences, input_dim]).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            lr: Learning rate.
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:
                x = batch[0].to(self.device)
                x_recon, mu, log_var = self.forward(x)

                # VAE loss: reconstruction + KL divergence
                recon_loss = F.mse_loss(x_recon, x, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _peptides_to_one_hot(self, peptides: List[str]) -> torch.Tensor:
        """
        Converts peptide sequences to one-hot encoding.
        """
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        one_hot = torch.zeros(len(peptides), self.input_dim, device=self.device)
        for i, peptide in enumerate(peptides):
            for j, aa in enumerate(peptide):
                if aa in amino_acids:
                    idx = amino_acids.index(aa)
                    one_hot[i, j * 20 + idx] = 1
        return one_hot

    def fit(self, peptides: List[str], **kwargs) -> None:
        """
        Convenience method to train the VAE from raw peptide sequences.
        """
        one_hot = self._peptides_to_one_hot(peptides)
        self.train(one_hot, **kwargs)
