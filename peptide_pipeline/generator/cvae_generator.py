"""
CVAE conditioning constraints (accepted in `constraints` dict):

Each property should be provided as a scalar, e.g.:
{"size": 12, "net_charge_pH5_5": 2.0, "hydrophobicity": -0.3}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Dict, Any
from peptide_pipeline.generator.base import BaseGenerator

constraints_default = {
    "size": 10.0,
    "molecular_weight": 1500.0,
    "net_charge_pH5_5": 0.0,
    "isoelectric_point": 7.0,
    "hydrophobicity": 0.0,
    "cathionicity": 0.0,
    "hydrophobic_moment": 0.5,
    "logp": 0.0,
    "boman_index": 0.0,
    "h_bond_donors": 5.0,
    "h_bond_acceptors": 5.0,
    "tpsa": 100.0,
}

class CVAEGenerator(BaseGenerator):

    def __init__(self, input_dim: int = 120, latent_dim: int = 64, hidden_dim: int = 256, condition_dim: int = 32, max_len: int = 14):
        super().__init__()
        self.max_len = max_len
        self.pad_idx = 20
        self.vocab_size = 21  # 20 amino acids + 1 PAD
        self.input_dim = self.max_len * self.vocab_size  
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim


        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)
        )

        # Decoder — NO Sigmoid, raw logits for cross_entropy
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim)
            # No Sigmoid — raw logits fed to cross_entropy
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _current_device(self) -> torch.device:
        """Return the actual device currently used by model parameters."""
        return next(self.parameters()).device

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        h = self.encoder(torch.cat([x, y], dim=-1))
        mu, log_var = h.chunk(2, dim=-1)
        z = self._reparameterize(mu, log_var)
        x_recon = self.decoder(torch.cat([z, y], dim=-1))
        return x_recon, mu, log_var

    def generate_peptides(self, count, constraints=None, temperature=1.0):
        self.eval()
        device = self._current_device()
        self.device = device
        with torch.no_grad():
            z = torch.randn(count, self.latent_dim, device=device)
            y = self._constraints_to_condition_tensor(constraints, count, device=device)

            target_length = self._extract_target_length(constraints)
            sampled_lengths = torch.full((count,), target_length, device=device, dtype=torch.long)

            logits = self.decoder(torch.cat([z, y], dim=-1))
            logits = logits.view(count, self.max_len, self.vocab_size)

            aa_logits = logits[:, :, :20]
            probs = F.softmax(aa_logits / max(temperature, 1e-6), dim=-1)
            idx = torch.multinomial(probs.view(-1, 20), num_samples=1).view(count, self.max_len)

            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            peptides = []
            for i in range(count):
                L = int(sampled_lengths[i].item())
                peptides.append("".join(amino_acids[j] for j in idx[i, :L].cpu().tolist()))

        self.train()
        return peptides

    def modify_peptides(self, peptides: List[str], feedback: Optional[Any] = None) -> List[str]:
        return self.generate_peptides(len(peptides))

    def train_model(self, data, conditions, lengths, epochs=300, batch_size=64, lr=1e-3, kl_anneal_epochs=100):
        self.train()
        device = self._current_device()
        self.device = device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        dataset = TensorDataset(data, conditions, lengths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            kl_weight = min(1.0, epoch / max(1, kl_anneal_epochs))
            epoch_recon = 0.0
            epoch_kl = 0.0

            for batch in dataloader:
                x, y, lengths = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                x_recon, mu, log_var = self.forward(x, y)

                recon_logits = x_recon.view(-1, self.max_len, self.vocab_size)
                targets = x.view(-1, self.max_len, self.vocab_size).argmax(dim=-1)  # [B, L]

                per_tok = F.cross_entropy(
                    recon_logits.reshape(-1, self.vocab_size),
                    targets.reshape(-1),
                    reduction="none"
                ).view(-1, self.max_len)

                # mask out PAD positions
                pos = torch.arange(self.max_len, device=device).unsqueeze(0)    # [1, L]
                mask = (pos < lengths.unsqueeze(1)).float()                          # [B, L]
                recon_loss = (per_tok * mask).sum() / mask.sum().clamp_min(1.0)

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

    def _constraints_to_condition_tensor(
        self,
        constraints: Optional[Dict[str, Any]],
        count: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Convert constraints into a conditioning tensor [count, condition_dim].

        Behavior:
        - Starts from `constraints_default`
        - Applies user overrides as scalar values
        """
        device = device or self.device
        cond = torch.zeros(count, self.condition_dim, dtype=torch.float32, device=device)

        property_order = [
            "size",
            "molecular_weight",
            "net_charge_pH5_5",
            "isoelectric_point",
            "hydrophobicity",
            "cathionicity",
            "hydrophobic_moment",
            "logp",
            "boman_index",
            "h_bond_donors",
            "h_bond_acceptors",
            "tpsa",
        ]

        user_constraints: Dict[str, Any] = constraints or {}

        # 2) Start from defaults and override with user input
        encoded_values: List[float] = []
        for prop in property_order:
            default_value = float(constraints_default.get(prop, 0.0))

            spec = user_constraints.get(prop, None)

            if spec is None:
                value = default_value
            else:
                if isinstance(spec, dict):
                    raise ValueError(f"Constraint '{prop}' must be a scalar value, not a min/max dictionary.")
                try:
                    value = float(spec)
                except (TypeError, ValueError):
                    raise ValueError(f"Constraint '{prop}' must be numeric. Got: {spec!r}")

            encoded_values.append(value)

        vec = torch.tensor(encoded_values, dtype=torch.float32, device=device)
        usable = min(vec.numel(), self.condition_dim)
        cond[:, :usable] = vec[:usable].unsqueeze(0).expand(count, -1)

        return cond

    def _extract_target_length(self, constraints: Optional[Dict[str, Any]]) -> int:
        """Return fixed peptide length from constraints, clamped to [1, max_len]."""
        size_value = (constraints or {}).get("size", constraints_default["size"])
        if isinstance(size_value, dict):
            raise ValueError("Constraint 'size' must be a scalar value, not a min/max dictionary.")

        try:
            target = float(size_value)
        except (TypeError, ValueError):
            raise ValueError(f"Constraint 'size' must be numeric. Got: {size_value!r}")

        return max(1, min(int(round(target)), self.max_len))

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        device = self._current_device()
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        self.device = device