"""
CVAE conditioning constraints (accepted in `constraints` dict):

Each property can be provided as either:
- scalar: `{"property": value}` (treated as min=max=value)
- range: `{"property": {"min": x, "max": y}}`

Supported properties:
- "size": peptide length (number of residues)
- "molecular_weight": molecular weight
- "net_charge_pH5_5": net charge at pH 5.5
- "isoelectric_point": isoelectric point (pI)
- "hydrophobicity": overall hydrophobicity
- "hydrophobic_moment": hydrophobic moment
- "logp": partition coefficient (logP)
- "boman_index": Boman index
- "h_bond_donors": hydrogen-bond donor count
- "h_bond_acceptors": hydrogen-bond acceptor count
- "tpsa": topological polar surface area

Notes:
- Properties are defaulted to 0.0 if not provided or if invalid.
"""

from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Dict, Any, Union
import numpy as np
from abc import ABC
from peptide_pipeline.generator.base import BaseGenerator

constraints_default = {
    "size": {"min": 5, "max": 30},
    "molecular_weight": {"min": 500, "max": 3000},
    "net_charge_pH5_5": {"min": -5, "max": 5},
    "isoelectric_point": {"min": 3, "max": 11},
    "hydrophobicity": {"min": -2, "max": 2},
    "hydrophobic_moment": {"min": 0, "max": 1},
    "logp": {"min": -3, "max": 3},
    "boman_index": {"min": -5, "max": 5},
    "h_bond_donors": {"min": 0, "max": 10},
    "h_bond_acceptors": {"min": 0, "max": 10},
    "tpsa": {"min": 0, "max": 200},
}

class CVAEGenerator(BaseGenerator):

    def __init__(self, input_dim: int = 120, latent_dim: int = 64, hidden_dim: int = 256, condition_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim


        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
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
            nn.Linear(hidden_dim, input_dim)
            # No Sigmoid — raw logits fed to cross_entropy
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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

    def generate_peptides(self, count: int, constraints: Optional[Dict[str, Any]] = constraints_default, temperature: float = 1.0) -> List[str]:
        self.eval()
        with torch.no_grad():
            z = torch.randn(count, self.latent_dim, device=self.device)
            y = self._constraints_to_condition_tensor(constraints, count, device=self.device)
            logits = self.decoder(torch.cat([z, y], dim=-1))
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
        return self.generate_peptides(len(peptides))

    def train_model(self, data: torch.Tensor, conditions: torch.Tensor, epochs: int = 300, batch_size: int = 64, lr: float = 1e-3, kl_anneal_epochs: int = 100) -> None:
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        dataset = TensorDataset(data, conditions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num_positions = self.input_dim // 20

        for epoch in range(epochs):
            kl_weight = min(1.0, epoch / max(1, kl_anneal_epochs))
            epoch_recon = 0.0
            epoch_kl = 0.0

            for batch in dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                x_recon, mu, log_var = self.forward(x, y)

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
                print(f"  Epoch {epoch+1:03d}/{epochs} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | KL weight: {kl_weight:.2f}")

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
        constraints: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
        count: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Convert constraints into a conditioning tensor [count, condition_dim].

        Behavior:
        - Starts from `constraints_default`
        - Applies user overrides (dict or list-of-dicts)
        - Missing properties keep default values
        """
        device = device or self.device
        cond = torch.zeros(count, self.condition_dim, dtype=torch.float32, device=device)

        property_order = [
            "size",
            "molecular_weight",
            "net_charge_pH5_5",
            "isoelectric_point",
            "hydrophobicity",
            "hydrophobic_moment",
            "logp",
            "boman_index",
            "h_bond_donors",
            "h_bond_acceptors",
            "tpsa",
        ]

        # 1) Normalize input into a dict
        user_constraints: Dict[str, Any] = {}
        if isinstance(constraints, dict):
            user_constraints = constraints
        elif isinstance(constraints, list):
            for row in constraints:
                if not isinstance(row, dict):
                    continue
                # row format: {"property": "size", "min": 8, "max": 12} or {"property": "size", "value": 10}
                if "property" in row:
                    prop = row.get("property")
                    if isinstance(prop, str):
                        if "value" in row:
                            user_constraints[prop] = row["value"]
                        else:
                            user_constraints[prop] = {
                                "min": row.get("min"),
                                "max": row.get("max"),
                            }
                else:
                    # row format: {"size": {"min": 8, "max": 12}} or {"size": 10}
                    user_constraints.update(row)

        # 2) Start from defaults and override with user input
        encoded_values: List[float] = []
        for prop in property_order:
            default_spec = constraints_default.get(prop, {"min": 0.0, "max": 0.0})
            default_min = float(default_spec.get("min", 0.0))
            default_max = float(default_spec.get("max", 0.0))

            spec = user_constraints.get(prop, None)

            if isinstance(spec, dict):
                min_v = default_min if spec.get("min") is None else float(spec.get("min"))
                max_v = default_max if spec.get("max") is None else float(spec.get("max"))
            elif spec is None:
                min_v, max_v = default_min, default_max
            else:
                try:
                    value = float(spec)
                    min_v, max_v = value, value
                except (TypeError, ValueError):
                    min_v, max_v = default_min, default_max

            encoded_values.extend([min_v, max_v])

        vec = torch.tensor(encoded_values, dtype=torch.float32, device=device)
        usable = min(vec.numel(), self.condition_dim)
        cond[:, :usable] = vec[:usable].unsqueeze(0).expand(count, -1)

        return cond

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)