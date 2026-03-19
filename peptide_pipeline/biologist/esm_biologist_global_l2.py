from __future__ import annotations

import os
from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from peptide_pipeline.biologist.base import BaseBiologist

_DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"

class ESMBiologistGlobalL2(BaseBiologist):
    """
    Biologist agent that scores candidate peptides using a global,
    batch-independent exponential transform of Euclidean distance to a reference peptide.

    Score formula: score = exp(-||embedding - reference||_2 / temperature)
    This avoids per-batch normalization and makes scores comparable across runs.
    """

    def __init__(
        self,
        reference_peptide: str,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        batch_size: int = 32,
        score_temperature: float = 8.0,
    ) -> None:
        self.reference_peptide = reference_peptide
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_temperature = max(float(score_temperature), 1e-6)

        token = hf_token or os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModel.from_pretrained(model_name, token=token)
        self.model.eval()
        self.model.to(self.device)

        # Pre-compute reference embedding (Using Layer 6 for better biochemical signal)
        self._reference_embedding = self._embed_sequences([reference_peptide])[0]

    def _embed_sequences(self, sequences: List[str]) -> torch.Tensor:
        all_embeddings: List[torch.Tensor] = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)
            
            # Use Layer 6 for better peptide-level contrast
            # CAUTION : Changing the model would require to change this value
            # Note : you can take the middle layer of each model
            hidden_states = outputs.hidden_states[6] 
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            
            # Mean pooling
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = sum_hidden / sum_mask

            all_embeddings.append(mean_pooled.cpu())

        return torch.cat(all_embeddings, dim=0)

    def score_peptides(self, peptides: List[str]) -> List[float]:
        if not peptides:
            return []

        candidate_embeddings = self._embed_sequences(peptides)
        ref = self._reference_embedding.unsqueeze(0)

        # Absolute Euclidean distance to reference embedding.
        distances = torch.norm(candidate_embeddings - ref, p=2, dim=1)

        # Global, batch-independent mapping with tunable compression.
        scores = torch.exp(-distances / self.score_temperature)

        return scores.tolist()

    def predict_activity(self, peptides: List[str], context: str = None) -> List[float]:
        if isinstance(context, str) and context.strip():
            original_ref = self._reference_embedding
            self._reference_embedding = self._embed_sequences([context])[0]
            scores = self.score_peptides(peptides)
            self._reference_embedding = original_ref
            return scores
        return self.score_peptides(peptides)

    def __repr__(self) -> str:
        return (
            f"ESMBiologistGlobalL2(model='{self.model_name}', layer=6, "
            f"scoring='Exp(-L2/tau)', tau={self.score_temperature:.3f})"
        )