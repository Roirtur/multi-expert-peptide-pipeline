from __future__ import annotations

import os
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from peptide_pipeline.biologist.base import BaseBiologist

# Default model is small & fast
# swap for other variant for better accuracy

# facebook/esm2_t6_8M_UR50D (8 Million)
# facebook/esm2_t12_35M_UR50D (35 Million)
# facebook/esm2_t30_150M_UR50D (150 Million)
# facebook/esm2_t33_650M_UR50D (650 Million)
# facebook/esm2_t36_3B_UR50D (3 Billion)
# facebook/esm2_t48_15B_UR50D (15 Billion)

_DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"


class ESMBiologistCos(BaseBiologist):
    """
    Biologist agent that scores candidate peptides by ESM-2 embedding similarity
    to a given reference peptide.
    """

    def __init__(
        self,
        reference_peptide: str,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.reference_peptide = reference_peptide
        self.model_name = model_name
        self.batch_size = batch_size
        
        self.logger.debug(f"Initializing ESMBiologistCos with model '{model_name}' and reference peptide '{reference_peptide[:20]}...'")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        token = hf_token or os.environ.get("HF_TOKEN")

        # Load tokeniser and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModel.from_pretrained(model_name, token=token)
        self.model.eval()
        self.model.to(self.device)

        # Pre-compute the reference embedding once so repeated calls are cheap
        self._reference_embedding: torch.Tensor = self._embed_sequences(
            [reference_peptide]
        )[0]
        self.logger.info(f"ESMBiologistCos initialized on device '{self.device}' with model '{model_name}'")

    def _embed_sequences(self, sequences: List[str]) -> torch.Tensor:
        """
        Tokenise sequences and return mean-pooled ESM-2 embeddings.
        """
        all_embeddings: List[torch.Tensor] = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]

            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

            # last_hidden_state: (batch, seq_len, hidden_dim)
            hidden_states = outputs.last_hidden_state

            # attention_mask: 1 for real tokens, 0 for padding
            mask = encoded["attention_mask"]  # (batch, seq_len)

            # Mean pooling (expand mask to match hidden dimension)
            mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (batch, hidden_dim)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)      # (batch, 1)
            mean_pooled = sum_hidden / sum_mask                       # (batch, hidden_dim)

            all_embeddings.append(mean_pooled.cpu())

        return torch.cat(all_embeddings, dim=0)  # (N, hidden_dim)

    @staticmethod
    def _cosine_to_01(cosine_scores: torch.Tensor) -> List[float]:
        """
        Map cosine similarity values from [-1, 1] to [0, 1].
        """
        scaled = (cosine_scores + 1.0) / 2.0
        return scaled.tolist()

    def score_peptides(self, peptides: List[str]) -> List[float]:
        """
        Score each peptide against the stored reference peptide.
        """
        if not peptides:
            self.logger.warning("No peptides provided to score_peptides. Returning empty list.")
            return []

        candidate_embeddings = self._embed_sequences(peptides)  # (N, hidden_dim)

        ref = self._reference_embedding.unsqueeze(0)

        ref_expanded = ref.expand(candidate_embeddings.size(0), -1)
        cosine_scores = F.cosine_similarity(
            candidate_embeddings, ref_expanded, dim=1
        )

        return self._cosine_to_01(cosine_scores)

    def predict_activity(
        self,
        peptides: List[str],
        context: str = None,
    ) -> List[float]:
        """
        Runs the score_peptides function but with a given context.
        """
        if isinstance(context, str) and context.strip():
            original_ref = self._reference_embedding
            self._reference_embedding = self._embed_sequences([context])[0]
            scores = self.score_peptides(peptides)
            self._reference_embedding = original_ref
            return scores

        return self.score_peptides(peptides)

    def __repr__(self) -> str:
        return (
            f"ESMBiologistCos("
            f"model='{self.model_name}', "
            f"device='{self.device}', "
            f"reference='{self.reference_peptide[:20]}...'"
            f")"
        )
