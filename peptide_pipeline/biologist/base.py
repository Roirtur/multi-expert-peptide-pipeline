from abc import ABC, abstractmethod
from typing import Any, List, Optional
import logging


class BaseBiologist(ABC):
    """
    Base class for the Biologist Agent.
    Responsible for scoring peptide activity and assessing biological properties.
    """

    logger = logging.getLogger("peptide_pipeline.biologist")


    @abstractmethod
    def score_peptides(self, peptides: List[str]) -> List[float]:
        """
        Assign a scalar score in [0, 1] to each peptide in the batch.
        """
        raise NotImplementedError("Subclasses must implement score_peptides.")

    @abstractmethod
    def predict_activity(
        self,
        peptides: List[str],
        context: Optional[Any] = None,
    ) -> List[float]:
        """
        Predict functional activity for a batch of peptides, optionally using
        contextual information (e.g. target description, reference sequence).
        """
        raise NotImplementedError("Subclasses must implement predict_activity.")