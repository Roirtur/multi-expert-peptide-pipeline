from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseBiologist(ABC):
    """
    Base class for the Biologist Agent (Function/Risk).
    Responsible for scoring activity and assessing biological risks (e.g., toxicity).
    """

    def __init__(self):
        pass

    @abstractmethod
    def predict_activity(self, peptides: List[str], target_description: str) -> List[float]:
        """
        Predicts the biological activity scores for a batch of peptides.
        """
        raise NotImplementedError("Subclasses must implement predict_activity method.")

    @abstractmethod
    def assess_risks(self, peptides: List[str]) -> List[Dict[str, float]]:
        """
        Assesses risks for a batch of peptides.
        """
        raise NotImplementedError("Subclasses must implement assess_risks method.")
    
    @abstractmethod
    def analyze_peptides(self, peptides: List[str]) -> List[Dict[str, Any]]:
        """
        Provides comprehensive biological analysis for a batch of peptides.
        """
        raise NotImplementedError("Subclasses must implement analyze_peptides method.")
