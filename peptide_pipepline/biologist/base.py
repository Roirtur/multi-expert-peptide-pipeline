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
    def predict_activity(self, peptide: str, target_description: str) -> float:
        """
        Predicts the biological activity score of a peptide against a target.
        """
        raise NotImplementedError("Subclasses must implement predict_activity method.")

    @abstractmethod
    def assess_risks(self, peptide: str) -> Dict[str, float]:
        """
        Assesses risks such as toxicity, hemolysis, or immunogenicity.
        """
        raise NotImplementedError("Subclasses must implement assess_risks method.")
    
    @abstractmethod
    def analyze_peptide(self, peptide: str) -> Dict[str, Any]:
        """
        Provides a comprehensive biological analysis, combining activity and risks,
        and potentially offering improvement suggestions.
        """
        raise NotImplementedError("Subclasses must implement analyze_peptide method.")
