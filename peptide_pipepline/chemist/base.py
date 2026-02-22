from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChemist(ABC):
    """
    Base class for the Chemist Agent (Constraints).
    Responsible for checking chemical validity and feasibility.
    """

    def __init__(self):
        pass

    @abstractmethod
    def check_validity(self, peptide: str) -> bool:
        """
        Checks if a single peptide sequence constitutes a valid molecule 
        according to defined chemical rules.
        """
        raise NotImplementedError("Subclasses must implement check_validity method.")

    @abstractmethod
    def calculate_properties(self, peptide: str) -> Dict[str, float]:
        """
        Calculates physicochemical properties (e.g., charge, hydrophobicity, solubility proxy).
        """
        raise NotImplementedError("Subclasses must implement calculate_properties method.")

    def filter_peptides(self, peptides: List[str], constraints: Dict[str, Any]) -> List[str]:
        """
        Filters a list of peptides based on chemical constraints.
        Default implementation iterates using check_validity, but can be overridden.
        """
        valid_peptides = []
        for p in peptides:
            if self.check_validity(p):
                # Placeholder for more complex constraint checking logic
                valid_peptides.append(p)
        return valid_peptides
