from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

class BaseChemist(ABC):
    """
    Base class for the Chemist Agent (Constraints).
    Responsible for checking chemical validity and feasibility.
    """
    basic_aa = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'}
    logger = logging.getLogger("peptide_pipeline.chemist")

    
    @abstractmethod
    def calculate_score(self, peptides: List[str]) -> List[Dict[str, float]]:
        """
        Calculates physicochemical properties for a list of peptides.
        """
        raise NotImplementedError("Subclasses must implement calculate_score method.")

    @abstractmethod
    def filter_peptides(self, peptides: List[str], constraints: Dict[str, Any]) -> List[str]:
        """
        Filters a list of peptides based on chemical constraints.
            peptides: List of peptide sequences to filter.
            constraints: Dictionary of chemical constraints to apply.
            Returns a list of peptides that satisfy the constraints.
            In case to few peptides are returned after filtering a minimum of x% peptides will be returned based on their score (score is calculated by the target and limit disance)
        """
        raise NotImplementedError("Subclasses must implement filter_peptides method.")

    @abstractmethod
    def evaluate_peptide(self, peptide: str, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates a single peptide against the provided chemical constraints and calculates a score for each property.
        Returns a dictionary with property names as keys and their corresponding scores as values.
        Also returns a boolean indicating if the peptide is within all limits or not.
        """
        raise NotImplementedError("Subclasses must implement evaluate_peptide method.")