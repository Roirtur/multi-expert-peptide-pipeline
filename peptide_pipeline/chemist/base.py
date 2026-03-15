from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

class BaseChemist(ABC):
    """
    Base class for the Chemist Agent (Constraints).
    Responsible for checking chemical validity and feasibility.
    """

    logger = logging.getLogger("peptide_pipeline.chemist")

    @abstractmethod
    def check_validity(self, peptides: List[str]) -> List[bool]:
        """
        Checks if a list of peptide sequences satisfy validity rules.
        """
        raise NotImplementedError("Subclasses must implement check_validity method.")

    @abstractmethod
    def calculate_properties(self, peptides: List[str]) -> List[Dict[str, float]]:
        """
        Calculates physicochemical properties for a list of peptides.
        """
        raise NotImplementedError("Subclasses must implement calculate_properties method.")

    def filter_peptides(self, peptides: List[str], constraints: Dict[str, Any]) -> List[str]:
        """
        Filters a list of peptides based on chemical constraints.
        """
        raise NotImplementedError("Subclasses must implement filter_peptides method.")

