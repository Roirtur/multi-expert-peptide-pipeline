from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseGenerator(ABC):
    """
    Base class for the Designer Agent (Generator).
    Responsible for proposing candidate peptides.
    """

    def __init__(self):
        pass

    @abstractmethod
    def generate_peptides(self, count: int, constraints: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generates a list of peptide sequences.
        """
        raise NotImplementedError("Subclasses must implement generate_peptides method.")

    @abstractmethod
    def modify_peptides(self, peptides: List[str], feedback: Optional[Any] = None) -> List[str]:
        """
        Modifies or evolves existing peptides based on feedback or mutation operators.
        """
        raise NotImplementedError("Subclasses must implement modify_peptides method.")
