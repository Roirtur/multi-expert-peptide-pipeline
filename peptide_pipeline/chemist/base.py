from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import logging


class Chemist_output_payload(BaseModel):
    sequence: str
    properties: Dict[str, float]
    scores: float
    
class BaseChemist(ABC):
    """
    Base class for the Chemist Agent (Constraints).
    Responsible for checking chemical validity and feasibility.
    """
    basic_aa = {'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'}
    logger = logging.getLogger("peptide_pipeline.chemist")

    
    def __init__(self, Config: BaseModel):
        super().__init__()
        self.config = Config
    
    @abstractmethod
    def get_top_filtered_peptides(self, peptides: List[str], topK: int) -> List[str]:
        """
        Returns the top K peptides filtered if any filter
        """
        raise NotImplementedError("Subclasses must implement get_top_filtered_peptides method.")
    
    @abstractmethod
    def evaluate_peptides(self, peptides: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluates a list of peptides and calculates scores for each peptide.
        Returns a list of dictionaries, each containing property names as keys and their corresponding scores as values for each peptide.
        """
        raise NotImplementedError("Subclasses must implement evaluate_peptides method.")