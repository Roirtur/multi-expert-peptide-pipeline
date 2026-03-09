from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch.nn as nn
import torch

class BaseGenerator(nn.Module, ABC):
    """
    Base class for the Designer Agent (Generator).
    Responsible for proposing candidate peptides.
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

    @abstractmethod
    def train_model(self, data: Any, **kwargs) -> None:
        """
        Trains the generator model on the provided data.
        We might want to change kwargs argument to something like epoch, lr, output_dir...
        """
        raise NotImplementedError("Subclasses must implement train method.")
