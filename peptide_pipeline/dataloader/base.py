from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class BaseDataLoader(ABC):
    """
    Base class for the Data Loader Agent.
    Responsible for loading and preprocessing peptide data for training or evaluation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def load_data(self, source: str, **kwargs) -> Any:
        """
        Loads data from a given source (file path, URL, etc.).
        """
        raise NotImplementedError("Subclasses must implement load_data method.")

    @abstractmethod
    def get_data(self) -> Any:
        """
        Returns the loaded and processed data.
        """
        raise NotImplementedError("Subclasses must implement get_data method.")
