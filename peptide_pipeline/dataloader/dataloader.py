import csv
import sys
import os
import pandas as pd
import torch
from typing import List, Any, Optional
from base import BaseDataLoader
import re
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
from logger import get_logger

logger = get_logger(__name__)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

class DataLoader(BaseDataLoader):
    """
    Data Loader Agent for peptide data.
    Responsible for loading and preprocessing peptide data for training or evaluation.
    """

    def __init__(self):
        super().__init__()
        self.data = None
        logger.info("DataLoader initialized.")

    def load_data(self, source: str, columns: Optional[List[str]] = None, **kwargs) -> None:
        """
        Loads data from a given source (file path, URL, etc.).
        Expects a CSV file containing columns 'SEQUENCE' and 'NAME'.
        If no columns are specified, defaults to ['NAME', 'SEQUENCE'].
        """
        try:
            self.data = pd.read_csv(source)
            if 'SEQUENCE' not in self.data.columns or 'NAME' not in self.data.columns:
                logger.error("CSV file must contain 'SEQUENCE' and 'NAME' columns.")
                raise ValueError("CSV file must contain 'SEQUENCE' and 'NAME' columns.")
            logger.info(f"Data loaded successfully from {source}. Total records: {len(self.data)}")
            if columns is None:
                logger.info("No columns specified, defaulting to ['NAME', 'SEQUENCE']")
                columns = ['NAME', 'SEQUENCE']
            
            missing = [col for col in columns if col not in self.data.columns]
            if missing:
                logger.error(f"Requested columns not found in CSV: {missing}")
                raise ValueError(f"Requested columns not found in CSV: {missing}")
            self.data = self.data[columns]

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)

    def get_data(self) -> pd.DataFrame:
        """
        Returns the loaded and processed data as a pandas DataFrame.
        """
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            sys.exit(1)
        logger.info("Data retrieval successful.")
        return self.data