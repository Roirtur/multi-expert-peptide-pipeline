import os
import sys
import tempfile
import pandas as pd
import torch
from dataloader import DataLoader
if __name__ == "__main__":
    loader = DataLoader()


    print("=== Default columns ===")
    loader.load_data("data/peptides.csv")
    print(loader.get_data().head())

    # Test 2: Custom columns
    print("\n=== Custom columns ===")
    loader.load_data("data/peptides.csv", columns=["NAME", "SEQUENCE", "TARGET GROUP"])
    print(loader.get_data().head())