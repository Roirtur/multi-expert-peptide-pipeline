#TODO : Delete dummy objects
#dummy objects to avoid cli errors
from enum import Enum
from pathlib import Path

def model_name_from_path(path):
    return Path(path).stem

import time

class Biologist:
    def process(self, data, progress_callback=None):
        for i in range(1, 11):
            time.sleep(0.2)
            if progress_callback:
                progress_callback(i * 10)
        return f"Biologist processed {data}"

class DataLoader:
    def load(self, path):
        return f"Loaded data from {path}"

class Generator:
    def generate(self, params, progress_callback=None):
        for i in range(1, 11):
            time.sleep(0.3)
            if progress_callback:
                progress_callback(i * 10)
        return ["Peptide1", "Peptide2", "Peptide3"]

class Orchestrator:
    def run_pipeline(self, progress_callback=None):
        for i in range(1, 11):
            time.sleep(0.5)
            if progress_callback:
                progress_callback(i * 10)
        return "Pipeline execution finished"
    
class Chemist:
    def analyze(self, sequence, progress_callback=None):
        for i in range(1, 101):
            time.sleep(0.1)
            if progress_callback:
                progress_callback(i * 10)
        return f"Chemist analyzed {sequence}"