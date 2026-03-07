#TODO : Delete dummy objects
#dummy objects to avoid cli errors
from enum import Enum
from pathlib import Path

def model_name_from_path(path):
    return Path(path).stem

class Biologist:
    def process(self, data):
        return f"Biologist processed {data}"

class DataLoader:
    def load(self, path):
        return f"Loaded data from {path}"

class Generator:
    def generate(self, params):
        return ["Peptide1", "Peptide2", "Peptide3"]

class Orchestrator:
    def run_pipeline(self):
        return "Pipeline execution finished"
    
class Chemist:
    def analyze(self, sequence):
        return f"Chemist analyzed {sequence}"