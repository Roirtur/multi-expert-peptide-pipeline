#TODO : Delete dummy objects
#dummy objects to avoid cli errors
from enum import Enum
from pathlib import Path

class models(Enum):
    CVAE = 1
    VAE = 2
    GAN = 3
    LLM = 4
    GNN = 5

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