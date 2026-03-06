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