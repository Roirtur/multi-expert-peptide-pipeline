from textual.widgets import Label, Input, Select, Checkbox, Static
from textual.containers import Horizontal, Vertical
from .base import ExpertWidget
from dummy import models

VIRUS_BACTERIA_SOURCES = [
    "Escherichia coli", "Staphylococcus aureus", "Bacillus subtilis",
    "Saccharomyces cerevisiae", "Pseudomonas aeruginosa", "Salmonella enterica",
    "Influenza A virus", "SARS-CoV-2", "Bacteriophage T4", "Lactobacillus acidophilus"
]

class DataLoaderWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Dataset Path:")
        yield Input(placeholder="data/peptides.csv", id="loader-input")

class BiologistWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Peptide Sequence (Amino Acids):")
        yield Input(placeholder="e.g. ARNDCEQGHILK...", id="bio-sequence")
        yield Label("Source Search (Bacteria/Virus):")
        yield Input(placeholder="Start typing source name...", id="bio-source-search")
        yield Select([(s, s) for s in VIRUS_BACTERIA_SOURCES], prompt="Select Source", id="bio-source-select")

class ChemistWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Peptide Sequence to Check:")
        yield Input(placeholder="e.g. MLYK...", id="chem-sequence")
        yield Label("Properties to Check:")
        with Horizontal(classes="chem-checks"):
            yield Checkbox("Toxicity", id="check-tox")
            yield Checkbox("Solubility", id="check-sol")
            yield Checkbox("Hydrophobicity", id="check-hydro")

class GeneratorWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Model Type:")
        yield Select([(m.name, m.name) for m in models], prompt="Select Model", id="gen-model")
        with Vertical(id="gen-extra-params"):
            yield Static("Select a model to see parameters.")

class OrchestratorWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Target Model for Pipeline:")
        yield Select([(m.name, m.name) for m in models], prompt="Select Model", id="orch-model")
        yield Label("Global Parameters:")
        yield Input(placeholder="e.g. batch=32, epochs=10", id="orch-params")
