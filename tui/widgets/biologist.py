from textual.widgets import Label, Input, Select
from .base import ExpertWidget

VIRUS_BACTERIA_SOURCES = [
    "Escherichia coli", "Staphylococcus aureus", "Bacillus subtilis",
    "Saccharomyces cerevisiae", "Pseudomonas aeruginosa", "Salmonella enterica",
    "Influenza A virus", "SARS-CoV-2", "Bacteriophage T4", "Lactobacillus acidophilus"
]

class BiologistWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Peptide Sequence (Amino Acids):")
        yield Input(placeholder="e.g. ARNDCEQGHILK...", id="bio-sequence")
        yield Label("Source Search (Bacteria/Virus):")
        yield Input(placeholder="Start typing source name...", id="bio-source-search")
        yield Select([(s, s) for s in VIRUS_BACTERIA_SOURCES], prompt="Select Source", id="bio-source-select")
