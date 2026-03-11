from textual.widgets import Label, Input, Checkbox
from textual.containers import Horizontal
from .base import ExpertWidget

class ChemistWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Peptide Sequence to Check:")
        yield Input(placeholder="e.g. MLYK...", id="chem-sequence")
        yield Label("Properties to Check:")
        with Horizontal(classes="chem-checks"):
            yield Checkbox("Toxicity", id="check-tox")
            yield Checkbox("Solubility", id="check-sol")
            yield Checkbox("Hydrophobicity", id="check-hydro")
