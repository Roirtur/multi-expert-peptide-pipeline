from textual.widgets import Label, Select, Static
from textual.containers import Vertical
from .base import ExpertWidget, MODEL_CONFIGS


class OrchestratorWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Target Model for Pipeline:")
        yield Select([(name, name) for name in MODEL_CONFIGS.keys()], prompt="Select Model", id="orch-model")
        with Vertical(id="orch-extra-params"):
            yield Static("Select a model to see pipeline parameters.")
