from textual.widgets import Label, Input, Select, Static
from textual.containers import Vertical
from .base import ExpertWidget, MODEL_CONFIGS

class GeneratorWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Model Type:")
        yield Select([(name, name) for name in MODEL_CONFIGS.keys()], prompt="Select Model", id="gen-model")
        with Vertical(id="gen-extra-params"):
            yield Static("Select a model to see parameters.")
