from textual.widgets import Collapsible, Input, Label, Select, Static
from textual.containers import Vertical
from .base import ExpertWidget, MODEL_CONFIGS
from .biologist import VIRUS_BACTERIA_SOURCES
from ..handlers.chemist_form import is_optional, is_range_target
from peptide_pipeline.chemist.config_chemist import ChemistConfig


class OrchestratorWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Pipeline Composer")

        with Collapsible(title="Generator", collapsed=False):
            yield Label("Model Type:")
            yield Select([(name, name) for name in MODEL_CONFIGS.keys()], prompt="Select Model", id="pipe-gen-model")
            with Vertical(id="pipe-gen-extra-params"):
                yield Static("Select a model to see generator parameters.")

        with Collapsible(title="Chemist", collapsed=True):
            yield Label("Properties to Check:")

            for name, field in ChemistConfig.model_fields.items():
                if not is_range_target(field.annotation):
                    required = not is_optional(field.annotation)
                    default = field.default if field.default is not None else ""
                    label = f"{name} (required)" if required else name
                    yield Label(label)
                    yield Input(placeholder=str(default), id=f"pipe-chem-{name}")
                    continue

                yield Label(f"{name} (min, max, target):")
                yield Input(placeholder="min", id=f"pipe-chem-{name}-min")
                yield Input(placeholder="max", id=f"pipe-chem-{name}-max")
                yield Input(placeholder="target", id=f"pipe-chem-{name}-target")

        with Collapsible(title="Biologist", collapsed=True):
            yield Label("Peptide Sequence (Amino Acids):")
            yield Input(placeholder="e.g. ARNDCEQGHILK...", id="pipe-bio-sequence")
            yield Label("Source Search (Bacteria/Virus):")
            yield Input(placeholder="Start typing source name...", id="pipe-bio-source-search")
            yield Select([(s, s) for s in VIRUS_BACTERIA_SOURCES], prompt="Select Source", id="pipe-bio-source-select")
