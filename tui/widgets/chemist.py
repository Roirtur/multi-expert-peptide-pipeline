from textual.widgets import Label, Input
from textual.containers import Horizontal
from .base import ExpertWidget
from ..handlers.chemist_form import is_optional, is_range_target
from peptide_pipeline.chemist.agent_v1.config_chemist import ChemistConfig


class ChemistWidget(ExpertWidget):
    def get_input_fields(self):
        yield Label("Peptide Sequence to Check:")
        yield Input(placeholder="e.g. MLYK...", id="chem-sequence")

        yield Label("Properties to Check:")

        fields = ChemistConfig.model_fields
        for name, field in fields.items():
            # Scalar fields (float, int, etc.)
            if not is_range_target(field.annotation):
                required = not is_optional(field.annotation)
                default = field.default if field.default is not None else ""
                label = f"{name} (required)" if required else name
                yield Label(label)
                yield Input(placeholder=str(default), id=f"chem-{name}")
                continue

            # RangeTarget fields
            yield Label(f"{name} (min, max, target):")
            yield Horizontal(
                Input(placeholder="min", id=f"{name}-min"),
                Input(placeholder="max", id=f"{name}-max"),
                Input(placeholder="target", id=f"{name}-target"),
                classes="range-target-row",
            )
