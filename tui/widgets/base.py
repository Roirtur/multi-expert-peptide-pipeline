import json
from pathlib import Path
from textual.widgets import Static, Label, Button, ProgressBar, Log, DataTable, Checkbox
from textual.containers import Vertical, VerticalScroll
from textual.app import ComposeResult

with Path(__file__).with_name("model_configs.json").open(encoding="utf-8") as f:
    MODEL_CONFIGS = json.load(f)["models"]


class ExpertWidget(Static):
    """A modular component for each expert."""
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.expert_name = name

    def compose(self) -> ComposeResult:
        with Vertical(classes="expert-block"):
            yield Label(f"[bold]{self.expert_name}[/bold]", classes="expert-title")
            with VerticalScroll(id=f"input-container-{self.expert_name.lower()}", classes="input-container"):
                yield from self.get_input_fields()
                yield Checkbox("Verbose", id=f"verbose-{self.expert_name.lower()}")
            yield Button("Run Task", variant="primary", id=f"run-{self.expert_name.lower()}")
            yield ProgressBar(id=f"progress-{self.expert_name.lower()}", show_percentage=True)
            yield Log(id=f"log-{self.expert_name.lower()}", classes="expert-log")
            yield DataTable(id=f"table-{self.expert_name.lower()}", classes="expert-table")

    def get_input_fields(self) -> ComposeResult:
        """Override this to add specific inputs for each expert."""
        yield Label("No parameters required.")
