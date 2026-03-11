import json
from pathlib import Path
from textual.widgets import Static, Label, Button, ProgressBar, Log, DataTable, Checkbox, Select
from textual.containers import Vertical, VerticalScroll
from textual.app import ComposeResult

with Path(__file__).with_name("model_configs.json").open(encoding="utf-8") as f:
    MODEL_CONFIGS = json.load(f)["models"]

LOGGING_LEVELS = ["DEBUG", "INFO", "NOTICE", "WARNING", "ERROR", "CRITICAL"]

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
                yield Label("Logging Level:")
                yield Select([(name, name) for name in LOGGING_LEVELS], prompt="Select Logging Level", id="loglevel", value=LOGGING_LEVELS[2])
            progress = ProgressBar(id=f"progress-{self.expert_name.lower()}", show_percentage=True, show_eta=False)
            progress.display = False
            yield progress
            yield Button("Run Task", variant="primary", id=f"run-{self.expert_name.lower()}")
            yield Log(id=f"log-{self.expert_name.lower()}", classes="expert-log")
            yield DataTable(id=f"table-{self.expert_name.lower()}", classes="expert-table")

    def get_input_fields(self) -> ComposeResult:
        """Override this to add specific inputs for each expert."""
        yield Label("No parameters required.")
