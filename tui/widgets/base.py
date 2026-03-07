from textual.widgets import Static, Vertical, Label, Button, ProgressBar, Log, DataTable
from textual.app import ComposeResult

class ExpertWidget(Static):
    """A modular component for each expert."""
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.expert_name = name

    def compose(self) -> ComposeResult:
        with Vertical(classes="expert-block"):
            yield Label(f"[bold]{self.expert_name}[/bold]", classes="expert-title")
            with Vertical(id=f"input-container-{self.expert_name.lower()}", classes="input-container"):
                yield from self.get_input_fields()
            yield Button("Run Task", variant="primary", id=f"run-{self.expert_name.lower()}")
            yield ProgressBar(id=f"progress-{self.expert_name.lower()}", show_percentage=True)
            yield Log(id=f"log-{self.expert_name.lower()}", classes="expert-log")
            yield DataTable(id=f"table-{self.expert_name.lower()}", classes="expert-table")

    def get_input_fields(self) -> ComposeResult:
        """Override this to add specific inputs for each expert."""
        yield Label("No parameters required.")
