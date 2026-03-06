import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Select, Input, Label, Static, Log, ProgressBar, TabbedContent, TabPane
from textual.containers import Vertical, Horizontal, Container
from textual.screen import Screen
from textual import on, work
from rich.panel import Panel
from rich.table import Table

# Simulation classes (to be replaced by actual implementations)
from dummy import Biologist, DataLoader, Generator, Orchestrator

class ExpertWidget(Static):
    """A modular component for each expert."""
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.expert_name = name

    def compose(self) -> ComposeResult:
        with Vertical(classes="expert-block"):
            yield Label(f"[bold]{self.expert_name}[/bold]", classes="expert-title")
            yield from self.get_input_fields()
            yield Button("Run Task", variant="primary", id=f"run-{self.expert_name.lower()}")
            yield ProgressBar(id=f"progress-{self.expert_name.lower()}", show_percentage=True)
            yield Log(id=f"log-{self.expert_name.lower()}", classes="expert-log")

    def get_input_fields(self) -> ComposeResult:
        """Override this to add specific inputs for each expert."""
        yield Static("No parameters required.")

class DataLoaderWidget(ExpertWidget):
    def get_input_fields(self) -> ComposeResult:
        yield Label("Dataset Path:")
        yield Input(placeholder="data/peptides.csv", id="loader-input")

class BiologistWidget(ExpertWidget):
    def get_input_fields(self) -> ComposeResult:
        yield Label("Fasta File:")
        yield Input(placeholder="input.fasta", id="bio-input")

class GeneratorWidget(ExpertWidget):
    def get_input_fields(self) -> ComposeResult:
        yield Label("Model Type:")
        yield Select([("CVAE", "CVAE"), ("VAE", "VAE"), ("GAN", "GAN")], id="gen-model")

class OrchestratorWidget(ExpertWidget):
    def get_input_fields(self) -> ComposeResult:
        yield Label("Pipeline Configuration:")
        yield Static("Full automated run combining all experts.")

class PeptideApp(App):
    """Modular Textual TUI with dedicated blocks for each expert."""
    CSS = """
    .expert-block {
        border: double $primary;
        padding: 1;
        margin: 1;
        height: 1fr;
    }
    .expert-title {
        background: $primary;
        color: $text;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }
    .expert-log {
        height: 10;
        border: solid $accent;
        margin-top: 1;
    }
    TabPane {
        padding: 1;
    }
    """

    TITLE = "Multi-Expert Peptide Pipeline"
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("DataLoader"):
                yield DataLoaderWidget("DataLoader")
            with TabPane("Biologist"):
                yield BiologistWidget("Biologist")
            with TabPane("Generator"):
                yield GeneratorWidget("Generator")
            with TabPane("Orchestrator"):
                yield OrchestratorWidget("Orchestrator")
            with TabPane("Global Log", id="global"):
                yield Log(id="main-global-log")
        yield Footer()

    @work(exclusive=True)
    async def run_task_execution(self, expert_name: str, task_func):
        # Dynamically find widgets for the expert
        name_lower = expert_name.lower()
        log = self.query_one(f"#log-{name_lower}", Log)
        progress = self.query_one(f"#progress-{name_lower}", ProgressBar)
        global_log = self.query_one("#main-global-log", Log)
        
        log.clear()
        log.write(f"Starting {expert_name}...")
        global_log.write(f"EXE: {expert_name} started.")
        progress.progress = 0
        
        # Simulate execution steps
        for i in range(1, 11):
            await asyncio.sleep(0.1)
            progress.advance(10)
            log.write(f"Processing step {i}/10...")
        
        # Explicit call to task logic
        result = task_func()
        log.write(f"Done! Result: {result}")
        global_log.write(f"SUCCESS: {expert_name} finished.")

    @on(Button.Pressed, "#run-dataloader")
    def on_load(self):
        val = self.query_one("#loader-input", Input).value or "default.csv"
        self.run_task_execution("DataLoader", lambda: DataLoader().load(val))

    @on(Button.Pressed, "#run-biologist")
    def on_bio(self):
        val = self.query_one("#bio-input", Input).value or "default.fasta"
        self.run_task_execution("Biologist", lambda: Biologist().process(val))

    @on(Button.Pressed, "#run-generator")
    def on_gen(self):
        self.run_task_execution("Generator", lambda: Generator().generate("params"))

    @on(Button.Pressed, "#run-orchestrator")
    def on_orch(self):
        self.run_task_execution("Orchestrator", lambda: Orchestrator().run_pipeline())

if __name__ == "__main__":
    PeptideApp().run()

if __name__ == "__main__":
    app = PeptideApp()
    app.run()
