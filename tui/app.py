import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Button, Select, Input, Log, ProgressBar, TabbedContent, TabPane, DataTable
from textual import on, work
from .widgets.experts import (
    DataLoaderWidget, BiologistWidget, ChemistWidget, 
    GeneratorWidget, OrchestratorWidget, VIRUS_BACTERIA_SOURCES
)
from dummy import Biologist, DataLoader, Generator, Orchestrator

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
    .input-container {
        height: auto;
        margin-bottom: 1;
    }
    .expert-log {
        height: 6;
        border: solid $accent;
        margin-top: 1;
    }
    .expert-table {
        height: 8;
        border: solid $secondary;
        display: none;
    }
    .chem-checks {
        height: auto;
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
            with TabPane("Chemist"):
                yield ChemistWidget("Chemist")
            with TabPane("Generator"):
                yield GeneratorWidget("Generator")
            with TabPane("Orchestrator"):
                yield OrchestratorWidget("Orchestrator")
            with TabPane("Global Log", id="global"):
                yield Log(id="main-global-log")
        yield Footer()

    @on(Input.Changed, "#bio-source-search")
    def filter_biologist_sources(self, event: Input.Changed):
        search_term = event.value.lower()
        filtered = [(s, s) for s in VIRUS_BACTERIA_SOURCES if search_term in s.lower()]
        select = self.query_one("#bio-source-select", Select)
        select.set_options(filtered if filtered else [(s, s) for s in VIRUS_BACTERIA_SOURCES])

    @on(Select.Changed, "#gen-model")
    async def update_generator_params(self, event: Select.Changed):
        from textual.widgets import Label, Static
        container = self.query_one("#gen-extra-params")
        await container.query("*").remove()
        
        if event.value == "CVAE":
            await container.mount(Label("Latent Dim:"), Input(placeholder="128", id="param-latent"))
        elif event.value == "GAN":
            await container.mount(Label("Noise Level:"), Input(placeholder="0.1", id="param-noise"))
        elif event.value == "LLM":
            await container.mount(Label("Temperature:"), Input(placeholder="0.7", id="param-temp"))

    @work(exclusive=True)
    async def run_task_execution(self, expert_name: str, task_func):
        name_lower = expert_name.lower()
        log = self.query_one(f"#log-{name_lower}", Log)
        progress = self.query_one(f"#progress-{name_lower}", ProgressBar)
        table = self.query_one(f"#table-{name_lower}", DataTable)
        global_log = self.query_one("#main-global-log", Log)
        
        log.clear()
        table.display = False
        log.write(f"Starting {expert_name}...")
        global_log.write(f"EXE: {expert_name} started.")
        progress.progress = 0
        
        for i in range(1, 6):
            await asyncio.sleep(0.1)
            progress.advance(20)
            log.write(f"Step {i}/5 in progress...")
        
        result = task_func()
        log.write(f"Done! Result: {result}")
        global_log.write(f"SUCCESS: {expert_name} finished.")

        if isinstance(result, list):
            table.display = True
            table.clear(columns=True)
            table.add_columns("Index", "Data")
            for idx, item in enumerate(result, 1):
                table.add_row(str(idx), str(item))

    @on(Button.Pressed, "#run-dataloader")
    def on_load(self):
        val = self.query_one("#loader-input", Input).value or "default.csv"
        self.run_task_execution("DataLoader", lambda: DataLoader().load(val))

    @on(Button.Pressed, "#run-biologist")
    def on_bio(self):
        seq = self.query_one("#bio-sequence", Input).value
        src = self.query_one("#bio-source-select", Select).value
        self.run_task_execution("Biologist", lambda: Biologist().process(f"Seq: {seq}, Src: {src}"))

    @on(Button.Pressed, "#run-chemist")
    def on_chem(self):
        seq = self.query_one("#chem-sequence", Input).value
        self.run_task_execution("Chemist", lambda: [f"Sequence: {seq}", "Toxicity: PASS", "Solubility: 0.85"])

    @on(Button.Pressed, "#run-generator")
    def on_gen(self):
        model = self.query_one("#gen-model", Select).value
        self.run_task_execution("Generator", lambda: Generator().generate(f"Model: {model}"))

    @on(Button.Pressed, "#run-orchestrator")
    def on_orch(self):
        model = self.query_one("#orch-model", Select).value
        self.run_task_execution("Orchestrator", lambda: Orchestrator().run_pipeline())

if __name__ == "__main__":
    PeptideApp().run()
