import asyncio
import logging
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Button, Select, Input, Log, ProgressBar, TabbedContent, TabPane, DataTable, Checkbox, Label
from textual import on, work
from .widgets.biologist import ( BiologistWidget, VIRUS_BACTERIA_SOURCES )
from .widgets.chemist import ChemistWidget
from .widgets.global_log import GlobalLogWidget
from .widgets.generator import GeneratorWidget, MODEL_CONFIGS
from .widgets.orchestrator import OrchestratorWidget
from dummy import Biologist, Generator



class TextualLogHandler(logging.Handler):
    def __init__(self, log_widget: Log):
        super().__init__()
        self.log_widget = log_widget

    def emit(self, record):
        # Format the message
        log_entry = self.format(record)
        # Use call_from_thread to ensure thread-safety if the AI runs in a worker
        self.log_widget.app.call_from_thread(self.log_widget.write, log_entry)

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
            height: 1fr;
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

    def _get_event_origin(self, event):
        for attr in ("button", "select", "control", "widget", "sender"):
            origin = getattr(event, attr, None)
            if origin is not None:
                return origin
        return None

    def _get_expert_root(self, origin):
        root = origin
        while root is not None and "expert-block" not in getattr(root, "classes", set()):
            root = getattr(root, "parent", None)
        return root

    def _query_in_root(self, root, selector, widget_type):
        if root is None:
            return self.query_one(selector, widget_type)
        return root.query_one(selector, widget_type)

    def _get_model_fields(self, model_name: str, include_hyperparameters: bool = False):
        cfg = MODEL_CONFIGS.get(model_name, {})
        fields = list(cfg.get("params", []))
        if include_hyperparameters:
            fields.extend(cfg.get("hyperparameters", []))
        return fields

    async def _mount_model_fields(self, container, fields):
        """Dynamically add input fields based on model configuration."""
        for child in list(container.children):
            await child.remove()

        for field in fields:
            field_type = field.get("type", "input")
            if field_type == "input":
                await container.mount(
                    Label(field["label"]),
                    Input(placeholder=field.get("placeholder", ""), id=field["id"]),
                )
            elif field_type == "select":
                await container.mount(
                    Label(field["label"]),
                    Select([(option, option) for option in field.get("options", [])], id=field["id"]),
                )
            elif field_type == "checkbox":
                await container.mount(Checkbox(field["label"], id=field["id"]))

    def _coerce_param_value(self, value, spec):
        coerce = spec.get("coerce")
        if coerce == "int":
            return int(value)
        if coerce == "float":
            return float(value)
        return value

    def _collect_model_params(self, root, fields):
        params = {}
        for field in fields:
            widget_id = f"#{field['id']}"
            field_type = field.get("type", "input")
            if field_type == "input":
                widget = self._query_in_root(root, widget_id, Input)
                value = widget.value or field.get("placeholder", "")
            elif field_type == "select":
                widget = self._query_in_root(root, widget_id, Select)
                value = widget.value
            elif field_type == "checkbox":
                widget = self._query_in_root(root, widget_id, Checkbox)
                value = widget.value
            else:
                continue
            params[field["id"]] = self._coerce_param_value(value, field)
        return params

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Generate Peptides"):
                yield OrchestratorWidget("Orchestrator")
            with TabPane("Chemist"):
                yield ChemistWidget("Chemist")
            with TabPane("Biologist"):
                yield BiologistWidget("Biologist")
            with TabPane("Generator"):
                yield GeneratorWidget("Generator")
            with TabPane("Global Log", id="global"):
                yield GlobalLogWidget(id="global-log-panel")
        yield Footer()

    def _get_chemist_agent(self):
        from peptide_pipeline.chemist.chemist_agent import ChemistAgent
        return ChemistAgent()

    @on(Input.Changed, "#bio-source-search")
    def filter_biologist_sources(self, event: Input.Changed):
        search_term = event.value.lower()
        filtered = [(s, s) for s in VIRUS_BACTERIA_SOURCES if search_term in s.lower()]
        select = self.query_one("#bio-source-select", Select)
        select.set_options(filtered if filtered else [(s, s) for s in VIRUS_BACTERIA_SOURCES])

    @on(Button.Pressed, "#run-orchestrator")
    async def on_orch(self, event: Button.Pressed):
        from peptide_pipeline.orchestrator.base import BaseOrchestrator
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        model = self._query_in_root(root, "#orch-model", Select).value
        fields = self._get_model_fields(model, include_hyperparameters=True)
        params = self._collect_model_params(root, fields)
        
        progress = self.query_one("#progress-orchestrator", ProgressBar)
        def progress_cb(val):
            self.call_from_thread(setattr, progress, "progress", val)

        self.run_task_execution(
            "Orchestrator",
            lambda: BaseOrchestrator().run(progress_callback=progress_cb)
        )

    @on(Select.Changed, "#gen-model")
    async def update_generator_params(self, event: Select.Changed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        container = self._query_in_root(root, "#gen-extra-params", Vertical)
        await self._mount_model_fields(container, self._get_model_fields(event.value))

    @on(Select.Changed, "#orch-model")
    async def update_orchestrator_params(self, event: Select.Changed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        container = self._query_in_root(root, "#orch-extra-params", Vertical)
        await self._mount_model_fields(
            container,
            self._get_model_fields(event.value, include_hyperparameters=True),
        )

    @on(Button.Pressed, "#run-biologist")
    async def on_bio(self):
        seq = self.query_one("#bio-sequence", Input).value
        src = self.query_one("#bio-source-select", Select).value
        progress = self.query_one("#progress-biologist", ProgressBar)
        def progress_cb(val):
            self.call_from_thread(setattr, progress, "progress", val)
            
        self.run_task_execution(
            "Biologist", 
            lambda: Biologist().process(f"Seq: {seq}, Src: {src}", progress_callback=progress_cb)
        )

    @on(Button.Pressed, "#run-chemist")
    async def on_chem(self, event: Button.Pressed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)

        # Retrieve the log level chosen in the form
        log_level_name = self._query_in_root(root, "#loglevel", Select).value or "INFO"

        seq = self.query_one("#chem-sequence", Input).value
        log_widget = self.query_one("#log-chemist", Log)

        chem_agent = self._get_chemist_agent()

        for handler in list(chem_agent.logger.handlers):
            if isinstance(handler, TextualLogHandler) and handler.log_widget is log_widget:
                chem_agent.logger.removeHandler(handler)

        # Attach the TUI handler so every logger.info/warning/etc. appears in the widget
        tui_handler = TextualLogHandler(log_widget)
        tui_handler.setLevel(getattr(logging, log_level_name.upper(), logging.INFO))
        tui_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        chem_agent.logger.addHandler(tui_handler)

        self.run_task_execution(
            "Chemist",
            lambda: chem_agent.check_validity([seq]),
        )

    @on(Button.Pressed, "#run-generator")
    async def on_gen(self, event: Button.Pressed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        model = self._query_in_root(root, "#gen-model", Select).value
        params = self._collect_model_params(root, self._get_model_fields(model))
        
        progress = self.query_one("#progress-generator", ProgressBar)
        def progress_cb(val):
            self.call_from_thread(setattr, progress, "progress", val)

        self.run_task_execution(
            "Generator",
            lambda: Generator().generate({"model": model, "params": params}, progress_callback=progress_cb),
        )

    @work(exclusive=True)
    async def run_task_execution(self, expert_name: str, task_func):
        name_lower = expert_name.lower()
        log = self.query_one(f"#log-{name_lower}", Log)
        progress = self.query_one(f"#progress-{name_lower}", ProgressBar)
        table = self.query_one(f"#table-{name_lower}", DataTable)
        global_log = self.query_one("#main-global-log", Log)
        verbose = False

        try:
            verbose = self.query_one(f"#verbose-{name_lower}", Checkbox).value
        except Exception:
            verbose = False

        log.clear()
        table.display = False
        log.write(f"Starting {expert_name}...")
        if verbose:
            log.write("Verbose mode enabled.")
        global_log.write(f"EXE: {expert_name} started.")
        progress.progress = 0
        progress.display = True

        # Run the (possibly blocking) task in a background thread
        try:
            result = await asyncio.to_thread(task_func)
        except Exception as exc:
            progress.display = False
            progress.progress = 0
            log.write(f"ERROR: {exc}")
            global_log.write(f"ERROR: {expert_name} failed: {exc}")
            return

        progress.progress = 100
        await asyncio.sleep(0.5)
        progress.display = False
        log.write(f"Done! Result: {result}")
        global_log.write(f"SUCCESS: {expert_name} finished.")

        if isinstance(result, list):
            table.display = True
            table.clear(columns=True)
            table.add_columns("Index", "Data")
            for idx, item in enumerate(result, 1):
                table.add_row(str(idx), str(item))
