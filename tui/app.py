import asyncio
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Button, Select, Input, Log, TabbedContent, TabPane, DataTable, Checkbox
from textual import on, work
from .widgets.biologist import ( BiologistWidget, VIRUS_BACTERIA_SOURCES )
from .widgets.chemist import ChemistWidget
from .widgets.global_log import GlobalLogWidget
from .widgets.generator import GeneratorWidget, MODEL_CONFIGS
from .widgets.orchestrator import OrchestratorWidget
from .handlers.chemist_form import build_chemist_config_from_raw_inputs, is_range_target
from .handlers.logging_handler import attach_textual_log_handler
from .handlers.model_params import collect_model_params, get_model_fields, mount_model_fields
from dummy import Biologist, Generator
from peptide_pipeline.chemist import ChemistAgent
from peptide_pipeline.chemist.agent_v1.config_chemist import ChemistConfig

class PeptideApp(App):
    """Modular Textual TUI with dedicated blocks for each expert."""
    CSS = """
        .expert-block {
            border: double $primary;
            padding: 1;
            margin: 1;
            height: 1fr;
            overflow-y: auto;
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
        #gen-extra-params,
        #pipe-gen-extra-params {
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
        .range-target-row {
            layout: horizontal;
            padding: 1;
        }
        .range-target-row Input {
            width: 1fr;
            min-width: 10;
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

    def _collect_chemist_raw_inputs(self, root, scalar_prefix: str = "chem-", range_prefix: str = ""):
        raw_inputs = {}
        for name, field in ChemistConfig.model_fields.items():
            if is_range_target(field.annotation):
                raw_inputs[f"{name}-min"] = self._query_in_root(root, f"#{range_prefix}{name}-min", Input).value
                raw_inputs[f"{name}-max"] = self._query_in_root(root, f"#{range_prefix}{name}-max", Input).value
                raw_inputs[f"{name}-target"] = self._query_in_root(root, f"#{range_prefix}{name}-target", Input).value
            else:
                raw_inputs[f"chem-{name}"] = self._query_in_root(root, f"#{scalar_prefix}{name}", Input).value
        return raw_inputs

    def _run_decoy_orchestrator(
        self,
        generator_agent,
        chemist_agent,
        biologist_agent,
        payload,
    ):
        chemist_result = chemist_agent.analyze_peptide(payload["chemist_sequence"])
        return {
            "status": "decoy orchestrator complete",
            "experts": {
                "generator": type(generator_agent).__name__,
                "chemist": type(chemist_agent).__name__,
                "biologist": type(biologist_agent).__name__,
            },
            "generator": {
                "model": payload["generator_model"],
                "params": payload["generator_params"],
            },
            "biologist": {
                "sequence": payload["biologist_sequence"],
                "source": payload["biologist_source"],
            },
            "chemist": chemist_result,
        }

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

    @on(Input.Changed, "#bio-source-search")
    def filter_biologist_sources(self, event: Input.Changed):
        search_term = event.value.lower()
        filtered = [(s, s) for s in VIRUS_BACTERIA_SOURCES if search_term in s.lower()]
        select = self.query_one("#bio-source-select", Select)
        select.set_options(filtered if filtered else [(s, s) for s in VIRUS_BACTERIA_SOURCES])

    @on(Input.Changed, "#pipe-bio-source-search")
    def filter_pipeline_biologist_sources(self, event: Input.Changed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        search_term = event.value.lower()
        filtered = [(s, s) for s in VIRUS_BACTERIA_SOURCES if search_term in s.lower()]
        select = self._query_in_root(root, "#pipe-bio-source-select", Select)
        select.set_options(filtered if filtered else [(s, s) for s in VIRUS_BACTERIA_SOURCES])

    @on(Button.Pressed, "#run-orchestrator")
    async def on_orch(self, event: Button.Pressed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        generator_model = self._query_in_root(root, "#pipe-gen-model", Select).value
        generator_fields = get_model_fields(MODEL_CONFIGS, generator_model)
        generator_params = collect_model_params(root, generator_fields, self._query_in_root, id_prefix="pipe-gen-")

        biologist_sequence = self._query_in_root(root, "#pipe-bio-sequence", Input).value
        biologist_source = self._query_in_root(root, "#pipe-bio-source-select", Select).value
        chemist_sequence = self._query_in_root(root, "#pipe-chem-sequence", Input).value or biologist_sequence

        log_widget = self._query_in_root(root, "#log-orchestrator", Log)
        if not chemist_sequence:
            log_widget.write("ERROR: provide a sequence in Chemist or Biologist section.")
            return

        try:
            chemist_raw = self._collect_chemist_raw_inputs(
                root,
                scalar_prefix="pipe-chem-",
                range_prefix="pipe-chem-",
            )
            chemist_config = build_chemist_config_from_raw_inputs(chemist_raw)
        except Exception as exc:
            log_widget.write(f"ERROR: {exc}")
            return

        generator_agent = Generator()
        chemist_agent = ChemistAgent(chemist_config)
        biologist_agent = Biologist()

        payload = {
            "generator_model": generator_model,
            "generator_params": generator_params,
            "biologist_sequence": biologist_sequence,
            "biologist_source": biologist_source,
            "chemist_sequence": chemist_sequence,
        }

        self.run_task_execution(
            "Orchestrator",
            lambda: self._run_decoy_orchestrator(
                generator_agent,
                chemist_agent,
                biologist_agent,
                payload,
            ),
        )

    @on(Select.Changed, "#gen-model")
    async def update_generator_params(self, event: Select.Changed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        container = self._query_in_root(root, "#gen-extra-params", Vertical)
        await mount_model_fields(container, get_model_fields(MODEL_CONFIGS, event.value))

    @on(Select.Changed, "#pipe-gen-model")
    async def update_pipeline_generator_params(self, event: Select.Changed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        container = self._query_in_root(root, "#pipe-gen-extra-params", Vertical)
        await mount_model_fields(container, get_model_fields(MODEL_CONFIGS, event.value), id_prefix="pipe-gen-")

    @on(Button.Pressed, "#run-biologist")
    async def on_bio(self):
        seq = self.query_one("#bio-sequence", Input).value
        src = self.query_one("#bio-source-select", Select).value
            
        self.run_task_execution(
            "Biologist", 
            lambda: Biologist().process(f"Seq: {seq}, Src: {src}")
        )

    @on(Button.Pressed, "#run-chemist")
    async def on_chem(self, event: Button.Pressed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)

        # Retrieve the log level chosen in the form
        log_level_name = self._query_in_root(root, "#loglevel", Select).value or "INFO"

        seq = self._query_in_root(root, "#chem-sequence", Input).value
        log_widget = self._query_in_root(root, "#log-chemist", Log)

        try:
            raw_inputs = self._collect_chemist_raw_inputs(root)
            config = build_chemist_config_from_raw_inputs(raw_inputs)
        except Exception as exc:
            log_widget.write(f"ERROR: {exc}")
            return

        chem_agent = ChemistAgent(config)
        attach_textual_log_handler(chem_agent, log_widget, log_level_name)

        self.run_task_execution(
            "Chemist",
            lambda: chem_agent.analyze_peptide(seq),
        )

    @on(Button.Pressed, "#run-generator")
    async def on_gen(self, event: Button.Pressed):
        origin = self._get_event_origin(event)
        root = self._get_expert_root(origin)
        model = self._query_in_root(root, "#gen-model", Select).value
        params = collect_model_params(root, get_model_fields(MODEL_CONFIGS, model), self._query_in_root)

        self.run_task_execution(
            "Generator",
            lambda: Generator().generate({"model": model, "params": params}),
        )

    @work(exclusive=True)
    async def run_task_execution(self, expert_name: str, task_func):
        name_lower = expert_name.lower()
        log = self.query_one(f"#log-{name_lower}", Log)
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

        # Run the (possibly blocking) task in a background thread
        try:
            result = await asyncio.to_thread(task_func)
        except Exception as exc:
            log.write(f"ERROR: {exc}")
            global_log.write(f"ERROR: {expert_name} failed: {exc}")
            return

        await asyncio.sleep(0.5)
        log.write(f"Done! Result: {result}")
        global_log.write(f"SUCCESS: {expert_name} finished.")

        if isinstance(result, dict):
            table.display = True
            table.clear(columns=True)
            table.add_columns("Field", "Value")
            for key, value in result.items():
                table.add_row(str(key), str(value))
            return

        if isinstance(result, list):
            table.display = True
            table.clear(columns=True)
            table.add_columns("Index", "Data")
            for idx, item in enumerate(result, 1):
                table.add_row(str(idx), str(item))
