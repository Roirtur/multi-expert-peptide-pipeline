import json
import os
import sys
import logging
from collections import deque
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pydantic import ValidationError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from peptide_pipeline.orchestrator.orchestrator_agent.orchestrator import Orchestrator
from peptide_pipeline.generator.cvae_generator_agent.cvae_generator import CVAEGenerator
from peptide_pipeline.generator.vae_generator_agent.vae_generator import VAEGenerator
from peptide_pipeline.biologist.esm_cos_bio_agent.esm_biologist_cos import ESMBiologistCos
from peptide_pipeline.biologist.esm_l2_bio_agent.esm_biologist_global_l2 import ESMBiologistGlobalL2
from peptide_pipeline.chemist.chemist_agent.chemist_agent import ChemistAgent
from peptide_pipeline.chemist.chemist_agent.config_chemist import ChemistConfig
from logger.logger import configure_logging
class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_placeholder, max_lines=500):
        super().__init__()
        self.log_placeholder = log_placeholder
        self.max_lines = max_lines
        self.logs = deque(maxlen=max_lines)

    def emit(self, record):
        try:
            log_entry = self.format(record)
            self.logs.append(log_entry)
            
            import html
            escaped_logs = html.escape("".join(self.logs))
            styled_html = f'''
            <div style="max-height: 400px; overflow-y: auto; background-color: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128,128,128,0.2); border-radius: 0.5rem; padding: 1rem;">
                <pre style="margin: 0; font-family: monospace; font-size: 0.85em; white-space: pre-wrap;">{escaped_logs}</pre>
            </div>
            '''
            self.log_placeholder.markdown(styled_html, unsafe_allow_html=True)
        except Exception:
            pass
# --- Helpers ---
def highlight_error_card(field_name):
    """Injects JS to highlight and expand a specific form expander if an error occurs."""
    title = f"Property: {field_name.replace('_', ' ').title()}"
    js = f"""
    <script>
    const expanders = window.parent.document.querySelectorAll('[data-testid="stExpander"]');
    expanders.forEach(exp => {{
        // Check if the expander contains the exact property title
        const summary = exp.querySelector('summary');
        if (summary && summary.innerText.includes("{title}")) {{
            exp.style.border = "2px solid #ff4b4b";
            exp.style.backgroundColor = "rgba(255, 75, 75, 0.05)";
            exp.style.borderRadius = "0.5rem";
            // Auto open the expander to show the error
            const details = exp.querySelector('details');
            if (details) details.open = true;
        }}
    }});
    </script>
    """
    components.html(js, height=0, width=0)
@st.cache_data
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)
config_data = load_config()
generators_config = config_data.get("generators", {})
biologists_config = config_data.get("biologists", {})
def parse_chemist_input(data: dict):
    '''Attempt to format fields properly to construct ChemistConfig'''
    cleaned_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            # Check if any field inside RangeTarget has a value
            if str(v["min"]).strip() or str(v["max"]).strip() or str(v["target"]).strip():
                rt = {}
                if str(v["min"]).strip(): rt["min"] = float(v["min"])
                if str(v["max"]).strip(): rt["max"] = float(v["max"])
                if str(v["target"]).strip(): rt["target"] = float(v["target"])
                cleaned_data[k] = rt
        else:
            if v and str(v).strip():
                try:
                    cleaned_data[k] = float(v) if "." in str(v) else int(v)
                except ValueError:
                    cleaned_data[k] = v
    return cleaned_data


def setup_streamlit_logger(log_placeholder):
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, StreamlitLogHandler):
            root_logger.removeHandler(handler)

    st_handler = StreamlitLogHandler(log_placeholder)
    st_handler.setLevel(getattr(logging, st.session_state.get("selected_log_level", "INFO")))
    st_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    
    # Filter out noisy external library logs
    class CleanFilter(logging.Filter):
        def filter(self, record):
            if record.name.startswith("tensorflow") or record.name.startswith("h5py"):
                return False
            return True
            
    st_handler.addFilter(CleanFilter())
    root_logger.addHandler(st_handler)
    
def render_chemist_form(prefix="chem_"):
    chem_config_data = {}
    chem_errors = {}

    fields = list(ChemistConfig.model_fields.items())

    # 1) Scalars first (full width), e.g. pH
    for name, field in fields:
        annotation_str = str(field.annotation)
        if "RangeTarget" not in annotation_str:
            default = field.default if field.default is not None else ""
            label = "pH" if name.lower() == "ph" else name.replace("_", " ").title()
            val = st.text_input(label, value=str(default), key=f"{prefix}scalar_{name}")
            chem_config_data[name] = val
            chem_errors[name] = st.empty()

    # 2) RangeTarget properties in two columns
    col_left, col_right = st.columns(2)
    range_idx = 0
    for name, field in fields:
        annotation_str = str(field.annotation)
        if "RangeTarget" in annotation_str:
            container = col_left if range_idx % 2 == 0 else col_right
            range_idx += 1

            with container:
                with st.expander(f"Property: {name.replace('_', ' ').title()}", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        min_val = st.text_input("min", key=f"{prefix}{name}_min")
                    with c2:
                        max_val = st.text_input("max", key=f"{prefix}{name}_max")
                    with c3:
                        target_val = st.text_input("target", key=f"{prefix}{name}_target")

                    chem_config_data[name] = {"min": min_val, "max": max_val, "target": target_val}
                    chem_errors[name] = st.empty()

    return chem_config_data, chem_errors

def get_available_models(model_type):
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", model_type)
    models = ["Base (Untrained)"]
    if os.path.exists(model_dir):
        models += [f for f in os.listdir(model_dir) if f.endswith('.pt') or f.endswith('.pth')]
    return models

def instantiate_generator(selected_gen, model_file="Base (Untrained)"):
    import torch
    gen_info = generators_config.get(selected_gen, {})
    hyperparams = {param["id"]: param["default"] for param in gen_info.get("hyperparameters", [])}

    if selected_gen == "CVAE":
        gen = CVAEGenerator(**hyperparams)
    else:
        gen = VAEGenerator(**hyperparams)

    if model_file and model_file != "Base (Untrained)":
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", selected_gen, model_file)
        if os.path.exists(model_path):
            try:
                gen.load_state_dict(torch.load(model_path, map_location=gen.device))
            except Exception as e:
                st.warning(f"Could not load weights from {model_file}: {e}")

    return gen

def instantiate_biologist(selected_bio, reference_sequence, bio_params_vals):
    if selected_bio == "ESMBiologistGlobalL2":
        return ESMBiologistGlobalL2(reference_peptide=reference_sequence, **bio_params_vals)
    else:
        return ESMBiologistCos(reference_peptide=reference_sequence, **bio_params_vals)

