import sys
import os
import logging
import streamlit as st

st.set_page_config(page_title="Multi-Expert Peptide Pipeline", layout="wide")

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app.utils import StreamlitLogHandler
from logger.logger import configure_logging
from streamlit_app.views import orchestrator, train_models, generator_tests, chemist_tests, biologist_tests

st.sidebar.title("Multi-Expert Peptide Pipeline")

view = st.sidebar.radio("Select View", ["Orchestrator", "Train Models", "Generator Tests", "Chemist Tests", "Biologist Tests"])

st.sidebar.divider()
st.sidebar.subheader("Logging Settings")
selected_log_level = st.sidebar.selectbox("Global Log Level", ["DEBUG", "INFO", "NOTICE", "WARNING", "ERROR"], index=1)
st.session_state["selected_log_level"] = selected_log_level
configure_logging(level=selected_log_level, enable_file=False)

# On every Streamlit script rerun, aggressively strip out old UI log handlers   
root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    if isinstance(h, StreamlitLogHandler):
        root_logger.removeHandler(h)

if view == "Orchestrator":
    orchestrator.render()
elif view == "Train Models":
    train_models.render()
elif view == "Generator Tests":
    generator_tests.render()
elif view == "Chemist Tests":
    chemist_tests.render()
elif view == "Biologist Tests":
    biologist_tests.render()
