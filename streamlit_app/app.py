import sys
import os
import logging
import streamlit as st
from streamlit_option_menu import option_menu

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_app.utils import StreamlitLogHandler
from logger.logger import configure_logging
from streamlit_app.views import orchestrator, train_models, generator_tests, chemist_tests, biologist_tests

st.set_page_config(page_title="Multi-Expert Peptide Pipeline", layout="wide")

# Main header
st.title("Multi-Expert Peptide Pipeline")

# --- SIDEBAR NAVIGATION AND LOGGING ---
with st.sidebar:
    st.header("Navigation")
    selected_view = option_menu(
        menu_title=None,
        options=["Orchestrator", "Train Models", "Generator Tests", "Chemist Tests", "Biologist Tests"],
        default_index=0,
        icons=None,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"display": "none"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0 0 6px 0",
                "font-weight": "600",
                "--hover-color": "#ffcc80",
            },
            "nav-link-selected": {
                "background-color": "#ff9800",
                "color": "white",
                "box-shadow": "0 0 15px rgba(255, 152, 0, 0.5)",
            },
        },
    )

    st.divider()
    selected_log_level = st.selectbox(
        "Global Log Level",
        ["DEBUG", "INFO", "NOTICE", "WARNING", "ERROR"],
        index=1,
    )
    st.session_state["selected_log_level"] = selected_log_level
    configure_logging(level=selected_log_level, enable_file=False)

# --- LOG HANDLER CLEANUP ---
root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    if isinstance(h, StreamlitLogHandler):
        root_logger.removeHandler(h)

# --- VIEW ROUTING ---
if selected_view == "Orchestrator":
    orchestrator.render()
elif selected_view == "Train Models":
    train_models.render()
elif selected_view == "Generator Tests":
    generator_tests.render()
elif selected_view == "Chemist Tests":
    chemist_tests.render()
elif selected_view == "Biologist Tests":
    biologist_tests.render()