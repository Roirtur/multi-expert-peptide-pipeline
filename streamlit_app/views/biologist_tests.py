import streamlit as st
import pandas as pd
import json
import os
from pydantic import ValidationError

from streamlit_app.utils import (
    highlight_error_card, parse_chemist_input, setup_streamlit_logger,
    render_chemist_form, get_available_models, instantiate_generator,
    instantiate_biologist, generators_config, biologists_config, config_data
)
from peptide_pipeline.orchestrator.orchestrator import Orchestrator
from peptide_pipeline.chemist.agent_v1.chemist_agent import ChemistAgent
from peptide_pipeline.chemist.agent_v1.config_chemist import ChemistConfig

def render():
    st.title("🧫 Biologist Tests")
    st.markdown("Evaluate peptide activity using a biologist expert")
    
    st.header("Setup Reference")
    ref_seq = st.text_input("Reference Target Sequence", value="MLYK")
    
    selected_bio_test = st.selectbox("Select Biologist Model", list(biologists_config.keys()))
    bio_info = biologists_config[selected_bio_test]
    
    bio_params_vals = {}
    with st.expander(f"{selected_bio_test} Parameters", expanded=True):
        for param in bio_info.get("params", []):
            if param["type"] == "int":
                bio_params_vals[param["id"]] = st.number_input(
                    param["label"], value=int(param["default"]), step=1, key=f"bio_test_{param['id']}"
                )
            elif param["type"] == "str":
                bio_params_vals[param["id"]] = st.text_input(
                    param["label"], value=str(param["default"]), key=f"bio_test_{param['id']}"
                )
                
    st.header("Sequences to Test")
    test_peptides = st.text_area("Enter peptide sequences (comma-separated or one per line)", value="MLYR, ACDEFGH, MLYK, MLYQ")
    
    if st.button("Score Similarities", type="primary"):
        peplist = [p.strip() for p in test_peptides.replace("\n", ",").split(",") if p.strip()]
        if not peplist or not ref_seq:
            st.error("Please enter a valid reference sequence and at least one test sequence.")
        else:
            st.subheader("Biologist Inference Logs")
            with st.expander("Live Logs", expanded=True):
                log_container = st.empty()
                setup_streamlit_logger(log_container)
    
            with st.spinner("Downloading/Loading inference model and scoring..."):
                try:
                    biologist_agent = instantiate_biologist(selected_bio_test, ref_seq, bio_params_vals)
                    scores = biologist_agent.score_peptides(peplist)
                    
                    st.success("Scoring completed!")
                    
                    # Present as a dataframe
                    df = pd.DataFrame({
                        "Peptide Sequence": peplist,
                        "Similarity Score": scores
                    }).sort_values(by="Similarity Score", ascending=False).reset_index(drop=True)
                    
                    st.dataframe(df, width="stretch")
                except Exception as e:
                    st.error(f"Execution failed: {e}")