import streamlit as st
import pandas as pd
from pathlib import Path
import base64

from streamlit_app.utils import (
    setup_streamlit_logger, get_available_models, instantiate_generator,
    generators_config
)

def render():
    icon_path = Path(__file__).resolve().parents[1] / "icons" / "generator.svg"
    icon_b64 = base64.b64encode(icon_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <div style="text-align:center; margin-top: 0.5rem; margin-bottom: 1rem;">
            <img src="data:image/svg+xml;base64,{icon_b64}" width="100" />
            <h1 style="margin: 0.25rem 0 0 0;">Generator</h1>
            <p style="margin: 0.25rem 0 0 0;">Test the generation models in isolation to see what sequences they produce without constraints filtering.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Setup Model")
    selected_gen_test = st.selectbox("Select Generator Model", list(generators_config.keys()))
    gen_info = generators_config[selected_gen_test]
    
    available_models = get_available_models(selected_gen_test)
    selected_model_file_test = st.selectbox(f"Load {selected_gen_test} Weights", available_models, key="test_gen_weights")
    
    st.header("Generation Params")
    
    col_params1, col_params2 = st.columns(2)
    with col_params1:
        gen_count = st.number_input("Number of peptides to generate", min_value=1, value=10)
    with col_params2:
        gen_temp = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)
    
    constraints = None
    if selected_gen_test == "CVAE":
        st.subheader("CVAE Conditioning Constraints")
        st.markdown("Configure the constraints vector to condition the CVAE generation.")
        try:
            from peptide_pipeline.generator.cvae_generator_agent.cvae_generator import constraints_default
            constraints = {}
            # Group constraints into 3 columns
            constraint_keys = list(constraints_default.keys())
            cols = st.columns(3)
            for idx, key in enumerate(constraint_keys):
                with cols[idx % 3]:
                    val = st.number_input(f"{key}", value=float(constraints_default[key]), key=f"cvae_{key}")
                    constraints[key] = val
        except ImportError:
            st.warning("Could not load CVAE constraints_default from cvae_generator.py")
    
    elif selected_gen_test == "VAE":
        # Usually VAE doesn't have conditioning constraints, but we leave the option open
        st.info("Standard VAE does not support conditioning constraints.")
        
    if st.button("Generate Peptides", type="primary"):
        st.subheader("Generation Runtime Logs")
        with st.expander("Live Logs", expanded=True):
            log_container = st.empty()
            setup_streamlit_logger(log_container)
            
        with st.spinner("Generating..."):
            generator_agent = instantiate_generator(selected_gen_test, selected_model_file_test)
                
            peptides = generator_agent.generate_peptides(
                count=int(gen_count),
                constraints=constraints,
                temperature=float(gen_temp)
            )
            
            st.success("Generation completed!")
            if peptides:
                if isinstance(peptides[0], str):
                    df_res = pd.DataFrame({"Generated Sequence": peptides})
                else:
                    df_res = pd.DataFrame(peptides)
                st.dataframe(df_res, width='stretch')
                
                csv_data = df_res.to_csv(index=False).encode('utf-8')
                json_data = df_res.to_json(orient="records", indent=4).encode('utf-8')
                
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv_data,
                        file_name="generated_peptides.csv",
                        mime="text/csv",
                        key="dl_gen_csv"
                    )
                with col_dl2:
                    st.download_button(
                        label="Download Results (JSON)",
                        data=json_data,
                        file_name="generated_peptides.json",
                        mime="application/json",
                        key="dl_gen_json"
                    )
                    
            else:
                st.warning("No peptides generated.")