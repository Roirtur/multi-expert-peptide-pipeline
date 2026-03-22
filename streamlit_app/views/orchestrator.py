import streamlit as st
import pandas as pd
import json
from pydantic import ValidationError

from streamlit_app.utils import (
    highlight_error_card, parse_chemist_input, setup_streamlit_logger,
    render_chemist_form, get_available_models, instantiate_generator,
    instantiate_biologist, generators_config, biologists_config
)
from peptide_pipeline.orchestrator.orchestrator_agent.orchestrator import Orchestrator
from peptide_pipeline.chemist.chemist_agent.chemist_agent import ChemistAgent
from peptide_pipeline.chemist.chemist_agent.config_chemist import ChemistConfig

def render():
    st.title("🧬 Multi-Expert Peptide Pipeline Orchestrator")
    st.markdown("Configure the pipeline once and run it. The properties are distributed to the Generator, Chemist, and Biologist below.")
    
    st.header("1. Global Parameters")
    reference_sequence = st.text_input(
        "Reference Peptide Sequence (Used by Biologist & Base Sequence)", 
        value="MLYK", 
        help="This sequence is used to score semantic similarity and as the base constraint."
    )
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.header("2. Generator Setup")
        selected_gen_orch = st.selectbox("Select Generator Model", list(generators_config.keys()))
        gen_info = generators_config[selected_gen_orch]
        st.caption(gen_info["description"])
        
        available_models = get_available_models(selected_gen_orch)
        selected_model_file_orch = st.selectbox(f"Load {selected_gen_orch} Weights", available_models, key="orch_gen_weights")
    
        orch_cvae_constraints = {}
        if selected_gen_orch == "CVAE":
            with st.expander("⚙️ CVAE Base Conditioning Constraints", expanded=True):
                st.markdown("Set default base generator conditions. Overridden by actual *Targets* in the Chemist tab if provided.")
                try:
                    from peptide_pipeline.generator.cvae_generator_agent.cvae_generator import constraints_default
                    cols_cvae = st.columns(2)
                    for idx, key in enumerate(constraints_default.keys()):
                        with cols_cvae[idx % 2]:
                            orch_cvae_constraints[key] = st.number_input(
                                key, value=float(constraints_default[key]), key=f"orch_cvae_{key}"
                            )
                except ImportError:
                    st.warning("Could not load CVAE constraints_default from cvae_generator.py")
    
        st.header("3. Biologist Setup")
        selected_bio_orch = st.selectbox("Select Biologist Model", list(biologists_config.keys()))
        bio_info = biologists_config[selected_bio_orch]
        st.caption(bio_info["description"])
    
        bio_params_vals = {}
        with st.expander(f"{selected_bio_orch} Parameters", expanded=False):
            for param in bio_info.get("params", []):
                if param["type"] == "int":
                    bio_params_vals[param["id"]] = st.number_input(
                        param["label"], value=int(param["default"]), step=1, key=f"orch_bio_{param['id']}"
                    )
                elif param["type"] == "str":
                    bio_params_vals[param["id"]] = st.text_input(
                        param["label"], value=str(param["default"]), key=f"orch_bio_{param['id']}"
                    )
    
    with col_right:
        st.header("4. Chemist Constraints")
        st.markdown("Set ranges and targets for chemical properties.")
        chem_data, chem_errors = render_chemist_form(prefix="orch_")
    
    st.divider()
    st.header("5. Execution")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        nb_iterations = st.number_input("Number of Iterations", min_value=1, value=5)
    with c2:
        nb_peptides = st.number_input("Peptides per Iteration", min_value=1, value=10)
    with c3:
        top_k = st.number_input("Top K Results", min_value=1, value=3)
    with c4:
        exploration_rate = st.slider("Exploration Rate", min_value=0.0, max_value=1.0, value=0.1)
    
    if st.button("Run Full Pipeline", type="primary"):
        if not reference_sequence:
            st.error("Reference Sequence is required.")
            st.stop()
            
        st.subheader("Runtime Logs")
        log_container = st.empty()
        setup_streamlit_logger(log_container)
        
        try:
            cleaned_chem = parse_chemist_input(chem_data)
            chem_config = ChemistConfig(**cleaned_chem)
            chemist_agent_instance = ChemistAgent(chem_config)
            
            # Explicitly extract the "target" dictionary to pass directly to orchestrator
            # Start with base Generator constraints, then override with explicit Chemist targets
            final_target_dict = {**orch_cvae_constraints}
            for k, v in cleaned_chem.items():
                if isinstance(v, dict) and "target" in v:
                    final_target_dict[k] = v["target"]
            
            generator_agent = instantiate_generator(selected_gen_orch, selected_model_file_orch)
            biologist_agent = instantiate_biologist(selected_bio_orch, reference_sequence, bio_params_vals)
            
            orchestrator = Orchestrator(
                generator=generator_agent,
                chemist=chemist_agent_instance,
                biologist=biologist_agent
            )
    
            with st.spinner(f"Running Orchestrator for {nb_iterations} iterations..."):
                top_peptides = orchestrator.run(
                    initial_peptide=reference_sequence,
                    nb_iterations=int(nb_iterations),
                    nb_peptides=int(nb_peptides),
                    top_k=int(top_k),
                    exploration_rate=float(exploration_rate),
                    final_target=final_target_dict
                )
            
            st.success("Pipeline sequence completed successfully!")
            
            if not top_peptides:
                st.warning("No peptides could be generated or evaluated.")
            else:
                st.subheader(f"Top {len(top_peptides)} Discovered Peptides")
                
                flattened_orchestrator = []
                for i, item in enumerate(top_peptides):
                    seq = item.get("peptide", "Unknown")
                    in_limits = item.get("in_limits", False)
                    combined_score = item.get("combined_score", 0.0)
                    chem_score = item.get("chemist_score", 0.0)
                    bio_score = item.get("biologist_score", 0.0)
                    iteration = item.get("iteration", 0)
                    props = item.get("properties", {})
                    
                    status_icon = "✅" if in_limits else "❌"
                    with st.expander(f"{status_icon} #{i+1} : {seq} (Combined Score: {combined_score:.4f})", expanded=True):
                        # Construct a clean summary table for the UI
                        pep_data = {
                            "Metric": ["Sequence", "In Limits", "Found in Iteration", "Combined Score", "Chemist Score", "Biologist Score"],
                            "Value": [seq, str(in_limits), str(iteration), f"{combined_score:.4f}", f"{chem_score:.4f}", f"{bio_score:.4f}"]
                        }
                        
                        flat_res = {
                            "Rank": i + 1,
                            "Sequence": seq,
                            "In Limits": in_limits,
                            "Iteration": iteration,
                            "Combined Score": combined_score,
                            "Chemist Score": chem_score,
                            "Biologist Score": bio_score
                        }
                        
                        if props:
                            for prop_name, val in props.items():
                                pep_data["Metric"].append(f"Property: {prop_name.replace('_', ' ').title()}")
                                pep_data["Value"].append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
                                flat_res[f"Val_{prop_name}"] = val
    
                        flattened_orchestrator.append(flat_res)
                        df_pep = pd.DataFrame(pep_data)
                        st.dataframe(df_pep, hide_index=True, width='stretch')
                        
                st.markdown("### 📥 Export Results")
                col_csv, col_json = st.columns(2)
                
                df_global = pd.DataFrame(flattened_orchestrator)
                with col_csv:
                    st.download_button(
                        label="📄 Download Results as CSV",
                        data=df_global.to_csv(index=False).encode('utf-8'),
                        file_name='orchestrator_results.csv',
                        mime='text/csv',
                        width='stretch'
                    )
                with col_json:
                    st.download_button(
                        label="📦 Download Results as JSON",
                        data=json.dumps(top_peptides, indent=4),
                        file_name='orchestrator_results.json',
                        mime='application/json',
                        width='stretch'
                    )
        except ValidationError as e:
            st.error("Chemist Configuration Layout Error: Check the red alerts above in the form.")
            for err in e.errors():
                loc = " -> ".join(map(str, err['loc']))
                field_name = str(err['loc'][0])
                if field_name in chem_errors:
                    chem_errors[field_name].error(f"**{loc}**: {err['msg']} (Input value: {err.get('input')})")
                    highlight_error_card(field_name)
                else:
                    st.warning(f"**{loc}**: {err['msg']} (Input value: {err.get('input')})")
        except Exception as e:
            st.error(f"Execution failed: {e}")