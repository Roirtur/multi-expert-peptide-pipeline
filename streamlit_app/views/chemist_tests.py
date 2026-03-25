import streamlit as st
import pandas as pd
from pydantic import ValidationError
from pathlib import Path
import base64
from streamlit_app.utils import (
    highlight_error_card, parse_chemist_input, setup_streamlit_logger,
    render_chemist_form
)
from peptide_pipeline.chemist.chemist_agent.chemist_agent import ChemistAgent
from peptide_pipeline.chemist.chemist_agent.config_chemist import ChemistConfig

def render():
    icon_path = Path(__file__).resolve().parents[1] / "icons" / "chemistry.svg"
    icon_b64 = base64.b64encode(icon_path.read_bytes()).decode("utf-8")
    download_icon_path = Path(__file__).resolve().parents[1] / "icons" / "download.svg"
    download_icon_b64 = base64.b64encode(download_icon_path.read_bytes()).decode("utf-8")
    csv_icon_path = Path(__file__).resolve().parents[1] / "icons" / "csv.svg"
    csv_icon_b64 = base64.b64encode(csv_icon_path.read_bytes()).decode("utf-8")
    col_left, col_right = st.columns(2)

    st.markdown(
        f"""
        <div style="text-align:center; margin-top: 0.5rem; margin-bottom: 1rem;">
            <img src="data:image/svg+xml;base64,{icon_b64}" width="100" />
            <h1 style="margin: 0.25rem 0 0 0;">Chemist</h1>
            <p style="margin: 0.25rem 0 0 0;">Evaluate raw peptide sequences against physicochemical limits and score them.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Input")
    raw_peptides = st.text_area("Enter peptide sequences (comma-separated or one per line)", value="MLYK, ACDEFGH, RWRFR")
    
    st.header("Chemist Profile Constraints")
    chem_data, chem_errors = render_chemist_form(prefix="chem_test_")
    
    if st.button("Evaluate Properties", type="primary"):
        peplist = [p.strip() for p in raw_peptides.replace("\n", ",").split(",") if p.strip()]
        if not peplist:
            st.error("Please enter at least one valid sequence.")
        else:
            st.subheader("Chemist Evaluation Logs")
            with st.expander("Live Logs", expanded=True):
                log_container = st.empty()
                setup_streamlit_logger(log_container)
                
            with st.spinner("Evaluating chemist properties..."):
                try:
                    cleaned_chem = parse_chemist_input(chem_data)
                    chem_config = ChemistConfig(**cleaned_chem)
                    chemist_agent = ChemistAgent(chem_config)
                    
                    results = chemist_agent.evaluate_peptides(peplist)
                    
                    st.success(f"Evaluated {len(peplist)} valid sequences.")
                    if not results:
                        st.warning("No sequences matched the constraints or could be parsed.")
                    else:
                        st.subheader("Evaluation Results")
                        
                        # Prepare global data for CSV export
                        flattened_results = []
                        
                        for i, res in enumerate(results):
                            seq = res.get("sequence")
                            in_limits = res.get("in_limits")
                            total_score = res.get("score")
                            
                            # Prepare for individual table
                            pep_data = {
                                "Metric": ["Sequence", "In Limits", "Total Score"],
                                "Value": [seq, str(in_limits), f"{total_score:.4f}" if total_score is not None else "N/A"]
                            }
                            
                            flat_res = {"Sequence": seq, "In Limits": in_limits, "Total Score": total_score}
                            
                            if "properties" in res and "property_scores" in res:
                                for prop_name in res["properties"].keys():
                                    val = res["properties"].get(prop_name)
                                    score = res["property_scores"].get(prop_name)
                                    
                                    # For individual display
                                    pep_data["Metric"].append(f"Property: {prop_name.replace('_', ' ').title()}")
                                    pep_data["Value"].append(f"{val:.4f} (Score: {score:.4f})" if isinstance(val, (int, float)) and isinstance(score, (int, float)) else f"{val} (Score: {score})")
                                    
                                    # For global export
                                    flat_res[f"Val_{prop_name}"] = val
                                    flat_res[f"Score_{prop_name}"] = score
                                    
                            flattened_results.append(flat_res)
                            
                            # Display an individual table per peptide
                            status_icon = "✅" if in_limits else "❌"
                            with st.expander(f"{status_icon} Sequence #{i+1}: {seq} - Score: {total_score:.4f}", expanded=True):
                                df_pep = pd.DataFrame(pep_data)
                                st.dataframe(df_pep, hide_index=True, width="stretch")
    
                        # Global CSV export logic
                        df_global = pd.DataFrame(flattened_results)
                        csv = df_global.to_csv(index=False).encode('utf-8')
                        st.markdown(
                            f"""
                            <div style=\"display:flex; align-items:center; gap:0.5rem; margin: 0.5rem 0 0.6rem 0;\">
                                <img src=\"data:image/svg+xml;base64,{download_icon_b64}\" width=\"20\" />
                                <h3 style=\"margin:0;\">Export Results</h3>
                            </div>
                            <div style=\"display:flex; align-items:center; gap:0.4rem; margin-bottom:0.35rem;\">
                                <img src=\"data:image/svg+xml;base64,{csv_icon_b64}\" width=\"18\" />
                                <span style=\"font-weight:600;\">CSV</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.download_button(
                            label="Download ALL Results as CSV",
                            data=csv,
                            file_name='chemist_evaluation_results.csv',
                            mime='text/csv',
                            type="primary",
                            width="stretch"
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