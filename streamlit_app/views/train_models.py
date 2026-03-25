import streamlit as st
import os
import sys
import subprocess
from pathlib import Path
import base64

from streamlit_app.utils import (
    setup_streamlit_logger,
    instantiate_generator,
    format_colorized_logs,
)

def render():
    icon_path = Path(__file__).resolve().parents[1] / "icons" / "train.svg"
    icon_b64 = base64.b64encode(icon_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <div style="text-align:center; margin-top: 0.5rem; margin-bottom: 1rem;">
            <img src="data:image/svg+xml;base64,{icon_b64}" width="100" />
            <h1 style="margin: 0.25rem 0 0 0;">Train Generator Models</h1>
            <p style="margin: 0.25rem 0 0 0;">Train a Generator (VAE or CVAE) from a backend dataset and save the weights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("1. Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        selected_gen_train = st.selectbox("Select Model Architecture to Train", ["VAE", "CVAE"], key="train_mod_sel")
        model_name = st.text_input("Model Save Name", value="my_trained_model", help="Will be saved in models/<ModelType>/<name>.pth")
    with col2:
        dataset_source = st.radio(
            "Dataset Source",
            [
                "Use JSON",
                "Upload JSON",
                "Fetch own dataset",
            ],
            horizontal=True,
        )
        project_root = Path(__file__).resolve().parents[2]
        db_dir = project_root / "database"
        db_fetch_script = db_dir / "get_data.py"
        
        uploaded_file = None
        dataset_path = None
        
        if dataset_source == "Use JSON":
            if db_dir.exists():
                available_jsons = [f for f in os.listdir(db_dir) if f.endswith('.json')]
                if available_jsons:
                    selected_json = st.selectbox("Select a dataset from /database", available_jsons)
                    dataset_path = f"database/{selected_json}"
                else:
                    st.warning("No JSON files found in the /database folder.")
            else:
                st.error("No /database folder found.")
        elif dataset_source == "Upload JSON":
            uploaded_file = st.file_uploader("Upload an external JSON dataset", type=["json"])
        else:
            st.caption("Run database/get_data.py to automatically download and build a dataset.")
            fetch_limit = st.number_input(
                "How many peptides to fetch",
                min_value=100,
                max_value=50000,
                value=5000,
                step=100,
                help="Passed to get_data.py as --limit",
            )
            dataset_name = st.text_input(
                "New dataset name",
                value=st.session_state.get("fetch_dataset_name", "my_fetched_dataset"),
                help="The file will be created in /database as <name>.json",
            )
            st.session_state["fetch_dataset_name"] = dataset_name

            target_file_name = None
            target_file_path = None
            invalid_name = False
            duplicate_name = False

            if not db_dir.exists():
                st.error("No /database folder found.")
            else:
                normalized_name = dataset_name.strip()
                if normalized_name.lower().endswith(".json"):
                    target_file_name = normalized_name
                else:
                    target_file_name = f"{normalized_name}.json"

                if not normalized_name:
                    invalid_name = True
                    st.warning("Please provide a dataset name.")
                elif any(ch in normalized_name for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']):
                    invalid_name = True
                    st.warning("Dataset name contains invalid filename characters.")
                else:
                    target_file_path = db_dir / target_file_name
                    if target_file_path.exists():
                        duplicate_name = True
                        dataset_path = f"database/{target_file_name}"
                        st.warning("This dataset name is already used. Pick another name to fetch a new dataset.")

            fetch_dataset = st.button(
                "Run Auto-Download Script",
                help="Run database/get_data.py and save output in /database.",
                disabled=(invalid_name or duplicate_name or not db_dir.exists()),
            )

            if fetch_dataset:
                if not db_fetch_script.exists():
                    st.error(f"Dataset fetch script not found: {db_fetch_script}")
                elif target_file_path is None:
                    st.error("Unable to resolve target dataset path from the provided name.")
                else:
                    st.subheader("📥 Dataset Fetch Logs")
                    st.markdown("""
                    <div style="background-color: #f0f2f6; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                        <p style="margin: 0; font-size: 0.9em; color: #666;">
                            🔄 Fetching from DBAASP API. Logs update in real-time below.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    status_box = st.empty()
                    logs_box = st.empty()
                    log_lines = []

                    status_box.info("Fetching dataset from DBAASP... This can take a while.")

                    try:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                str(db_fetch_script),
                                "--limit",
                                str(int(fetch_limit)),
                                "--output-file",
                                str(target_file_path),
                            ],
                            cwd=str(project_root),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                        )

                        if process.stdout is not None:
                            for line in process.stdout:
                                clean_line = line.rstrip()
                                if clean_line:
                                    log_lines.append(clean_line)
                                    # Keep UI responsive by showing only the latest lines.
                                    log_lines = log_lines[-250:]
                                    styled_html = format_colorized_logs(log_lines)
                                    logs_box.markdown(styled_html, unsafe_allow_html=True)

                        return_code = process.wait()
                    except Exception as e:
                        st.error(f"Failed to run dataset fetch script: {e}")
                        return_code = 1

                    if return_code == 0:
                        st.session_state["last_fetched_dataset"] = target_file_name
                        status_box.success(f"Dataset retrieved successfully: {target_file_name}")
                        st.rerun()
                    else:
                        status_box.error("Dataset retrieval failed while running database/get_data.py.")
                        if log_lines:
                            styled_html = format_colorized_logs(log_lines[-120:])
                            logs_box.markdown(styled_html, unsafe_allow_html=True)

            if dataset_path is None:
                last_fetched = st.session_state.get("last_fetched_dataset")
                if last_fetched:
                    last_fetched_path = db_dir / last_fetched
                    if last_fetched_path.exists():
                        dataset_path = f"database/{last_fetched}"
                        st.info(f"Current fetched dataset: {last_fetched}")
        

    
    st.header("2. Hyperparameters")
    col3, col4, col5 = st.columns(3)
    with col3:
        epochs = st.number_input("Epochs", min_value=1, value=50)
    with col4:
        batch_size = st.number_input("Batch Size", min_value=1, value=64)
    with col5:
        lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1.0, value=1e-3, format="%.5f")
        
    if st.button("Start Training", type="primary"):
        import torch
        import pandas as pd
        
        model_save_dir = os.path.join(str(project_root), "models", selected_gen_train)
        os.makedirs(model_save_dir, exist_ok=True)
        final_save_path = os.path.join(model_save_dir, f"{model_name}.pth")
        
        st.subheader("Training Runtime Logs")
        with st.expander("Live Training Logs", expanded=True):
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <p style="margin: 0; font-size: 0.9em; color: #666;">
                    Logs are updated in real-time below. Watch for INFO, WARNING, and completion messages.
                </p>
            </div>
            """, unsafe_allow_html=True)
            log_container = st.empty()
            setup_streamlit_logger(log_container)
        
        # Instantiate base architecture
        with st.spinner("Initializing Model Structure..."):
            generator = instantiate_generator(selected_gen_train, model_file="Base (Untrained)")
        
        # Load data
        with st.spinner("Loading data..."):
            if dataset_source == "Upload JSON" and uploaded_file is not None:
                try:
                    df = pd.read_json(uploaded_file)
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_json(uploaded_file, lines=True)
            elif dataset_path is not None:
                # dataset_path is like database/ai_training_peptides.json, which is at the root
                abs_dataset_path = os.path.join(str(project_root), dataset_path)
                if not os.path.exists(abs_dataset_path):
                    st.error(f"Dataset not found at {abs_dataset_path}")
                    st.stop()
                else:
                    try:
                        df = pd.read_json(abs_dataset_path)
                    except Exception:
                        df = pd.read_json(abs_dataset_path, lines=True)
            else:
                st.error("No dataset selected, uploaded, or downloaded.")
                st.stop()
                
            # Keep only standard aminos
            df = df[df["sequence"].str.fullmatch(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)]
            
            # Enforce max length
            max_len = getattr(generator, 'max_len', 14) if selected_gen_train == "CVAE" else (generator.input_dim // 20)
            df = df[(df["sequence"].str.len() <= max_len) & (df["sequence"].str.len() >= 5)]
            
            sequences = df["sequence"].tolist()
            
            if not sequences:
                st.error("No valid sequences found in dataset matching constraints (length <= max_len).")
            else:
                st.info(f"Filtered to {len(sequences)} valid sequences for training.")
            
                # Setup tensors
                with st.spinner("Preparing tensors..."):
                    if selected_gen_train == "VAE":
                        x_tensor = generator._peptides_to_one_hot(sequences)
                        
                    elif selected_gen_train == "CVAE":
                        VOCAB_SIZE = 21
                        PAD_IDX = '20s'
                        aa_index = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
                        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=generator.device)
                        
                        x_tensor = torch.zeros(len(sequences), max_len * VOCAB_SIZE)
                        for i, seq in enumerate(sequences):
                            for j in range(max_len):
                                x_tensor[i, j * VOCAB_SIZE + PAD_IDX] = 1.0
                            for j, aa in enumerate(seq[:max_len]):
                                if aa in aa_index:
                                    x_tensor[i, j * VOCAB_SIZE + PAD_IDX] = 0.0
                                    x_tensor[i, j * VOCAB_SIZE + aa_index[aa]] = 1.0
                        
                        try:
                            from peptide_pipeline.generator.cvae_generator_agent.cvae_generator import constraints_default
                            conditions_tensors = []
                            alias = {"length": "size", "net_charge": "net_charge_pH5_5"}
                            df_renamed = df.rename(columns=alias)
                            
                            for idx, row in df_renamed.iterrows():
                                row_dict = row.to_dict()
                                merged = {**constraints_default, **{k:v for k,v in row_dict.items() if k in constraints_default}}
                                c_tens = generator._constraints_to_condition_tensor(merged, count=1, device=generator.device)
                                conditions_tensors.append(c_tens)
                                
                            conditions = torch.cat(conditions_tensors, dim=0)
                        except Exception as e:
                            st.warning(f"Failed to extract row properties, using default constraints. Error: {e}")
                            conditions = generator._constraints_to_condition_tensor(constraints=constraints_default, count=len(sequences), device=generator.device)
    
                    with st.spinner("Training Model... (Please check your terminal for exact progress and metrics loss!)"):
                        try:
                            if selected_gen_train == "VAE":
                                generator.train_model(data=x_tensor, epochs=int(epochs), batch_size=int(batch_size), lr=float(lr))
                            else:
                                x_tensor = x_tensor.to(generator.device)
                                generator.train_model(data=x_tensor, conditions=conditions, lengths=lengths, epochs=int(epochs), batch_size=int(batch_size), lr=float(lr))
                                
                            torch.save(generator.state_dict(), final_save_path)
                            st.markdown("""
                            <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;">
                                <p style="margin: 0; color: #155724; font-weight: bold;">✅ Training Completed Successfully!</p>
                                <p style="margin: 0.5rem 0 0 0; color: #155724; font-size: 0.9em;">Model saved to: <code style="background-color: rgba(0,0,0,0.1); padding: 0.25rem 0.5rem; border-radius: 0.25rem;">{final_save_path}</code></p>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f"""
                            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;">
                                <p style="margin: 0; color: #721c24; font-weight: bold;">❌ Training Error</p>
                                <p style="margin: 0.5rem 0 0 0; color: #721c24; font-size: 0.9em;">{str(e)}</p>
                            </div>
                            """, unsafe_allow_html=True)