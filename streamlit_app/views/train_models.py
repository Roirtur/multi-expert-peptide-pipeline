import streamlit as st
import pandas as pd
import json
import os
from pydantic import ValidationError
from pathlib import Path
import base64
from streamlit_app.utils import (
    highlight_error_card, parse_chemist_input, setup_streamlit_logger,
    render_chemist_form, get_available_models, instantiate_generator,
    instantiate_biologist, generators_config, biologists_config, config_data
)
from peptide_pipeline.orchestrator.orchestrator import Orchestrator
from peptide_pipeline.chemist.agent_v1.chemist_agent import ChemistAgent
from peptide_pipeline.chemist.agent_v1.config_chemist import ChemistConfig

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
        dataset_source = st.radio("Dataset Source", ["Workspace /database folder", "Upload File"], horizontal=True)
        
        uploaded_file = None
        dataset_path = None
        
        if dataset_source == "Workspace /database folder":
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "database")
            if os.path.exists(db_dir):
                available_jsons = [f for f in os.listdir(db_dir) if f.endswith('.json')]
                if available_jsons:
                    selected_json = st.selectbox("Select an existing dataset", available_jsons)
                    dataset_path = f"database/{selected_json}"
                else:
                    st.warning("No JSON files found in the /database folder.")
            else:
                st.error("No /database folder found.")
        else:
            uploaded_file = st.file_uploader("Upload an external JSON dataset", type=["json"])

    
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
        
        model_save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", selected_gen_train)
        os.makedirs(model_save_dir, exist_ok=True)
        final_save_path = os.path.join(model_save_dir, f"{model_name}.pth")
        
        st.subheader("Training Runtime Logs")
        with st.expander("Live Logs", expanded=True):
            log_container = st.empty()
            setup_streamlit_logger(log_container)
        
        # Instantiate base architecture
        with st.spinner("Initializing Model Structure..."):
            generator = instantiate_generator(selected_gen_train, model_file="Base (Untrained)")
        
        # Load data
        with st.spinner("Loading data..."):
            if dataset_source == "Upload File" and uploaded_file is not None:
                try:
                    df = pd.read_json(uploaded_file)
                except Exception:
                    uploaded_file.seek(0)
                    df = pd.read_json(uploaded_file, lines=True)
            elif dataset_path is not None:
                # dataset_path is like database/ai_training_peptides.json, which is at the root
                abs_dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), dataset_path)
                if not os.path.exists(abs_dataset_path):
                    st.error(f"Dataset not found at {abs_dataset_path}")
                    st.stop()
                else:
                    try:
                        df = pd.read_json(abs_dataset_path)
                    except Exception:
                        df = pd.read_json(abs_dataset_path, lines=True)
            else:
                st.error("No dataset selected or uploaded.")
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
                            from peptide_pipeline.generator.cvae_generator import constraints_default
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
                            st.success(f"Model successfully trained and saved to {final_save_path}")
                        except Exception as e:
                            st.error(f"Training Error: {e}")