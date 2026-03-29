# Multi-Expert Peptide Pipeline

Multi-Expert Peptide Pipeline is a proof-of-concept project for in silico peptide design with a multi-agent workflow.

The project combines:

- a Generator expert to propose peptide candidates,
- a Chemist expert to enforce physicochemical constraints,
- a Biologist expert to score biological relevance proxies,
- an Orchestrator to run iterative propose-filter-score loops and return Top-K candidates.

The main objective is to show measurable improvement versus a non-agentic baseline and to provide a modular, extensible framework for future multi-agent design projects.

## What We Built

- A modular peptide pipeline package with clear expert boundaries.
- A baseline CVAE workflow for non-agentic comparison.
- Streamlit interface for interactive pipeline execution, expert interaction, and visualization.
- Data utilities for collecting and preparing peptide datasets.
- Experiment notebooks and result artifacts for analysis.

## Project Structure

- Core pipeline package: [peptide_pipeline](peptide_pipeline)
- Baseline model workflow: [baseline](baseline)
- Streamlit interface: [streamlit](streamlit)
- Data collection and datasets: [database](database)
- Notebooks and experiment assets: [notebooks](notebooks), [experiments](experiments)
- Logging utilities: [logger](logger), [logs](logs)

## Quick Start

1. Install dependencies.

    ```bash
    pip install -r requirements.txt
    ```

2. Launch the Streamlit interface.

    ```bash
    python main.py
    ```

## Where To Start In Code

- Application entrypoint: [main.py](main.py)
- Streamlit interface: [streamlit/streamlit_app.py](streamlit/streamlit_app.py)
- Orchestration loop (core pipeline): [peptide_pipeline/orchestrator/orchestrator.py](peptide_pipeline/orchestrator/orchestrator.py)

## Documentation

Each expert module includes a technical sheet for implementation details and extension guidelines:

- [peptide_pipeline/doc/generator_technical_sheet.md](peptide_pipeline/doc/generator_technical_sheet.md)
- [peptide_pipeline/doc/chemist_technical_sheet.md](peptide_pipeline/doc/chemist_technical_sheet.md)
- [peptide_pipeline/doc/biologist_technical_sheet.md](peptide_pipeline/doc/biologist_technical_sheet.md)
- [peptide_pipeline/doc/dataloader_technical_sheet.md](peptide_pipeline/doc/dataloader_technical_sheet.md)
- [peptide_pipeline/doc/orchestrator_technical_sheet.md](peptide_pipeline/doc/orchestrator_technical_sheet.md)

For a full codebase map and practical navigation notes, see [technical_notice.md](technical_notice.md).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
