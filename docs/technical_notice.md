# Technical Notice

This document explains how the project is organized, how execution paths differ, and where to look first depending on your goal.

## 1) Big Picture

The repository contains two complementary tracks:

- Agentic pipeline track: modular experts and orchestrator in [peptide_pipeline](peptide_pipeline).
- Baseline track: non-agentic CVAE workflow in [baseline](baseline) for comparison experiments.

There is also a TUI interface in [tui](tui) to interact with components.

## 2) Main Entry Path

- Entrypoint: [main.py](main.py)
- Logging setup: [logger/logger.py](logger/logger.py)
- UI app: [tui/app.py](tui/app.py)

Running python main.py starts the Textual app and attaches global logging handlers.

## 3) Core Pipeline Modules

### Generator

- Base contract: [peptide_pipeline/generator/base.py](peptide_pipeline/generator/base.py)
- Implementations:
- [peptide_pipeline/generator/vae_generator.py](peptide_pipeline/generator/vae_generator.py)
- [peptide_pipeline/generator/cvae_generator.py](peptide_pipeline/generator/cvae_generator.py)

Role:

- Produces peptide candidates.
- Supports model training and weight save/load.
- CVAE variant supports scalar property conditioning.

### Chemist

- Base contract: [peptide_pipeline/chemist/base.py](peptide_pipeline/chemist/base.py)
- Main implementation: [peptide_pipeline/chemist/agent_v1/chemist_agent.py](peptide_pipeline/chemist/agent_v1/chemist_agent.py)
- Config model: [peptide_pipeline/chemist/agent_v1/config_chemist.py](peptide_pipeline/chemist/agent_v1/config_chemist.py)
- Property functions: [peptide_pipeline/chemist/agent_v1/properties.py](peptide_pipeline/chemist/agent_v1/properties.py)

Role:

- Validates sequences.
- Computes physicochemical properties.
- Scores candidates against range/target constraints.

### Biologist

- Base contract: [peptide_pipeline/biologist/base.py](peptide_pipeline/biologist/base.py)
- Implementations:
- [peptide_pipeline/biologist/esm_biologist_cos.py](peptide_pipeline/biologist/esm_biologist_cos.py)
- [peptide_pipeline/biologist/esm_biologist_global_l2.py](peptide_pipeline/biologist/esm_biologist_global_l2.py)

Role:

- Computes biological relevance proxies via ESM embeddings.
- Supports context-based scoring by temporarily switching reference embedding.

### Orchestrator

- Base contract: [peptide_pipeline/orchestrator/base.py](peptide_pipeline/orchestrator/base.py)
- Main loop: [peptide_pipeline/orchestrator/orchestrator.py](peptide_pipeline/orchestrator/orchestrator.py)

Role:

- Coordinates generator, chemist, and biologist.
- Runs iterative rounds.
- Balances exploration and exploitation.
- Returns final Top-K ranking.

### DataLoader

- Base contract: [peptide_pipeline/dataloader/base.py](peptide_pipeline/dataloader/base.py)
- CSV loader: [peptide_pipeline/dataloader/dataloader.py](peptide_pipeline/dataloader/dataloader.py)
- JSON/JSONL loader: [peptide_pipeline/dataloader/dataloader_json.py](peptide_pipeline/dataloader/dataloader_json.py)

Role:

- Loads and normalizes training/evaluation data.
- Applies optional schema checks and sequence cleanup.

## 4) Baseline Track

Files:

- Dataset and scaler pipeline: [baseline/data_handler.py](baseline/data_handler.py)
- CVAE model: [baseline/model.py](baseline/model.py)
- Training loop: [baseline/training.py](baseline/training.py)
- Generation/inference: [baseline/inference.py](baseline/inference.py)

Role:

- Provides a non-agentic reference path for quantitative comparison.

## 5) TUI Layer

Main app and event logic:

- [tui/app.py](tui/app.py)

Widgets and forms:

- [tui/widgets](tui/widgets)
- [tui/handlers](tui/handlers)

Important note:

- The TUI currently mixes real components and placeholder components from [dummy.py](dummy.py) for some actions.
- Chemist interactions are wired to the real ChemistAgent config and evaluation path.
- Some generator/biologist/orchestrator actions in the UI are still decoy/demo style.

## 6) Data and Experiments

- Dataset scripts and JSON files: [database](database)
- Experiment outputs and saved models: [experiments](experiments)
- Demonstration notebooks: [notebooks](notebooks)
- Example molecule output: [new_pep_files/ACDE.sdf](new_pep_files/ACDE.sdf)

## 7) Logging

- Logging configuration and custom NOTICE level: [logger/logger.py](logger/logger.py)
- Runtime logs: [logs](logs)

## 8) Module Technical Sheets

For module-specific contracts and extension guidance:

- [peptide_pipeline/dataloader/technical_sheet.md](peptide_pipeline/dataloader/technical_sheet.md)
- [peptide_pipeline/generator/technical_sheet.md](peptide_pipeline/generator/technical_sheet.md)
- [peptide_pipeline/chemist/technical_sheet.md](peptide_pipeline/chemist/technical_sheet.md)
- [peptide_pipeline/biologist/technical_sheet.md](peptide_pipeline/biologist/technical_sheet.md)
- [peptide_pipeline/orchestrator/technical_sheet.md](peptide_pipeline/orchestrator/technical_sheet.md)

## 9) Practical Navigation Guide

If you want to:

- Understand end-to-end orchestration logic: start with [peptide_pipeline/orchestrator/orchestrator.py](peptide_pipeline/orchestrator/orchestrator.py).
- Add or modify chemical constraints: start with [peptide_pipeline/chemist/agent_v1/config_chemist.py](peptide_pipeline/chemist/agent_v1/config_chemist.py) and [peptide_pipeline/chemist/agent_v1/properties.py](peptide_pipeline/chemist/agent_v1/properties.py).
- Improve biological scoring: start with [peptide_pipeline/biologist](peptide_pipeline/biologist).
- Work on generation models: start with [peptide_pipeline/generator](peptide_pipeline/generator) and compare with [baseline/model.py](baseline/model.py).
- Work on UI behavior and forms: start with [tui/app.py](tui/app.py), then [tui/widgets](tui/widgets) and [tui/handlers](tui/handlers).
