# Automated Compliance Verification for EU AI Act Article 9

A three-layer neuro-symbolic framework for verifying EU AI Act Article 9
compliance evidence: a validated regulatory knowledge graph, LLM-based
semantic verification, and a noise-robust dual-shield graph neural network.

MSc Dissertation, Liverpool John Moores University.

## Structure
- `data/`         legal text, gold standard, validation sources, synthetic packages
- `src/`          implementation, organised by layer
- `shapes/`       SHACL constraint shapes (Layer 1)
- `prompts/`      LLM prompt templates (Layers 1 and 2)
- `experiments/`  experiment scripts and configs
- `results/`      metrics, tables, figures
- `notebooks/`    exploratory and demo notebooks
- `tests/`        unit and integration tests

## Status
In development.

## Licence
Code: Apache 2.0. Data and documentation: CC-BY-4.0.
