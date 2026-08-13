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

## Reproduction

Environment (conda):

conda env create -f environment.yml
conda activate aiact


Reproduce the results by layer:
Layer 1: extraction + SHACL detection (RQ1)

python src/layer1_kg/evaluate_extraction.py
python src/layer1_kg/shacl_detection_benchmark.py

Dataset generation (seeded, reproducible)

python src/dataset/build_dataset.py
python src/dataset/add_requirement_labels.py

Layer 2: semantic verification (RQ5)

python src/layer2_semantic/evaluate.py

Layer 3: fusion comparison under structured noise (RQ3, RQ4)

python src/layer3_gnn/experiment_diffuse.py
python src/layer3_gnn/baseline.py

End-to-end pipeline

python src/pipeline.py


Results are written to `results/`. A local Qwen2.5-7B-Instruct model
(via Hugging Face Transformers) is required for Layers 1 and 2; no external
API is used. GPU recommended (developed on RTX 4090).

## Status
Complete. MSc dissertation submission, 2026.

## Licence
Code: MIT (see LICENSE). Data and documentation: CC-BY-4.0.
