# Automated Compliance Verification for EU AI Act Article 9: A Three-Layer Neuro-Symbolic Framework

Code, data and results for the paper

> H. Vahedi Sarrigani and G. M. Lee, "Automated Compliance Verification for EU AI Act
> Article 9: A Three-Layer Neuro-Symbolic Framework", submitted to *Applied Sciences* (MDPI), 2026.

The framework verifies Article 9 (risk management) compliance evidence for healthcare AI:

- **Layer 1** extracts Article 9 into a knowledge graph with a local LLM and validates it with SHACL.
- **Layer 2** uses the same LLM to check that a mitigation's description fits its claimed
  risk-control level, and that a severity fits the described harm.
- **Layer 3** classifies each evidence package with a dual-shield graph neural network. It
  compares static, attention and the proposed noise-aware fusion, and defers uncertain cases
  to human review.

The work began as an MSc dissertation at Liverpool John Moores University.

## Authors
- Hoda Vahedi Sarrigani: School of Computer Science and Mathematics, Liverpool John Moores
  University, UK; UrbanChain Ltd, Manchester, UK
- Gyu Myoung Lee (corresponding author): School of Computer Science and Mathematics,
  Liverpool John Moores University, UK

## Repository structure
| Path | Contents |
|---|---|
| `data/article9/` | Verbatim text of Article 9, split by paragraph |
| `data/gold_standard/` | Gold-standard requirements (`requirements.csv`), the RDF knowledge graph (`article9_kg.ttl`), the LLM extraction output (`llm_extracted.json`) and its alignment to the gold standard (`llm_alignment.csv`) |
| `shapes/` | SHACL shapes, including the SHACL-SPARQL constraint |
| `prompts/` | Prompt templates for extraction (Layer 1) and the hierarchy and severity checks (Layer 2) |
| `src/` | Implementation, organised by layer (`layer1_kg`, `layer2_semantic`, `layer3_gnn`), the dataset generator (`dataset`), the end-to-end pipeline and the figure scripts (`figures`) |
| `results/paper/` | **The result files reported in the paper** |
| `results/earlier_implementation/` | Noisy Layer 3 runs from the earlier implementation of the noise weighting (see the note below) |
| `results/figures/` | Figures 1–5 as they appear in the paper |

## Where each result in the paper comes from
| Paper | Script | Result file (`results/paper/`) |
|---|---|---|
| Sec. 4.1, Tables 6–7 (extraction) | `src/layer1_kg/extract_llm.py`, `src/layer1_kg/evaluate_extraction.py` | `layer1_extraction_metrics.json` |
| Sec. 4.1, Table 8 (SHACL benchmark) | `src/layer1_kg/shacl_detection_benchmark.py` | `layer1_shacl_detection.json` |
| Sec. 4.2, Table 9 (semantic layer) | `src/layer2_semantic/evaluate.py` | `layer2_metrics.json` |
| Sec. 4.3, clean-data accuracy (0.993 ± 0.008) | `src/layer3_gnn/experiment_cleancv.py` (run on CPU with PyTorch 2.11 and PyTorch Geometric 2.8.1) | `layer3_cleancv.json` |
| Sec. 4.3, Table 10 "Structured" rows, Figure 3 | `src/layer3_gnn/experiment_diffuse.py` | `layer3_structured.json` |
| Sec. 4.3, Table 10 "Earlier run" rows | an earlier version of `src/layer3_gnn/experiment_diffuse.py` that used simple blanking and link breaking instead of the structured noise model (kept for the record) | `layer3_diffuse.json` |
| Sec. 4.3, single-channel baseline (uniform noise model) | `src/layer3_gnn/baseline.py` | `layer3_baseline.json` |
| Sec. 4.3, confidence and deferral | `src/layer3_gnn/confidence.py` (prints the counts) | – |
| Sec. 4.4, Table 11, Figure 4 | `src/layer3_gnn/experiment_degradation.py` | `layer3_degradation.json` |
| Sec. 4.5, Table 12, Figure 5 | `src/experiment_ablation.py` (Layer 2 runs the hierarchy and severity checks); the reference row is the accuracy of labelling all 120 test packages non-compliant (60 non-compliant + 32 manipulated) | `ablation.json` |
| Figures 1–2 | `src/figures/fig1_pipeline.py`, `src/figures/fig2_article9_graph.py` | – |
| Figures 3–5 | `src/figures/fig3_5_results.py` | reads `results/paper/` by default |

Experiment scripts write new runs to `results/`, so the files in `results/paper/` are never
overwritten. To plot a fresh run, use `python src/figures/fig3_5_results.py results`.

### Note on noise weighting in mini-batches (paper, Sec. 5)
`DualShieldGNN` gives each graph in a mini-batch its own measured noise, as described in the
paper (Sec. 3.4.2); the noisy Layer 3 results in `results/paper/` use this setting. An earlier
implementation used the measured noise of the first graph in each batch for the whole batch.
Its results are kept in `results/earlier_implementation/` and can be reproduced with
`AIACT_PER_GRAPH_NOISE=0`; noise-aware accuracy differs by at most 0.043, and no conclusion of
the paper changes. Runs without added noise (clean-data accuracy, confidence, ablation) are
unaffected.

## Reproduction

```bash
conda env create -f environment.yml      # Linux, CUDA 12.1
conda activate aiact
# or, with pip:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Layers 1 and 2 need a local Qwen2.5-7B-Instruct model, loaded through Hugging Face
Transformers. No external API is used. A GPU is recommended; the experiments were developed
on an RTX 4090 with CUDA 12.1.

```bash
# Synthetic dataset (seeded; regenerates the 600-package base set and labels)
python src/dataset/build_dataset.py
python src/dataset/add_requirement_labels.py
python src/dataset/build_diffuse_dataset.py      # 600-package diffuse-violation set

# Layer 1 (Sec. 4.1)
python src/layer1_kg/build_kg.py
python src/layer1_kg/extract_llm.py
python src/layer1_kg/evaluate_extraction.py
python src/layer1_kg/shacl_detection_benchmark.py

# Layer 2 (Sec. 4.2)
python src/layer2_semantic/evaluate.py

# Layer 3 (Secs. 4.3-4.4)
python src/layer3_gnn/experiment_cleancv.py
python src/layer3_gnn/experiment_diffuse.py
python src/layer3_gnn/baseline.py
python src/layer3_gnn/confidence.py
python src/layer3_gnn/experiment_degradation.py

# Ablation and end-to-end pipeline (Sec. 4.5)
python src/experiment_ablation.py
python src/pipeline.py

# Figures
python src/figures/fig1_pipeline.py
python src/figures/fig2_article9_graph.py
python src/figures/fig3_5_results.py
```

The synthetic packages (about 200 MB) are not stored in git. The generator is seeded; the
dataset used in the paper was built before the package order in `build_diffuse_dataset.py` was
made deterministic, so a regenerated diffuse-violation set is statistically equivalent but not
identical. Note that `build_kg.py` and `extract_llm.py` rewrite files in `data/gold_standard/`;
re-running LLM extraction samples new outputs, which will no longer match the manual
alignment in `llm_alignment.csv`.

## Citation
See `CITATION.cff`. Please cite the paper once it is published.

## Licence
Code: MIT (see `LICENSE`). Data, results and documentation: CC BY 4.0 (see `LICENSE-DATA`).
