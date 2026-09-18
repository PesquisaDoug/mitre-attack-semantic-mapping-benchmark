# Transformer-Based Semantic Mapping of CTI to MITRE ATT&CK

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22821273.svg)](https://doi.org/10.5281/zenodo.22821273)

Reproducible retrieval and reranking benchmark for mapping cyber threat
intelligence procedure descriptions to MITRE ATT&CK Enterprise techniques and
sub-techniques.

## Overview

This repository evaluates zero-shot retrieval methods for ranking ATT&CK
techniques given ATT&CK-derived procedure descriptions. The benchmark compares
lexical baselines with pretrained Transformer retrieval models:

- TF-IDF cosine similarity
- BM25 using `rank_bm25.BM25Okapi`
- `sentence-transformers/all-MiniLM-L6-v2` bi-encoder retrieval
- MiniLM bi-encoder retrieval followed by
  `cross-encoder/ms-marco-MiniLM-L6-v2` reranking over the top-20 candidates

The full experiment reports both an `all` query subset and a `strict` subset.
The strict subset excludes queries containing an exact relevant ATT&CK ID or
exact relevant technique-name occurrence.

## Repository Structure

```text
app/                         Streamlit research demonstrator
configs/                     Experiment and model configuration files
data/                        Local ATT&CK data and processed benchmark files
notebooks/                   Preserved reference notebook
paper/                       IEEE manuscript, references, and paper figures
results/                     Metrics, manifests, rankings, and analysis outputs
scripts/                     Pipeline entry points
src/                         Modular implementation
tests/                       Unit and smoke tests
```

## Requirements

The full run recorded the following environment:

- Intel Core i7-14700K CPU (20 physical cores, 28 logical threads), 32 GiB RAM
- Python 3.12.0
- Windows 11
- PyTorch 2.14.0+cpu
- Transformers 5.17.0
- Sentence-Transformers 6.0.1
- NumPy 2.5.3
- pandas 3.0.5
- SciPy 1.18.1
- scikit-learn 1.9.1

## Installation

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip check
```

## Dataset / Data Availability

The benchmark is constructed from the official MITRE ATT&CK Enterprise STIX 2.1
bundle:

```text
https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json
```

The validated full run used a local bundle with SHA-256:

```text
dc1639caa5501d720e280cf1cbd8fbe009884a0c9b3e6e9ed9d0c25166c3d8f4
```

Full benchmark counts:

- STIX objects: 26,086
- Candidate techniques/sub-techniques: 697
- All procedure queries: 17,030
- Strict procedure queries: 14,868
- Exact target-ID leakage queries: 0
- Exact target-name leakage queries: 2,162
- Multi-relevant queries: 91

MITRE ATT&CK data are governed by MITRE's official terms of use:

```text
https://attack.mitre.org/resources/terms-of-use/
```

## Running the Pipeline

Quick profile:

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m scripts.run_retrieval --config configs/experiment.yaml --profile quick
.venv\Scripts\python -m scripts.run_reranking --config configs/experiment.yaml --profile quick
.venv\Scripts\python -m scripts.generate_results --config configs/experiment.yaml --profile quick
```

Streamlit demonstrator:

```powershell
.venv\Scripts\streamlit run app/streamlit_app.py
```

## Full Experimental Execution

**Important:** the `quick` profile (`max_queries: 1000`, used for fast
smoke-testing of the pipeline) and the `full` profile (`max_queries: null`,
the profile whose results are reported in the manuscript) produce separate,
clearly named artifacts (e.g. `retrieval_rankings_quick.json` vs.
`retrieval_rankings_full.json`). Do not cite `*_quick.json` outputs or
metrics generated with `--profile quick` as manuscript results. The
manuscript, `results/metrics.csv`, `results/experiment_manifest.json`, and
`results/full_run_summary.json` all correspond to `--profile full`
(`experiment_manifest.json` records `"profile": "full"` and
`"quick_mode": false`).

The complete experiment was executed with:

```powershell
.venv\Scripts\python -m scripts.run_retrieval --config configs/experiment.yaml --profile full
.venv\Scripts\python -m scripts.run_reranking --config configs/experiment.yaml --profile full
.venv\Scripts\python -m scripts.generate_results --config configs/experiment.yaml --profile full
.venv\Scripts\python -m scripts.bootstrap_ci
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m pip check
```

Validation status:

- `pytest`: 18 passed
- `pip check`: no broken requirements
- Full manifest: `results/experiment_manifest.json`
- Full summary: `results/full_run_summary.json`
- Parity/validation notes: `results/parity_validation.json`
- Bootstrap confidence intervals: `results/bootstrap_ci.json`

## Reproducing the Results

The official experimental configuration is `configs/experiment.yaml`.

Important configuration values recorded in the manifest:

- `profile`: `full`
- `seed`: 42
- `max_queries`: `null`
- `rerank_top_k`: 20
- `report_cutoff`: 10
- device: CPU

The Cross-Encoder reranks only the bi-encoder top-20 candidates and cannot
recover relevant techniques absent from that candidate set.

## Generated Results

Primary outputs:

- `results/metrics.csv`
- `results/per_query_metrics.csv`
- `results/rankings.csv`
- `results/efficiency.csv`
- `results/reranking_gain.csv`
- `results/experiment_manifest.json`
- `results/model_metadata.json`
- `results/attention_example.json`
- `results/attention_token_weights.csv`
- `results/strict_crossencoder_failure_cases.csv`
- `results/bootstrap_ci.json`
- `results/parity_validation.json`
- `results/full_run_summary.json`

Strict-subset headline metrics:

| Method | Recall@10 | MRR@10 | nDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.5530 | 0.3235 | 0.3775 | 0.3229 |
| BM25 | 0.4904 | 0.2948 | 0.3411 | 0.2943 |
| MiniLM-BiEncoder | 0.6467 | 0.3862 | 0.4478 | 0.3854 |
| MiniLM-BiEncoder+CrossEncoder | 0.6853 | 0.4310 | 0.4915 | 0.4303 |

Strict-subset reranking gains over the MiniLM bi-encoder:

- Recall@1: +0.0418
- MRR@10: +0.0449
- nDCG@10: +0.0437

## Figures and Tables

Paper figures are available under `paper/figures/`:

- `01_technique_corpus.png`
- `02_query_leakage_audit.png`
- `03_query_source_types.png`
- `04_strict_retrieval_benchmark.png`
- `05_all_vs_strict_ndcg.png`
- `06_query_latency.png`
- `07_attention_top_tokens.png`
- `08_attention_matrix.png`

The manuscript is available at `paper/manuscript.tex`.

## Code Availability

The source code, experimental configurations, reproducibility scripts, and
supporting artifacts associated with this study are available at:

https://github.com/PesquisaDoug/mitre-attack-semantic-mapping-benchmark

## Archive / DOI

This manuscript and its associated research archive are deposited on Zenodo.

Zenodo DOI: `10.5281/zenodo.22821273`

Permanent link:

`https://doi.org/10.5281/zenodo.22821273`

## Citation

If you use this work, please cite the associated article:

Douglas Felipe de Lima Silva.
"Transformer-Based Semantic Mapping of Cyber Threat Intelligence to MITRE
ATT&CK Techniques: A Retrieval and Reranking Benchmark." 2026.
Repository: https://github.com/PesquisaDoug/mitre-attack-semantic-mapping-benchmark
DOI: https://doi.org/10.5281/zenodo.22821273 (see Archive / DOI above).

Software citation metadata are also provided in `CITATION.cff`.

## License

Repository code is released under the MIT License. MITRE ATT&CK data remain
subject to MITRE's official terms of use.
