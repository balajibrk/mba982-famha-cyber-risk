# Project Summary — Temporal KG + X-FAMHA-GNN Cybersecurity Risk Assessment

A scoped, honest reproduction of Bag, Sarkar & Bose (2025), *"Enhancing
cybersecurity risk assessment using temporal knowledge graph-based explainable
decision support system"* (Decision Support Systems 198, 114526), plus a novel
LLM security-assistant layer. Built solo on an NVIDIA RTX 2000 Ada (8 GB) laptop.

## What was built (all 7 phases, end to end)

| Phase | Deliverable | Status |
|------:|-------------|--------|
| 0 | Repo scaffold, config for 18 companies, 4-class labeling heuristic | PASS |
| 1 | Curated policy+breach corpus (grounded in public facts) + NLTK cleaning | PASS |
| 2 | Temporal KG per company: coref -> SVO OIE -> cyber NER -> verb-clustering to the paper's 8 relations -> temporal labels | PASS |
| 3 | **FAMHA (Algorithm 1)** + X-FAMHA-GNN, with unit tests | PASS |
| 4 | Leave-one-out training, GATConv + majority baselines, Mann-Whitney U | PASS |
| 5 | SHAP entity attributions + FAMHA attention heatmaps | PASS |
| 6 | Counterfactual re-scoring + grounded Ollama narrative + Streamlit app | PASS |

## Faithful core: FAMHA (the paper's key novelty)

`src/model/famha.py` reproduces Algorithm 1 to the math:

- **Optimal head count** from the feature-covariance eigenvalue spread (Kaiser
  criterion). Unit test confirms heads rise with spread (concentrated spectrum -> 2
  heads, spread spectrum -> 12 heads).
- **Principal Axis Factor Analysis** partitions the `d` feature columns into `h`
  groups (`sum(len_i) = d`, Eq. 1), each head getting its own `len_i x len_i`
  projections.
- **Attention** with the paper's `sqrt(d/2)` scaling and sigmoid, applied over
  graph neighbours (adjacency-masked message passing), then **composed** back to
  the original column order (Eq. 2).
- **Parameter reduction (Theorem 3.1 / Eq. 6):** `theta_FAMHA = 3*sum(len_i^2)`.
  In the trained model this is **1,638 vs 9,216** for equivalent vanilla
  multi-head attention — a **5.6x reduction** — with `sum(len_i) = d` verified.

## Key results

Case-study set: **18 companies** (15 breached, 3 comparatively clean).
Label distribution: low 3 / medium 5 / high 8 / critical 2.

### Leave-one-out cross-validation (single run)

| Model | Accuracy | Macro-F1 |
|-------|---------:|---------:|
| **X-FAMHA-GNN** | **0.611** | **0.461** |
| GATConv baseline | 0.444 | 0.329 |
| Majority-class | 0.444 | 0.154 |

### Robustness across 5 seeds + significance (Mann-Whitney U, one-sided)

| Model | Accuracy (mean ± std) | Macro-F1 (mean) |
|-------|----------------------:|----------------:|
| X-FAMHA-GNN | 0.600 ± 0.042 | 0.495 |
| GATConv | 0.456 ± 0.074 | 0.340 |

- X-FAMHA-GNN **> GATConv**: p = **0.0040** (F1), 0.0096 (accuracy).
- X-FAMHA-GNN **> majority**: p = **0.0037** (F1), 0.0035 (accuracy).

All below α = 0.05 — X-FAMHA-GNN significantly outperforms both baselines,
reproducing the paper's central claim on this scoped dataset.
(See `artifacts/pvalue_heatmap.png`, `docs/results.csv`.)

### Interpretability (Phase 5)

- **SHAP** correctly surfaces each breach's real drivers: Uber -> `credentials`,
  `aws s3 bucket`, `S3`; Capital One -> `cloud configurations`, `internal
  metadata`, `cloud security posture` (see `artifacts/shap_*.png`).
- **FAMHA attention** heatmaps link policy entities to attack entities
  (`artifacts/attention_*.png`).

### Novel assistant layer (Phase 6) — beyond the paper

- **Real counterfactual re-scoring on the trained model** (not a static table):
  simulate remediation (neutralize exploited-weakness entities + enforce MFA),
  re-run the GNN, report the true delta.
  - Uber: high/critical risk **1.000 -> 0.041 (-95.9%, critical -> low)**.
  - Capital One: **1.000 -> 0.924 (-7.6%)**.
- **Grounded LLM narrative** via local **Ollama (llama3.1:8b)** with a hard
  evidence-only rule and an automatic ungrounded-number check; deterministic
  template fallback if the daemon is down. Emits a strict
  `{one_line_verdict, why, fix, impact}` JSON plus exec-summary and
  engineer-ticket views.
- **Streamlit demo** (`app/demo_app.py`) launches and serves (HTTP 200).

## What was scoped down vs. the paper (stated honestly)

| Aspect | This repo | Paper | Why |
|--------|-----------|-------|-----|
| Companies | 18 (proof-of-concept) | 190 train / 154 test | Solo hackathon, no scraping infra |
| Labels | documented 4-factor heuristic (`docs/labeling_methodology.md`) | idtheftcenter / upguard severity | Those sources are proprietary |
| Corpus | curated text grounded in public facts, flagged PoC | scraped policy + breach pages | Live scraping is fragile |
| Baselines | GATConv + majority-class | 10 SOTA models | Time-boxed; GATConv is the core graph-attention baseline |
| Benchmarks | (descoped) | 4 benchmark datasets | Focused on the case study |
| Coref / OIE / NER | rule-based / NLTK-POS / gazetteer | neural coref / Stanford OpenIE / CyNER | Smart App Control blocks compiled DLLs (see below) |

## Environment deviations (forced, documented)

- **Python 3.12 via `uv`** — the system default (3.14) lacks ML-stack wheels.
- **Smart App Control** is enabled on the machine and blocks unsigned compiled
  DLLs. This forced two swaps, both handled cleanly:
  1. **spaCy -> NLTK** for tokenize/lemmatize/POS. NLTK is pure Python and
     provides the exact `WordNetLemmatizer` the paper cites (arguably *more*
     faithful).
  2. **numba/llvmlite** (an optional `shap` JIT dep whose `llvmlite.dll` is
     blocked) replaced by a no-op shim in `./stubs`; `shap` runs in pure-Python
     mode. Correctness unaffected.
- GPU (CUDA 12.4, PyTorch 2.6) is used for all transformer embedding and the
  model's forward/backward (verified `is_cuda`); the tiny-graph LOO loop is
  batched block-diagonally for ~8x speedup.

## Honest limitations

- **Small, skewed N** (18 companies, only 2 critical): results are illustrative,
  not statistically powered like the 190-company study. G-mean is low because the
  critical class cannot be learned under leave-one-out with 2 examples.
- **Labels are a heuristic proxy**, not audited severity data.
- **Curated corpus** is grounded in public facts but authored for this project;
  it is not scraped ground truth.

## How to reproduce

```powershell
# env (see README.md for full setup)
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# pipeline
.venv\Scripts\python.exe -m src.preprocess.build_corpus
.venv\Scripts\python.exe -m src.preprocess.clean
.venv\Scripts\python.exe -m src.kg.build_graph
.venv\Scripts\python.exe -m src.model.features
.venv\Scripts\python.exe -m src.train.train_case_study
.venv\Scripts\python.exe -m src.baselines.run_baselines
.venv\Scripts\python.exe -m src.train.significance
.venv\Scripts\python.exe -m src.interpret.shap_explain
.venv\Scripts\python.exe -m src.interpret.attention_heatmap
.venv\Scripts\python.exe -m src.assistant.pipeline uber capital_one

# per-phase validation
.venv\Scripts\python.exe -m tests.validate_phase1   # ... through validate_phase6

# demo
.venv\Scripts\streamlit run app\demo_app.py
```
