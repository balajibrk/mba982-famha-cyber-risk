# Temporal KG + X-FAMHA-GNN Cybersecurity Risk Assessment

A scoped, honest reproduction of:

> Bag, S., Sarkar, S., & Bose, I. (2025). *Enhancing cybersecurity risk assessment
> using temporal knowledge graph-based explainable decision support system.*
> Decision Support Systems 198, 114526.

plus a novel LLM security-assistant layer (grounded narrative + real counterfactual
re-scoring) not present in the original paper.

## What this is

An end-to-end pipeline that, for each of ~18 companies:

1. builds a **temporal cybersecurity knowledge graph** from policy + breach text,
2. classifies its policy **risk level (4 classes)** with a faithful reimplementation
   of the paper's **FAMHA** mechanism (Factor-Analysis-based Multi-Head Attention,
   Algorithm 1) wrapped in a GAT-style GNN,
3. **explains** the prediction with SHAP values + FAMHA attention heatmaps,
4. runs a **counterfactual re-scoring** (close the top flagged gap, re-run the
   trained model, report the real risk delta) narrated by a **local LLM**.

## Honest scope (vs. the paper)

| Aspect | This repo | Paper |
|---|---|---|
| Companies | ~18 proof-of-concept | 190 train / 154 test |
| Labels | documented heuristic (`docs/labeling_methodology.md`) | idtheftcenter / upguard severity |
| Baselines | GATConv + majority-class | 10 SOTA models |
| Benchmarks | UPFD-Politifact (stretch) | 4 benchmark datasets |
| FAMHA mechanism | **reproduced faithfully to the math** | original |

## Environment

Built for a Windows laptop with an NVIDIA RTX 2000 Ada (8 GB) GPU.

- Python **3.12** (installed via `uv`; the system default 3.14 lacks wheels for the ML stack).
- CUDA PyTorch 2.6 (`cu124`), installed from the PyTorch index.
- **Smart App Control note:** the target machine blocks unsigned compiled DLLs, so
  spaCy (blocked) is replaced with **NLTK** (pure Python, and it provides the exact
  `WordNetLemmatizer` the paper cites), and `numba`/`llvmlite` (an optional `shap`
  JIT dependency, also blocked) are shimmed by a no-op package in `./stubs`.

### Setup

```powershell
pip install uv
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','stopwords']]"
```

## Documentation for review

- **[docs/MBA982_Project_Module_Report.md](docs/MBA982_Project_Module_Report.md)** / **[PDF](docs/MBA982_Project_Module_Report.pdf)** — MBA982 project module report (phases, data, AI + LLM)
- **[docs/SUMMARY.md](docs/SUMMARY.md)** — concise results and scope-down table
- **[docs/labeling_methodology.md](docs/labeling_methodology.md)** — how 4-class labels are derived

## Why the LLM layer stands out

Most “AI + LLM” demos have a chatbot *describe* a chart. Here the language layer reports a number the **trained GNN recalculated** after simulating a remediation on the knowledge graph—not a number the LLM invented. Impact is proof from the model; prose is packaging.

## Layout

```
data/        raw policy + attack text, cleaned text, KG exports, labels
src/
  preprocess/  corpus build + cleaning (NLTK)
  kg/          coref, OIE, NER, canonicalization, temporal, graph build
  model/       node embeddings, FAMHA (Algorithm 1), X-FAMHA-GNN
  train/       leave-one-out training, significance testing
  baselines/   GATConv baseline
  interpret/   SHAP + attention heatmaps
  assistant/   counterfactual re-scoring + grounded LLM narrative
app/         Streamlit demo
docs/        MBA982 module report (md+pdf), methodology, stats, summary
stubs/       no-op numba shim (Smart App Control workaround)
```
