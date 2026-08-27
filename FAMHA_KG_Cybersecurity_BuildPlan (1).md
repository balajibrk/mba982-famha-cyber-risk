# Temporal KG + X-FAMHA-GNN Cybersecurity Risk Assessment
## Reproduction Build Plan — Solo, 3–4 Days, Local GPU (RTX Ada 2000, 8GB VRAM)

**GPU/VRAM notes (read before Phase 2):** 8GB is plenty for this project — the
X-FAMHA-GNN model itself is tiny (small graphs, low-dim embeddings), so training
in Phase 3-4 will barely touch VRAM. The risk is transformer models stacking up:
sentence-transformers MiniLM (~80MB) is fine, but a NER model like SecureBERT-based
CyNER plus a spaCy transformer pipeline loaded simultaneously can get tight.
Rules of thumb: load one HF/transformer model at a time and `del model; torch.cuda.
empty_cache()` between pipeline stages in Phase 2; use `model.half()` / fp16 for any
transformer inference (embeddings/NER) to roughly halve memory; keep FAMHA-GNN
training batch size small (your graphs are small anyway, this won't hurt); prefer
spaCy's CPU pipeline (`en_core_web_sm`/`md`) over `en_core_web_trf` for NER unless
accuracy demands it — frees the GPU entirely for embedding + model training.

Scope: and-curate a small real-world set (~10-15 companies) with public policies + one well-documented breach each (Uber, Capital One, JPMorgan, etc. — several already named in the paper's own case study, which gives you labeled ground truth for free). Treat this as a proof-of-concept dataset, explicitly flagged as smaller-scale, faithful FAMHA-GAT architecture, 2–3 baselines,
1 benchmark dataset for external validation, SHAP + attention interpretability, LLM-based
explainer layer on top. This is an honest scaled-down reproduction, not a literal 190-company
rebuild — flag this clearly in your hackathon writeup/demo.

Repo layout to create first:

```
famha-cyberrisk/
├── data/
│   ├── raw/policies/            # scraped/pasted policy text per company
│   ├── raw/attacks/              # breach articles per company
│   ├── labels/                   # risk labels (low/med/high/critical)
│   └── processed/                # cleaned text, KG exports
├── src/
│   ├── preprocess/
│   ├── kg/
│   ├── model/
│   ├── train/
│   ├── baselines/
│   ├── interpret/
│   └── assistant/
├── notebooks/                    # Colab entry points per phase
├── app/                          # optional Streamlit demo
└── docs/
```

---

## PHASE 0 — Setup & Scoping (2–3 hrs, Day 1 morning)

**Goal:** environment ready, company list locked, labeling scheme defined.

```python
# notebooks/00_setup.ipynb

# TODO: pip install torch torch_geometric spacy sentence-transformers
#       shap scikit-learn networkx neo4j-driver (optional) openai/anthropic-sdk

# TODO: define COMPANY_LIST = [ ... 15-25 companies ... ]
#       Reuse companies already documented in the paper's own case studies —
#       free ground truth: Uber (2016 breach), Capital One (2019), JPMorgan (2014),
#       Citigroup (2011), Wells Fargo (2008), Tesla (insider breach), Equifax (2017),
#       Target (2013), Marriott (2018), SolarWinds (2020), Colonial Pipeline (2021),
#       Twitter/X (2022), LastPass (2022), MOVEit/Progress (2023), Okta (2022)...
#       Mix breached + a few "clean" companies for label balance.

# TODO: define LABEL_SCHEME = {0: 'low_risk', 1: 'medium_risk', 2: 'high_risk', 3: 'critical_risk'}
#       Label heuristic (since you don't have idtheftcenter/upguard scale access):
#       combine (a) breach severity proxy = records exposed (log-scaled bucket),
#               (b) time-to-disclosure (regulatory delay = higher risk),
#               (c) recurrence (repeat breaches = higher risk)
#       Document this scoring rule explicitly in docs/labeling_methodology.md —
#       judges will ask about label provenance, be ready to defend it.
```

**Acceptance criteria:** repo scaffolded, company list + label rule committed to `docs/`.

---

## PHASE 1 — Data Collection & Preprocessing (Day 1, ~6 hrs)

**Goal:** raw policy text + attack articles per company, cleaned per Phase 1 of the paper.

```python
# src/preprocess/scrape.py

# TODO: for each company in COMPANY_LIST:
#         - pull public cybersecurity/privacy/trust-center policy pages
#           (requests + BeautifulSoup; many companies publish these at
#            /trust, /security, /privacy — check robots.txt, be polite)
#         - pull 1-3 breach/news articles (TechCrunch, Krebs on Security,
#           company press releases) — save raw HTML + extracted text
#       Save to data/raw/policies/{company}.txt and data/raw/attacks/{company}.txt
#       NOTE: budget 3-4 hrs here max. If scraping stalls on any company,
#       manually paste text from the company's published policy PDF — faster
#       and still legitimate given your 15-25 scale.

# src/preprocess/clean.py

# TODO: replicate paper's Phase 1 pipeline:
#   1. sentence tokenize (spacy / nltk)
#   2. strip stopwords, URLs, emoji, repeated punctuation (regex)
#   3. spellcheck pass (use `symspellpy` — Hunspell is finicky to install in Colab)
#   4. lemmatize (spacy lemmatizer, equivalent to WordNetLemmatizer)
#   Output: data/processed/{company}_policy_clean.txt
#           data/processed/{company}_attack_clean.txt
```

**Acceptance criteria:** clean text files for every company, a `data_manifest.csv` logging
source URL, scrape date, word count per company (this table doubles as your dataset-card
for the writeup).

---

## PHASE 2 — Temporal Knowledge Graph Construction (Day 2, full day)

**Goal:** per-company temporal KG, matching paper's 6-step pipeline (Section 3.2.2).

```python
# src/kg/coref.py

# TODO: coreference resolution using spacy + coreferee (lighter than AllenNLP coref,
#       small footprint, runs fine on CPU — no need to burn GPU/VRAM on this step).
#       Resolve "the company"/"the firm"/pronouns -> canonical
#       company name across policy + attack text.

# src/kg/oie.py

# TODO: Open Information Extraction — use Stanford OpenIE via `openie` python wrapper,
#       or fallback to spaCy dependency-parse SVO extraction (simpler, good enough at
#       this scale, doesn't need a JVM). Extract (subject, verb, object) triples from
#       every coref-resolved sentence.

# src/kg/ner.py

# TODO: Cybersecurity NER — don't train CyNER from scratch (out of scope for 3-4 days).
#       Use the pretrained CyNER model (github.com/aiforsec/CyNER, based on SecureBERT)
#       for entity typing: organization / attack-method / policy-area / asset.
#       Run it in fp16 on GPU, then `del model; torch.cuda.empty_cache()` before the
#       next pipeline stage — keeps you well within 8GB VRAM.
#       If VRAM gets tight or setup is fiddly, fallback: spaCy en_core_web_sm/md
#       (CPU, zero VRAM cost) + a small custom gazetteer of ~40 cybersecurity terms
#       (breach, ransomware, phishing, MFA, encryption, access control, vendor risk,
#       incident response, ...).

# src/kg/canonicalize.py

# TODO: replicate paper's exact verb-clustering step:
#   1. for each triple <s,v,o>: build context string "s-lemma v-lemma o-lemma"
#   2. embed with sentence-transformers 'all-MiniLM-L6-v2' (384d) — matches paper exactly
#   3. agglomerative clustering, average linkage, cosine distance, threshold tau=0.30
#      (sklearn.cluster.AgglomerativeClustering(distance_threshold=0.30, n_clusters=None,
#       linkage='average', metric='cosine'))
#   4. map each cluster -> nearest of {implements, aligns-with, violates, mitigates,
#      causes, impacts, reports, regulates} via small seed lexicon + majority vote
#   This collapses your messy verb triples into the paper's 8 canonical relation types.

# src/kg/temporal.py

# TODO: attach timestamps as node/edge attributes:
#   - attack event nodes get 'occurred_at' (extract via dateparser on attack article text)
#   - policy nodes get 'published_at' if available, else scrape date as proxy
#   - build edges Company -[relation]-> PolicyArea, Company -[occurred_on]-> AttackEvent,
#     AttackEvent -[exploited]-> PolicyArea (heuristic: co-occurring entities in same
#     paragraph, or manual review for your 15-25 companies — feasible at this scale)

# src/kg/build_graph.py

# TODO: assemble per-company networkx.DiGraph, export to:
#   data/processed/kg/{company}.graphml   (for inspection / Neo4j import if desired)
#   Log basic stats (nodes, edges, density) per company into docs/kg_stats.csv
#   — mirrors paper's Section 4.2 descriptive stats table, good for your writeup.
```

**Acceptance criteria:** one queryable KG per company, `kg_stats.csv` populated,
at least 3 KGs manually spot-checked for correctness (open in Gephi or networkx draw).

---

## PHASE 3 — Graph Embeddings + FAMHA-GAT Model (Day 3, full day)

**Goal:** implement the paper's core novelty — FAMHA — faithfully, wrapped in a GAT backbone.

```python
# src/model/embedding.py

# TODO: node feature construction per paper's "Graph Embedding Construction":
#   - node type one-hot (company/policy/attack/entity)
#   - text embedding of node label (sentence-transformers, reuse MiniLM)
#   - temporal feature (days since earliest event in graph, normalized)
#   - concat -> pass through a learnable nn.Linear embedding layer (paper: "randomly
#     initialized... updated via backpropagation")

# src/model/famha.py  <-- THE CORE DELIVERABLE, implement exactly per Algorithm 1

class FAMHA(nn.Module):
    """
    Factor-Analysis-based Multi-Head Attention.
    Reproduces paper Section 3.2.3, Algorithm 1, Eqs. 1-6.
    """
    def determine_num_heads(self, G):
        # TODO Step 1: covariance C = (1/n) sum (psi_j - psi_bar)(psi_j - psi_bar)^T
        #      eigen-decompose C, sort ascending, find "downward trend" (elbow) -> h
        #      practical elbow rule: h = argmax second-derivative of sorted eigenvalues,
        #      or simplest: h = number of eigenvalues > mean(eigenvalues) (Kaiser criterion
        #      — this is literally what Principal Factor Analysis uses for factor retention,
        #      consistent with paper's PAFA citation [29])
        pass

    def decompose(self, G, h):
        # TODO Step 2: Principal Axis Factor Analysis via sklearn's FactorAnalysis
        #      (n_components=h) OR statsmodels Factor(method='pa') for closer match
        #      to "Principal Axis-based Factor Analysis" terminology.
        #      Split G's d columns into h groups by dominant-factor-loading assignment
        #      (each original dimension assigned to whichever factor loads highest on it)
        #      -> {G_1, ..., G_h}, sum of widths = d  (Eq. 1)
        pass

    def forward(self, G):
        # TODO Step 3: for each i in 1..h:
        #        q_i, k_i, v_i = W_q_i @ G_i, W_k_i @ G_i, W_v_i @ G_i
        #        alpha_i = softmax(q_i k_i^T / sqrt(d_i/2)) v_i     (Eq. 2, note the /2 !)
        #        G_i' = sigmoid( sum over neighbors_j alpha_kj * G_ik )
        #      Step 4: G' = concat/compose G_1'...G_h' back to original dimension order
        pass
```

```python
# src/model/xfamha_gnn.py

# TODO: wrap FAMHA as the attention layer inside a GAT-style message-passing block
#       (torch_geometric.nn.MessagePassing base class), stack N layers, follow with
#       ELU activation (paper explicitly chooses ELU over ReLU), then a 4-way softmax
#       classification head (matches paper's 4 risk classes).
#       Reference param-count reduction from Theorem 3.1 — log actual param count vs.
#       a same-shape vanilla multi-head GAT to reproduce that comparison in your writeup.
```

**Acceptance criteria:** FAMHA unit-tested on a synthetic random graph (does head count
respond sensibly to eigenvalue spread? do parameter counts satisfy Eq. 6 inequality?),
full model does one forward/backward pass without shape errors on a real company KG.

---

## PHASE 4 — Training, Baselines & Benchmark Validation (Day 4, full day)

**Goal:** train on your 15-25 company set, compare against 2-3 baselines, validate on one
public benchmark dataset.

```python
# src/train/train_case_study.py

# TODO: split 15-25 companies -> small train/val/test (given the tiny N, use
#       leave-one-out or 5-fold CV instead of the paper's 10-fold*5-run — be explicit
#       in the writeup that N is small and results are illustrative, not statistically
#       powered like the paper's 190-company study)
# TODO: train X-FAMHA-GNN, log accuracy/F1/precision/recall/G-mean per fold

# src/baselines/run_baselines.py

# TODO: implement/import from torch_geometric.nn:
#         - GATConv  (paper baseline i)
#         - GINConv  (paper baseline ii)
#         - SAGEConv (paper baseline iii, "GraphSage")
#       Skip DropGIN/GAT-BiSep/GCNFN/DiffWire/HGFND/UPFD-Sage/GGNN — not worth the
#       integration cost at this scope; note in writeup as "future work / full paper
#       compares against 10 SOTA models, we validate against the 3 most foundational."

# src/baselines/benchmark_dataset.py

# TODO: from torch_geometric.datasets import UPFD
#       dataset = UPFD(root='data/upfd', name='politifact', feature='bert')
#       Run X-FAMHA-GNN + your 3 baselines on this ONE public dataset for an
#       apples-to-apples sanity check that your architecture isn't just overfitting
#       to your tiny hand-labeled set. This is your strongest credibility signal.

# src/train/significance.py

# TODO: scipy.stats.mannwhitneyu(famha_scores, baseline_scores) per baseline,
#       report p-values in a small heatmap (matplotlib), matching paper Fig. 6 style
```

**Acceptance criteria:** results table (accuracy/F1 per model, case-study set + UPFD-Politifact),
at least one p-value comparison, honest note in docs on sample-size limitations.

---

## PHASE 5 — Interpretability Layer (Day 5 morning, ~3 hrs)

**Goal:** SHAP values + attention heatmaps, matching paper Section 3.2.4 / 5.5.

```python
# src/interpret/shap_explain.py

# TODO: wrap model.predict as a shap.Explainer-compatible callable over node/entity
#       feature vectors; use shap.KernelExplainer (model-agnostic, works regardless
#       of GNN internals) on a background set of ~20 sampled nodes.
#       Output: bar chart of top entities pushing prediction toward
#       lower/higher vulnerability, per company — mirrors paper Fig. 9(a).

# src/interpret/attention_heatmap.py

# TODO: hook into FAMHA.forward to cache alpha_i per layer per company at inference,
#       render as a seaborn heatmap: rows=policy entities, cols=attack-article entities,
#       mirrors paper Fig. 9(b) (their Tesla example: employees/insider -> access control).
#       Pick your 2 most interesting companies (ideally ones with a real, documented
#       breach mechanism) to replicate that kind of narrative explanation.
```

**Acceptance criteria:** for at least 2 companies, a SHAP bar chart + attention heatmap
that tells a coherent "here's why this policy is risky" story — this is your best demo material.

---

## PHASE 6 — LLM Security-Assistant Layer (Day 5 afternoon, ~3–4 hrs)

**Goal:** don't just summarize the model's output — use the LLM to narrate a number your
model *actually recomputed*. This is the headline differentiator: most "LLM + ML" hackathon
projects just wrap a prompt around a chart. Here the LLM reports a real counterfactual
re-scoring, not an estimate it invented.

### 6a. Core (build this, non-negotiable — ~2 hrs)

```python
# src/assistant/counterfactual.py

# TODO: given the top-1 SHAP/attention-flagged vulnerability (e.g. the
#       "employee -> access -> confidential_data" edge with high attention weight),
#       programmatically edit a COPY of the company's KG:
#         - remove/reweight that edge, OR
#         - inject a mitigating node (e.g. add "MFA_enforced" node connected to
#           the flagged access-control policy node)
#       Re-run the TRAINED X-FAMHA-GNN forward pass on the edited graph.
#       Report: risk_before -> risk_after as a probability delta.
#       This mirrors the paper's Appendix C / Table 3 "simulated residual risk"
#       exercise, but the paper does it as a static table — you do it live, per
#       company, on demand. That's the upgrade, and it's honest: the delta comes
#       from your own model's forward pass, not from the LLM.

# src/assistant/narrative.py

# TODO: prompt template, evidence-only, structured/JSON output:
#   INPUT (all pre-computed — LLM never invents anything):
#     - company, risk_class, confidence
#     - top-3 SHAP entities with values
#     - top attention pairs (policy_entity, attack_entity, weight)
#     - counterfactual result: risk_before, risk_after, what_changed
#   OUTPUT (force via JSON schema):
#     - one_line_verdict
#     - why            (cites ONLY the entities/pairs given in input)
#     - fix            (the specific change actually tested in the counterfactual)
#     - impact         ("closing this gap reduces predicted risk by X%, per model re-scoring")
#   Hard system-prompt rule: "Only reference entities, edges, or numbers present in
#   the input. Never name a policy area, control, or number not provided." — this
#   is what keeps the narrative grounded and defensible if a judge fact-checks it.
```

### 6b. Stretch (if Day 5 goes well — ~1 hr)

```python
# src/assistant/audience.py

# TODO: two renderers off the SAME grounded narrative object:
#   - exec_summary(): 3 sentences, board-level, no jargon, leads with risk % / dollar proxy
#   - engineer_ticket(): Jira-style — title, severity, affected control, remediation
#     steps, acceptance criteria
#   Cheap to add (same evidence, two prompt templates), but signals you thought about
#   who actually consumes this output — judges notice that kind of detail.

# src/assistant/kg_agent.py

# TODO: give the LLM ONE tool: query_kg(company, entity) -> subgraph context.
#   Instead of dumping the whole KG into the prompt, let it ask a follow-up
#   ("what other policies touch 'vendor risk'?") mid-generation. Small effort if
#   the KG is already networkx — just wrap a lookup function as a tool call.
#   Makes the assistant feel investigative rather than templated.
```

### 6c. Reach (say it, don't build it — roadmap slide only)

- Policy clause **rewrite drafting** (LLM proposes actual replacement policy language,
  not just "add MFA")
- Cross-company peer benchmarking narrative ("your access-control language is weaker
  than most peers in your sample")
- Continuous monitoring mode — re-run automatically as new attack articles are published

```python
# app/demo_app.py

# TODO: minimal Streamlit app: dropdown of your companies -> shows KG snapshot,
#       risk class, SHAP chart, attention heatmap, counterfactual delta,
#       LLM narrative (exec + engineer views if 6b done) — this is your live demo.
#       Budget 1-2 hrs; judges respond far more to a working UI than a notebook.
```

**Acceptance criteria:** for a chosen demo company, one clean end-to-end run:
KG → risk score → SHAP/attention → counterfactual re-score → grounded LLM narrative →
displayed in the app. Pitch line to say out loud: *"the LLM doesn't estimate the risk
reduction — it reports what our model measured when we simulated the fix."*

---

## Day-by-Day Time Budget (3–5 days, solo)

| Day | Focus | Output |
|---|---|---|
| 1 | Setup + data collection + cleaning | Clean policy/attack text, 15-25 companies |
| 2 | KG construction (coref → OIE → NER → canon → temporal) | Per-company temporal KG |
| 3 | FAMHA implementation + X-FAMHA-GNN model | Working, unit-tested model |
| 4 | Training + baselines + UPFD benchmark + significance | Results tables, p-values |
| 5 | SHAP/attention interpretability + LLM assistant + demo app | Live demo, writeup |

If you only have 3 days instead of 5: cut Phase 4's UPFD benchmark validation and
baseline count down to just GAT (1 baseline), and simplify Phase 6 to notebook output
instead of a Streamlit app — protect Phases 2–3 (KG + FAMHA) since that's your core novelty.

## What to say explicitly in your writeup (protects credibility with judges)

- Dataset is a **15–25 company proof-of-concept**, not the paper's 190+154 company corpus —
  state this upfront, don't let it look discovered.
- Labels use a **documented heuristic** (see `docs/labeling_methodology.md`), not the paper's
  idtheftcenter/upguard-sourced severity data.
- Baseline comparison is **3 models on 1 benchmark + your case study**, not 10 models on 4 benchmarks.
- The **FAMHA mechanism itself is reproduced faithfully** to the paper's math — this is your
  strongest, most defensible claim, lead with it.
