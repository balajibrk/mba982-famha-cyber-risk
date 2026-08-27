I'm building a scoped reproduction of an academic paper for a 3-4 day solo hackathon,
running on my laptop with a local NVIDIA RTX Ada 2000 GPU (8GB VRAM). I want you to
work as autonomously as possible across this whole build, using yourself as an agent
that writes code, runs it, checks results, and self-corrects — only involving me when
you're genuinely stuck. Read this entire brief before writing any code.

### PROJECT

Explainable Temporal Knowledge Graph + FAMHA-GNN Cybersecurity Risk Assessment System,
reproducing the core architecture from "Enhancing cybersecurity risk assessment using
temporal knowledge graph-based explainable decision support system" (Bag, Sarkar,
Bose — Decision Support Systems, 2025), scoped down for hackathon time, plus a novel
LLM assistant layer not in the original paper.

### SCOPE CONSTRAINTS

- Hand-curated dataset of 15-25 companies (mix of publicly breached and clean
  companies), NOT the paper's full 190-company corpus
- 1 baseline model (GATConv) for comparison, NOT the paper's 10 baselines
- No external benchmark dataset validation unless Phase 4 finishes early
- Local GPU has only 8GB VRAM: load one transformer model at a time, unload with
  `del model; torch.cuda.empty_cache()` between pipeline stages, use fp16 for
  transformer inference, prefer spaCy's CPU pipeline over `en_core_web_trf` unless
  accuracy clearly demands GPU. Check `torch.cuda.is_available()` and move tensors/
  models to GPU for every step that benefits from it (embeddings, NER inference,
  FAMHA-GNN training) — do not silently run GPU-eligible work on CPU.
- Be honest in code comments and docs about what's scoped-down vs. faithful to the
  paper — this matters for hackathon judging credibility

### AUTONOMOUS EXECUTION PROTOCOL — FOLLOW THIS EXACTLY

Work through the 7 phases below IN ORDER. For each phase:

1. **Build.** Write and run the code for that phase.
2. **Validate.** Run the validation checks listed under that phase. These are not
   optional — actually execute them, don't just claim they'd pass.
3. **If validation passes:** print a short PASS summary (what you built, what the
   validation checks showed, key numbers/artifacts produced), then immediately
   proceed to the next phase without waiting for my confirmation.
4. **If validation fails:** attempt to self-debug, up to 2 retry attempts:
   - Diagnose the likely cause from the error/output
   - Make a targeted fix
   - Re-run the validation checks
   - Repeat up to 2 times total
5. **If still failing after 2 retry attempts:** STOP. Do not proceed to the next
   phase and do not guess/paper over the issue. Instead, output a short plain-language
   summary covering:
   - What deviated from the plan (in simple terms, no jargon dump)
   - What you tried to fix it (briefly)
   - What you think the actual blocker is
   - What you need from me to proceed (a decision, a resource, clarification, etc.)
   Then wait for my input before continuing.
6. Never silently skip a validation failure and move on. Never silently substitute
   a different approach without flagging it in the phase summary.

---

### PHASE 0 — Setup & Scoping

Build: repo scaffold (data/, src/{preprocess,kg,model,train,baselines,interpret,
assistant}/, notebooks/, app/, docs/), environment setup (requirements.txt), a
company list of 15-25 companies (mix breached/clean, reuse well-documented breaches
like Uber, Capital One, JPMorgan, Citigroup, Wells Fargo, Tesla, Equifax, Target,
Marriott, SolarWinds, Colonial Pipeline, Okta, LastPass...), and a documented
labeling heuristic in docs/labeling_methodology.md (combine breach severity proxy,
disclosure delay, recurrence into 4 risk classes).

Validate:
- [ ] repo structure exists and matches the layout
- [ ] requirements.txt installs cleanly in a fresh venv
- [ ] company list has 15-25 entries, at least 3 "clean" (no major breach) for
      label balance
- [ ] labeling_methodology.md exists and defines the scoring rule explicitly

### PHASE 1 — Data Collection & Preprocessing

Build: scraper/manual-paste pipeline for policy text + breach articles per company,
cleaning pipeline (tokenize, strip noise, spellcheck, lemmatize), data_manifest.csv
logging source/date/word-count per company.

Validate:
- [ ] every company in the list has a non-empty cleaned policy file AND attack
      article file
- [ ] data_manifest.csv has one row per company, no missing/zero word counts
- [ ] spot-check 2 random cleaned files — readable, not garbled, no leftover HTML

### PHASE 2 — Temporal Knowledge Graph Construction

Build: coref resolution (spaCy-based, CPU), SVO/OIE triple extraction, cybersecurity
NER (CyNER or spaCy+gazetteer fallback), verb-clustering canonicalization
(sentence-transformers MiniLM + agglomerative clustering, cosine, threshold ~0.30,
mapped to {implements, aligns-with, violates, mitigates, causes, impacts, reports,
regulates}), temporal attributes on nodes/edges, per-company graph export to
data/processed/kg/{company}.graphml, kg_stats.csv.

Validate:
- [ ] every company has a non-trivial .graphml file (>0 nodes, >0 edges)
- [ ] kg_stats.csv populated for all companies, no NaN/zero rows
- [ ] canonicalized relation types are limited to the 8-type set (no raw messy
      verbs leaking through)
- [ ] spot-check 2 graphs by loading with networkx and printing node/edge lists —
      relationships should make semantic sense, not be nonsense pairings

### PHASE 3 — FAMHA Attention Mechanism + X-FAMHA-GNN Model

Build: node feature/embedding construction, FAMHA module implementing paper
Algorithm 1 exactly (eigenvalue-based automatic head-count determination via
covariance matrix + downward-trend/Kaiser-style elbow detection, Principal Axis
Factor Analysis decomposition, per-head scaled dot-product attention with the
paper's sqrt(d/2) scaling and sigmoid activation, composition back to original
dimension order), wrapped in a GAT-style message-passing layer stacked N times,
ELU activation, 4-way softmax classification head.

Validate:
- [ ] unit test: FAMHA on a synthetic random graph — does head count respond
      sensibly to eigenvalue spread (more spread-out eigenvalues -> more heads)?
- [ ] unit test: total FAMHA parameter count is measurably lower than an
      equivalent-shape vanilla multi-head attention layer (per Theorem 3.1's
      inequality) — print both counts
- [ ] full model runs one forward + backward pass without shape errors on a real
      company KG from Phase 2
- [ ] GPU is actually used for this (check `next(model.parameters()).is_cuda`)

### PHASE 4 — Training, Baseline Comparison

Build: leave-one-out (or small k-fold) training loop on the case-study set, 1
baseline (GATConv from torch_geometric) trained the same way, accuracy/F1/
precision/recall logging, Mann-Whitney U significance test between the two.

Validate:
- [ ] training loss decreases over epochs (no NaN, no flatlining at random-chance
      accuracy)
- [ ] X-FAMHA-GNN achieves accuracy meaningfully above the majority-class baseline
      (print the majority-class baseline number for comparison)
- [ ] baseline model also trains successfully and produces a comparable metrics
      table
- [ ] p-value from Mann-Whitney U test is computed and reported (even if not
      significant given small N — report honestly)

### PHASE 5 — Interpretability

Build: SHAP explainer (KernelExplainer, model-agnostic) over node/entity features,
attention weight extraction/heatmap from trained FAMHA layers, for at least 2
companies.

Validate:
- [ ] SHAP values generated without errors for >=2 companies, bar chart saved
- [ ] attention heatmap generated for the same companies, saved
- [ ] sanity check: do the top SHAP/attention entities make plausible sense given
      that company's known breach (if it's a breached company)? Flag in the
      summary if they look nonsensical rather than silently accepting them

### PHASE 6 — LLM Assistant Layer (counterfactual + grounded narrative)

Build:
(a) counterfactual re-scoring — edit a copy of a company's KG to simulate closing
    the top SHAP/attention-flagged gap, re-run the TRAINED model on the edited
    graph, report real risk_before -> risk_after delta
(b) grounded narrative generation — prompt an LLM with ONLY precomputed evidence
    (risk class, top SHAP entities, top attention pairs, counterfactual delta),
    hard system-prompt rule to never reference anything not in that evidence,
    structured JSON output: one_line_verdict, why, fix, impact
(c) if time remains: minimal Streamlit demo app tying KG -> risk score -> SHAP/
    attention -> counterfactual -> narrative together for a chosen demo company

Validate:
- [ ] counterfactual re-score produces a real, nonzero delta computed by the
      actual trained model (not a hardcoded/estimated number)
- [ ] narrative JSON output conforms to the schema and, on manual check, does not
      reference any entity/number absent from the input evidence
- [ ] end-to-end run for at least 1 company: KG -> risk score -> SHAP/attention ->
      counterfactual -> narrative, no manual intervention required mid-run
- [ ] (if built) Streamlit app launches and displays the above for a dropdown-
      selected company without errors

---

### FINAL STEP

After Phase 6 passes validation, produce a short overall summary: what was built,
what was scoped down vs. the paper and why, key results (accuracy/F1 vs baseline,
p-value, example counterfactual delta), and a list of anything that had to be
descoped due to time or repeated validation failures.