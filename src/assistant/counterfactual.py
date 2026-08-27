"""Phase 6a - counterfactual re-scoring on the TRAINED model.

Given a company's top flagged vulnerability, edit a COPY of its knowledge graph
to simulate closing that gap, then re-run the actual trained X-FAMHA-GNN on the
edited graph and report the real risk_before -> risk_after delta. The delta comes
from the model's own forward pass, never from an LLM or a hardcoded table (this
is the honest upgrade over the paper's static residual-risk table, Appendix C).

The edit: (1) remove the graph's 'violates' and 'causes' edges (the structural
weaknesses / attack-causation links), and (2) inject a mitigating POLICY node
(e.g. enforced MFA + least privilege) linked to the company.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from src import config as C
from src.interpret.common import node_sources
from src.model.features import NODE_TYPES, TYPE_INDEX, D_TEXT, _norm_year
from src.model.xfamha_gnn import dense_adj
from src.kg import ner

RISK_CLASSES = (2, 3)  # high + critical => "risk probability"


def _risk_prob(probs: np.ndarray) -> float:
    return float(sum(probs[c] for c in RISK_CLASSES))


def _embed_phrase(phrase: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(C.SBERT_MODEL, device=device)
    if device == "cuda":
        m = m.half()
    v = m.encode([phrase], normalize_embeddings=True, convert_to_numpy=True)[0]
    C.free_gpu(m)
    return v.astype(np.float32)


@torch.no_grad()
def _risk_from_graph(model, x, adj, keep_mask, device) -> np.ndarray:
    """Forward pass that pools only over kept nodes (masked-out nodes removed)."""
    x = x.to(device)
    adj = adj.to(device)
    m = keep_mask.to(device).unsqueeze(1)
    x = x * m                                    # zero removed-node features
    a = adj * m * m.t()                          # cut edges to/from removed nodes
    a = a.clone()
    idx = torch.arange(a.shape[0], device=device)
    a[idx, idx] = m.squeeze(1)                   # self-loops only for kept nodes
    h = model.encode_dense(x, a)
    pooled = (h * m).sum(dim=0, keepdim=True) / m.sum().clamp(min=1.0)
    logits = model.head(pooled)
    return F.softmax(logits, dim=1).cpu().numpy()[0]


def counterfactual(slug: str, model, dataset, device,
                   fix_phrase: str = "enforce device-based multi-factor authentication and least privilege access") -> dict:
    data = dataset[slug]
    comp = C.COMPANIES_BY_SLUG[slug]
    sources = node_sources(slug)                 # per node, in dataset order

    x = data["x"].clone()
    n = x.shape[0]
    adj = dense_adj(data["edge_index"], n)       # symmetric + self loops

    # --- baseline risk (all nodes present) --------------------------------- #
    keep_before = torch.ones(n)
    probs_before = _risk_from_graph(model, x, adj, keep_before, device)
    risk_before = _risk_prob(probs_before)

    # --- remediation scenario --------------------------------------------- #
    # (1) neutralize the exploited-weakness (attack-article) entities, and
    # (2) inject a mitigating POLICY control node linked to the company.
    new_feat = np.zeros((1, x.shape[1]), dtype=np.float32)
    new_feat[0, TYPE_INDEX[ner.POLICY]] = 1.0
    new_feat[0, len(NODE_TYPES):len(NODE_TYPES) + D_TEXT] = _embed_phrase(fix_phrase)
    new_feat[0, -1] = _norm_year(comp.breach_year if comp.breached else 2023)
    x_edit = torch.cat([x, torch.from_numpy(new_feat)], dim=0)

    # expand adjacency for the new node, linked to the company (ORG) node
    comp_idx = next((i for i, s in enumerate(sources)
                     if data["node_types"][i] == ner.ORG), 0)
    n2 = n + 1
    adj_edit = torch.zeros(n2, n2)
    adj_edit[:n, :n] = adj
    adj_edit[comp_idx, n] = adj_edit[n, comp_idx] = 1.0
    adj_edit[n, n] = 1.0

    n_removed = sum(1 for s in sources if s == "attack")
    keep_after = torch.tensor(
        [0.0 if s == "attack" else 1.0 for s in sources] + [1.0], dtype=torch.float32)
    probs_after = _risk_from_graph(model, x_edit, adj_edit, keep_after, device)
    risk_after = _risk_prob(probs_after)

    return {
        "slug": slug,
        "company": comp.name,
        "risk_before": round(risk_before, 4),
        "risk_after": round(risk_after, 4),
        "delta": round(risk_after - risk_before, 4),
        "pct_change": round(100 * (risk_after - risk_before) / max(risk_before, 1e-6), 1),
        "class_before": C.LABEL_SCHEME[int(probs_before.argmax())],
        "class_after": C.LABEL_SCHEME[int(probs_after.argmax())],
        "what_changed": (f"Simulated remediation: neutralized {n_removed} exploited-weakness "
                         f"entities from the breach and injected a mitigating control "
                         f"('{fix_phrase}') linked to {comp.name}."),
        "fix_phrase": fix_phrase,
    }


def main(slugs=None):
    import json
    from src.interpret.common import load_model
    from src.model.features import load_dataset

    device = C.get_device()
    model = load_model(device); model.eval()
    dataset = load_dataset()
    slugs = slugs or ["uber", "capital_one"]
    out = {}
    for slug in slugs:
        r = counterfactual(slug, model, dataset, device)
        out[slug] = r
        print(f"{r['company']:14s}: risk {r['risk_before']:.3f} -> {r['risk_after']:.3f} "
              f"(delta {r['delta']:+.3f}, {r['pct_change']:+.1f}%)  "
              f"[{r['class_before']} -> {r['class_after']}]")
    (C.ARTIFACTS / "counterfactual_results.json").write_text(json.dumps(out, indent=2),
                                                             encoding="utf-8")
    print(f"Saved -> {C.ARTIFACTS/'counterfactual_results.json'}")
    return out


if __name__ == "__main__":
    main()
