"""Phase 5 - FAMHA attention heatmaps.

Extracts the trained FAMHA layer's attention weights and renders a heatmap of
attention between policy-side entities (rows) and attack-side entities (columns),
mirroring the paper's Fig. 9(b) (e.g. their Tesla insider -> access-control
narrative). Here we use companies with a documented breach mechanism.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src import config as C
from src.interpret.common import load_model, node_sources
from src.model.features import load_dataset
from src.model.xfamha_gnn import dense_adj

TOP = 12


def attention_matrix(model, data, device) -> np.ndarray:
    """Mean block-0 FAMHA attention over heads, shape (n, n)."""
    x = data["x"].to(device)
    adj = dense_adj(data["edge_index"].to(device), x.shape[0], device=device)
    with torch.no_grad():
        h0 = model.embed(x)
        mats = model.blocks[0].famha.last_attention(h0, adj)  # list of (n,n)
    A = torch.stack(mats, dim=0).mean(dim=0).numpy()
    return A


def heatmap_company(slug: str, model, dataset, device) -> dict:
    data = dataset[slug]
    labels = data["node_labels"]
    sources = node_sources(slug)
    A = attention_matrix(model, data, device)

    pol = [i for i, s in enumerate(sources) if s == "policy"]
    atk = [i for i, s in enumerate(sources) if s == "attack"]
    if not atk:  # clean company: fall back to all nodes as columns
        atk = list(range(len(labels)))

    # pick the most-attended policy rows / attack cols
    sub = A[np.ix_(pol, atk)]
    row_rank = np.argsort(sub.sum(axis=1))[::-1][:TOP]
    col_rank = np.argsort(sub.sum(axis=0))[::-1][:TOP]
    rows = [pol[i] for i in row_rank]
    cols = [atk[j] for j in col_rank]
    M = A[np.ix_(rows, cols)]

    _plot(slug, M, [labels[i] for i in rows], [labels[j] for j in cols])

    # top attention pairs for the assistant layer
    pairs = []
    for i, ri in enumerate(rows):
        for j, cj in enumerate(cols):
            pairs.append((labels[ri], labels[cj], float(M[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return {"slug": slug,
            "top_attention_pairs": [{"policy_entity": a, "attack_entity": b,
                                     "weight": w} for a, b, w in pairs[:8]]}


def _plot(slug, M, rlabels, clabels):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(M, xticklabels=[c[:24] for c in clabels],
                yticklabels=[r[:24] for r in rlabels], cmap="rocket_r",
                ax=ax, cbar_kws={"label": "attention weight"})
    ax.set_xlabel("attack-article entities")
    ax.set_ylabel("policy entities")
    ax.set_title(f"{C.COMPANIES_BY_SLUG[slug].name} - FAMHA attention "
                 f"(policy x attack)", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(C.ARTIFACTS / f"attention_{slug}.png", dpi=140)
    plt.close(fig)


def main(slugs=None):
    device = C.get_device()
    model = load_model(device)
    model.eval()
    dataset = load_dataset()
    slugs = slugs or ["uber", "capital_one"]
    results = {}
    for slug in slugs:
        print(f"Attention heatmap {slug}...")
        results[slug] = heatmap_company(slug, model, dataset, device)
        for p in results[slug]["top_attention_pairs"][:4]:
            print(f"    {p['weight']:.3f}  {p['policy_entity']}  <->  {p['attack_entity']}")
    (C.ARTIFACTS / "attention_results.json").write_text(json.dumps(results, indent=2),
                                                        encoding="utf-8")
    print(f"Saved attention heatmaps + {C.ARTIFACTS/'attention_results.json'}")
    return results


if __name__ == "__main__":
    main()
