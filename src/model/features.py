"""Phase 3 - build fixed node-feature tensors for every company graph.

Per the paper's "Graph Embedding Construction", each node gets a fixed feature
vector from its observable attributes:
  * node-type one-hot        (5 dims: ORG/POLICY/ATTACK/ASSET/ENTITY)
  * text embedding of label  (384 dims, Sentence-BERT all-MiniLM-L6-v2)
  * temporal feature         (1 dim: breach/publish year, globally normalized)

These fixed vectors are later passed through a *learnable* embedding layer inside
the model (updated by backprop), exactly as the paper specifies. The MiniLM pass
runs once on GPU in fp16, then frees VRAM.

Saves artifacts/dataset.pt: {slug: {x, edge_index, y, node_labels, node_types, ts}}.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import torch

from src import config as C
from src.kg import ner

NODE_TYPES = [ner.ORG, ner.POLICY, ner.ATTACK, ner.ASSET, ner.ENTITY]
TYPE_INDEX = {t: i for i, t in enumerate(NODE_TYPES)}

YEAR_MIN, YEAR_MAX = 2008, 2023
D_TEXT = C.SBERT_DIM
D_IN = len(NODE_TYPES) + D_TEXT + 1  # 5 + 384 + 1 = 390


def _norm_year(ts) -> float:
    try:
        y = int(ts)
    except (TypeError, ValueError):
        return 0.5
    if y < 0:
        return 0.5
    return float(np.clip((y - YEAR_MIN) / (YEAR_MAX - YEAR_MIN), 0.0, 1.0))


def _encode_labels(labels: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(C.SBERT_MODEL, device=device)
    if device == "cuda":
        model = model.half()
    emb = model.encode(labels, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False, batch_size=256)
    C.free_gpu(model)
    return emb.astype(np.float32)


def build_dataset(save: bool = True) -> dict:
    graphs = {c.slug: nx.read_graphml(C.KG_DIR / f"{c.slug}.graphml")
              for c in C.COMPANY_LIST}

    # 1. collect every node label across all graphs, encode once
    all_labels, index = [], {}
    for slug, G in graphs.items():
        for n, d in G.nodes(data=True):
            lab = d.get("label", n)
            if lab not in index:
                index[lab] = len(all_labels)
                all_labels.append(lab)
    text_emb = _encode_labels(all_labels)
    print(f"Encoded {len(all_labels)} unique node labels -> {text_emb.shape}")

    # 2. per-graph feature tensors
    dataset = {}
    for slug, G in graphs.items():
        comp = C.COMPANIES_BY_SLUG[slug]
        nodes = list(G.nodes())
        nidx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        x = np.zeros((n, D_IN), dtype=np.float32)
        node_labels, node_types, node_ts = [], [], []
        for i, node in enumerate(nodes):
            d = G.nodes[node]
            lab = d.get("label", node)
            ntype = d.get("ntype", ner.ENTITY)
            ts = d.get("timestamp", -1)
            x[i, TYPE_INDEX.get(ntype, TYPE_INDEX[ner.ENTITY])] = 1.0
            x[i, len(NODE_TYPES):len(NODE_TYPES) + D_TEXT] = text_emb[index[lab]]
            x[i, -1] = _norm_year(ts)
            node_labels.append(lab)
            node_types.append(ntype)
            node_ts.append(ts)

        # directed edges from the graph
        if G.number_of_edges() > 0:
            ei = np.array([[nidx[u], nidx[v]] for u, v in G.edges()], dtype=np.int64).T
        else:
            ei = np.zeros((2, 0), dtype=np.int64)

        dataset[slug] = {
            "x": torch.from_numpy(x),
            "edge_index": torch.from_numpy(ei),
            "y": int(comp.label),
            "node_labels": node_labels,
            "node_types": node_types,
            "ts": node_ts,
        }

    if save:
        C.ARTIFACTS.mkdir(exist_ok=True)
        torch.save({"dataset": dataset, "d_in": D_IN, "node_types": NODE_TYPES},
                   C.ARTIFACTS / "dataset.pt")
        print(f"Saved dataset -> {C.ARTIFACTS / 'dataset.pt'} (d_in={D_IN})")
    return dataset


def load_dataset() -> dict:
    blob = torch.load(C.ARTIFACTS / "dataset.pt", weights_only=False)
    return blob["dataset"]


if __name__ == "__main__":
    build_dataset()
