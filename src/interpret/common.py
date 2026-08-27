"""Shared helpers for interpretability + the assistant layer."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from src import config as C
from src.model.xfamha_gnn import dense_adj


def load_model(device=None):
    from src.model.xfamha_gnn import XFAMHAGNN
    return XFAMHAGNN.load_saved(C.ARTIFACTS / "model_full.pt", device=device)


def node_sources(slug: str) -> list[str]:
    """Source ('policy'/'attack') per node, in graph node order (== dataset order)."""
    G = nx.read_graphml(C.KG_DIR / f"{slug}.graphml")
    return [G.nodes[n].get("source", "policy") for n in G.nodes()]


@torch.no_grad()
def predict(model, data, device) -> tuple[int, np.ndarray]:
    x = data["x"].to(device)
    ei = data["edge_index"].to(device)
    logits = model(x, ei)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs


def masked_prob_fn(model, data, target: int, device):
    """Return f(masks)->prob(target): masks are (m, n) node-presence matrices.

    Absent nodes have their feature vectors zeroed and are excluded from pooling,
    so SHAP attributes the graph-level prediction to individual entities/nodes.
    """
    x = data["x"].to(device)
    adj = dense_adj(data["edge_index"].to(device), x.shape[0], device=device)

    @torch.no_grad()
    def f(masks: np.ndarray) -> np.ndarray:
        out = np.zeros(masks.shape[0], dtype=np.float32)
        for i, row in enumerate(masks):
            m = torch.as_tensor(row, dtype=x.dtype, device=device).unsqueeze(1)
            xm = x * m
            h = model.encode_dense(xm, adj)
            denom = m.sum().clamp(min=1.0)
            pooled = (h * m).sum(dim=0, keepdim=True) / denom
            logits = model.head(pooled)
            out[i] = F.softmax(logits, dim=1)[0, target].item()
        return out

    return f
