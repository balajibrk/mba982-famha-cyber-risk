"""Phase 4 - GATConv baseline (the paper's baseline (i)), trained identically.

Scope note: the full paper compares against 10 SOTA models. Per the hackathon
plan we validate against the most foundational graph-attention baseline (GATConv),
trained with the same features, LOO protocol, and metrics as X-FAMHA-GNN, plus the
majority-class baseline. Additional baselines (GIN, GraphSAGE) are future work.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from src import config as C
from src.model.features import D_IN, load_dataset
from src.model.xfamha_gnn import dense_adj  # reuse for symmetric self-looped edges
from src.train.common import majority_baseline, train_and_eval_loo

D_MODEL = 32
N_LAYERS = 3
HEADS = 4
EPOCHS = 150


class GATBaseline(nn.Module):
    def __init__(self, d_in: int, d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 heads: int = HEADS, n_classes: int = C.NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GATConv(d_model, d_model // heads, heads=heads,
                                      dropout=dropout, add_self_loops=True))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def _undirected(self, edge_index, n, device):
        if edge_index.numel() == 0:
            return torch.arange(n, device=device).repeat(2, 1)
        u, v = edge_index[0], edge_index[1]
        ei = torch.stack([torch.cat([u, v]), torch.cat([v, u])])
        return ei.to(device)

    def _encode(self, x, edge_index):
        ei = self._undirected(edge_index, x.shape[0], x.device)
        h = self.embed(x)
        for conv in self.convs:
            h = F.elu(conv(h, ei))
            h = self.drop(h)
        return h

    def forward(self, x, edge_index):
        h = self._encode(x, edge_index)
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.head(global_mean_pool(h, batch))

    def batch_logits(self, samples, device):
        xs, eis, batch, off = [], [], [], 0
        for gi, s in enumerate(samples):
            n = s["x"].shape[0]
            xs.append(s["x"].to(device))
            if s["edge_index"].numel() > 0:
                eis.append(s["edge_index"].to(device) + off)
            batch.append(torch.full((n,), gi, dtype=torch.long, device=device))
            off += n
        big_x = torch.cat(xs, dim=0)
        big_ei = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long, device=device)
        batch = torch.cat(batch, dim=0)
        h = self._encode(big_x, big_ei)
        return self.head(global_mean_pool(h, batch))


def build_gat(train_samples, device):
    return GATBaseline(D_IN).to(device)


def main():
    device = C.get_device()
    dataset = load_dataset()
    slugs = [c.slug for c in C.COMPANY_LIST]
    print(f"Device: {device} | GATConv baseline | {len(slugs)} companies")

    res = train_and_eval_loo(build_gat, dataset, slugs, epochs=EPOCHS,
                             seed=C.RANDOM_SEED, device=device, verbose=True)
    maj = majority_baseline(dataset, slugs)

    print("\n=== GATConv baseline (LOO) ===")
    for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "g_mean"):
        print(f"  {k:16s}: {res[k]:.4f}")
    print(f"  loss curve (fold 1): {res['first_loss_curve'][0]:.3f} -> "
          f"{res['first_loss_curve'][-1]:.3f}")

    out = {
        "gat": {k: res[k] for k in ("accuracy", "f1_macro", "precision_macro",
                                    "recall_macro", "g_mean")},
        "gat_predictions": {s: {"true": t, "pred": p} for s, t, p in
                            zip(slugs, res["y_true"], res["y_pred"])},
        "majority_baseline": {k: maj[k] for k in ("accuracy", "f1_macro",
                                                  "precision_macro", "recall_macro", "g_mean")},
        "loss_curve_fold1": res["first_loss_curve"],
    }
    (C.ARTIFACTS / "baseline_results.json").write_text(json.dumps(out, indent=2),
                                                       encoding="utf-8")
    print(f"\nSaved -> {C.ARTIFACTS/'baseline_results.json'}")


if __name__ == "__main__":
    main()
