"""X-FAMHA-GNN - the paper's predictive model (Section 3.2.3, "Model architecture").

Pipeline: fixed node features -> learnable graph-embedding layer -> a stack of N
identical blocks, each = FAMHA (message passing over neighbours) + a position-wise
feed-forward network, with ELU activation (the paper explicitly prefers ELU over
ReLU) and a residual connection -> global mean pooling -> 4-way soft-max head
(the paper's four risk classes).

The FAMHA structure (head count + factor partition) is determined once from the
projected features of the training nodes via ``build_structure`` before training,
giving stable learnable parameters while keeping the automatic, data-driven head
determination the paper describes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.famha import FAMHA


def dense_adj(edge_index: torch.Tensor, n: int, device=None) -> torch.Tensor:
    """Symmetric adjacency with self-loops (for neighbour message passing)."""
    A = torch.zeros(n, n, device=device)
    if edge_index.numel() > 0:
        u, v = edge_index[0], edge_index[1]
        A[u, v] = 1.0
        A[v, u] = 1.0
    A.fill_diagonal_(1.0)
    return A


class FAMHABlock(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int = 2, dropout: float = 0.1,
                 scale_div: float = 2.0, max_heads: int | None = None):
        super().__init__()
        self.famha = FAMHA(d_model, scale_div=scale_div, max_heads=max_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.famha(x, adj)                 # FAMHA attention aggregation
        h = F.elu(h)                           # paper: ELU after FAMHA
        x = self.norm1(x + self.drop(h))       # residual
        x = self.norm2(x + self.ffn(x))        # position-wise FFN + residual
        return x


class XFAMHAGNN(nn.Module):
    def __init__(self, d_in: int, d_model: int = 32, n_layers: int = 3,
                 n_classes: int = 4, dropout: float = 0.1,
                 scale_div: float = 2.0, max_heads: int | None = None):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Linear(d_in, d_model)     # learnable graph-embedding layer
        self.blocks = nn.ModuleList([
            FAMHABlock(d_model, dropout=dropout, scale_div=scale_div, max_heads=max_heads)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self._built = False

    # ------------------------------------------------------------------ #
    def build_structure(self, samples: list[dict]) -> None:
        """Determine every FAMHA block's head count + partition from data."""
        with torch.no_grad():
            feats = [self.embed(s["x"].to(self.embed.weight.device)) for s in samples]
            stacked = torch.cat(feats, dim=0)     # (total_nodes x d_model)
        for blk in self.blocks:
            blk.famha.fit_structure(stacked)
        self._built = True
        # newly created FAMHA projection params default to CPU; move to model device
        self.to(self.embed.weight.device)

    @classmethod
    def load_saved(cls, path, device=None):
        """Reconstruct a trained model (restoring FAMHA structure) from a checkpoint."""
        import torch as _t
        device = device or (_t.device("cuda") if _t.cuda.is_available() else _t.device("cpu"))
        blob = _t.load(path, weights_only=False)
        model = cls(d_in=blob["d_in"], d_model=blob["d_model"], n_layers=blob["n_layers"])
        for blk, groups in zip(model.blocks, blob["groups"]):
            blk.famha.set_structure(groups)
        model._built = True
        model.load_state_dict(blob["state_dict"])
        return model.to(device)

    def famha_param_counts(self) -> tuple[int, int]:
        """(theta_FAMHA, theta_normal) summed over blocks."""
        fam = sum(b.famha.param_count() for b in self.blocks)
        van = sum(b.famha.vanilla_param_count() for b in self.blocks)
        return fam, van

    def head_counts(self) -> list[int]:
        return [b.famha.h for b in self.blocks]

    # ------------------------------------------------------------------ #
    def encode_dense(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, adj)
        return h

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj = dense_adj(edge_index, x.shape[0], device=x.device)
        return self.encode_dense(x, adj)          # node embeddings (n x d_model)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if not self._built:
            raise RuntimeError("Call build_structure(...) before forward.")
        h = self.encode(x, edge_index)
        g = h.mean(dim=0, keepdim=True)           # global mean pooling
        return self.head(g)                       # (1 x n_classes) logits

    def batch_logits(self, samples: list[dict], device) -> torch.Tensor:
        """One forward over all graphs at once (block-diagonal adjacency).

        Equivalent to looping ``forward`` per graph but far faster: attention is
        adjacency-masked, so a block-diagonal adjacency prevents any cross-graph
        leakage, and pooling is done per graph.
        """
        if not self._built:
            raise RuntimeError("Call build_structure(...) before forward.")
        xs, batch, sizes = [], [], []
        for gi, s in enumerate(samples):
            xs.append(s["x"].to(device))
            n = s["x"].shape[0]
            sizes.append(n)
            batch.append(torch.full((n,), gi, dtype=torch.long, device=device))
        big_x = torch.cat(xs, dim=0)
        batch = torch.cat(batch, dim=0)
        total = big_x.shape[0]
        adj = torch.zeros(total, total, device=device)
        off = 0
        for gi, s in enumerate(samples):
            n = sizes[gi]
            ei = s["edge_index"]
            if ei.numel() > 0:
                u = ei[0].to(device) + off
                v = ei[1].to(device) + off
                adj[u, v] = 1.0
                adj[v, u] = 1.0
            off += n
        adj.fill_diagonal_(1.0)
        h = self.encode_dense(big_x, adj)
        # per-graph mean pooling
        ng = len(samples)
        pooled = torch.zeros(ng, h.shape[1], device=device)
        pooled.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=ng).clamp(min=1).unsqueeze(1).float()
        pooled = pooled / counts
        return self.head(pooled)                  # (num_graphs x n_classes)
