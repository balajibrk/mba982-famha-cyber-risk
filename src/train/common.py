"""Shared training utilities: seeding, class weights, metrics, LOO runner.

Given the tiny sample (18 graphs), we use leave-one-out cross-validation (as the
plan specifies) rather than the paper's 10-fold x 5-run protocol. Metrics are
pooled over the 18 held-out predictions. This is illustrative, not statistically
powered like the paper's 190-company study - stated honestly in docs.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from src import config as C


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(labels: list[int], n_classes: int = C.NUM_CLASSES,
                  device=None) -> torch.Tensor:
    counts = Counter(labels)
    w = torch.tensor([1.0 / max(counts.get(k, 0), 1) for k in range(n_classes)],
                     dtype=torch.float32)
    w = w * n_classes / w.sum()
    return w.to(device) if device else w


def g_mean(y_true: list[int], y_pred: list[int], n_classes: int = C.NUM_CLASSES) -> float:
    """Geometric mean of per-class recall (paper reports G-mean)."""
    recalls = recall_score(y_true, y_pred, labels=list(range(n_classes)),
                           average=None, zero_division=0)
    recalls = np.clip(recalls, 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(recalls))))


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "g_mean": g_mean(y_true, y_pred),
    }


def majority_baseline(dataset: dict, slugs: list[str]) -> dict:
    """Leave-one-out majority-class baseline metrics."""
    y_true, y_pred = [], []
    for held in slugs:
        train_labels = [dataset[s]["y"] for s in slugs if s != held]
        maj = Counter(train_labels).most_common(1)[0][0]
        y_true.append(dataset[held]["y"])
        y_pred.append(maj)
    m = metrics(y_true, y_pred)
    m["_name"] = "majority_class"
    return m


def train_and_eval_loo(build_fn, dataset: dict, slugs: list[str], *,
                       epochs: int = 150, lr: float = 0.01, weight_decay: float = 5e-4,
                       seed: int = 0, device=None, verbose: bool = False) -> dict:
    """Leave-one-out CV. ``build_fn(train_samples, device)`` returns a built model.

    Returns dict with pooled y_true/y_pred, metrics, per-fold correctness, and the
    first-fold loss curve (for the 'loss decreases' validation check).
    """
    device = device or C.get_device()
    set_seed(seed)
    y_true, y_pred, per_fold = [], [], []
    first_loss_curve = []

    for fi, held in enumerate(slugs):
        train_samples = [dataset[s] for s in slugs if s != held]
        train_labels = [s["y"] for s in train_samples]
        w = class_weights(train_labels, device=device)

        model = build_fn(train_samples, device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_y = torch.tensor([s["y"] for s in train_samples], device=device)

        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            logits = model.batch_logits(train_samples, device)  # (num_graphs x C)
            loss = F.cross_entropy(logits, train_y, weight=w)
            loss.backward()
            opt.step()
            if fi == 0:
                first_loss_curve.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            s = dataset[held]
            logits = model(s["x"].to(device), s["edge_index"].to(device))
            pred = int(logits.argmax(dim=1).item())
        y_true.append(dataset[held]["y"])
        y_pred.append(pred)
        per_fold.append(int(pred == dataset[held]["y"]))
        if verbose:
            print(f"  fold {fi+1:2d}/{len(slugs)} held={held:16s} "
                  f"true={dataset[held]['y']} pred={pred} "
                  f"{'OK' if pred==dataset[held]['y'] else 'x'}")

    m = metrics(y_true, y_pred)
    m.update({"y_true": y_true, "y_pred": y_pred, "per_fold": per_fold,
              "first_loss_curve": first_loss_curve})
    return m
