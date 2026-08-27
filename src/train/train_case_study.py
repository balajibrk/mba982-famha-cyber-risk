"""Phase 4 - train X-FAMHA-GNN with leave-one-out CV on the case-study set.

Also trains a final model on all 18 companies and saves it (with its FAMHA
structure) for the Phase 5 interpretability and Phase 6 assistant stages.
"""

from __future__ import annotations

import json

import torch

from src import config as C
from src.model.features import D_IN, load_dataset
from src.model.xfamha_gnn import XFAMHAGNN
from src.train.common import (majority_baseline, metrics, set_seed,
                              train_and_eval_loo)

D_MODEL = 32
N_LAYERS = 3
EPOCHS = 150


def build_xfamha(train_samples, device):
    model = XFAMHAGNN(d_in=D_IN, d_model=D_MODEL, n_layers=N_LAYERS).to(device)
    model.build_structure([{"x": s["x"]} for s in train_samples])
    return model


def train_final_model(dataset, slugs, device, seed=C.RANDOM_SEED):
    """Train on all companies and persist for downstream phases."""
    from src.train.common import class_weights
    import torch.nn.functional as F

    set_seed(seed)
    samples = [dataset[s] for s in slugs]
    w = class_weights([s["y"] for s in samples], device=device)
    model = build_xfamha(samples, device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    ys = torch.tensor([s["y"] for s in samples], device=device)
    model.train()
    for ep in range(EPOCHS):
        opt.zero_grad()
        logits = model.batch_logits(samples, device)
        F.cross_entropy(logits, ys, weight=w).backward()
        opt.step()
    torch.save({
        "state_dict": model.state_dict(),
        "d_in": D_IN, "d_model": D_MODEL, "n_layers": N_LAYERS,
        "groups": [[g.tolist() for g in blk.famha.groups] for blk in model.blocks],
        "head_counts": model.head_counts(),
    }, C.ARTIFACTS / "model_full.pt")
    print(f"Saved final model -> {C.ARTIFACTS/'model_full.pt'} "
          f"(head counts {model.head_counts()})")
    return model


def main():
    device = C.get_device()
    dataset = load_dataset()
    slugs = [c.slug for c in C.COMPANY_LIST]
    print(f"Device: {device} | {len(slugs)} companies | d_in={D_IN}")

    print("\nLeave-one-out CV (X-FAMHA-GNN)...")
    res = train_and_eval_loo(build_xfamha, dataset, slugs,
                             epochs=EPOCHS, seed=C.RANDOM_SEED, device=device,
                             verbose=True)

    maj = majority_baseline(dataset, slugs)
    fam_c, van_c = build_xfamha([dataset[s] for s in slugs], device).famha_param_counts()

    print("\n=== X-FAMHA-GNN (LOO) ===")
    for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "g_mean"):
        print(f"  {k:16s}: {res[k]:.4f}")
    print(f"\n  majority-class baseline accuracy: {maj['accuracy']:.4f}  "
          f"f1_macro: {maj['f1_macro']:.4f}")
    print(f"  loss curve (fold 1): {res['first_loss_curve'][0]:.3f} -> "
          f"{res['first_loss_curve'][-1]:.3f}")
    print(f"  FAMHA params {fam_c} vs vanilla-MHA {van_c}")

    out = {
        "xfamha": {k: res[k] for k in ("accuracy", "f1_macro", "precision_macro",
                                       "recall_macro", "g_mean")},
        "xfamha_predictions": {s: {"true": t, "pred": p} for s, t, p in
                               zip(slugs, res["y_true"], res["y_pred"])},
        "majority_baseline": {k: maj[k] for k in ("accuracy", "f1_macro",
                                                  "precision_macro", "recall_macro", "g_mean")},
        "loss_curve_fold1": res["first_loss_curve"],
        "famha_params": fam_c, "vanilla_mha_params": van_c,
    }
    (C.ARTIFACTS / "case_study_results.json").write_text(json.dumps(out, indent=2),
                                                         encoding="utf-8")
    print(f"\nSaved -> {C.ARTIFACTS/'case_study_results.json'}")

    print("\nTraining final model on all companies...")
    train_final_model(dataset, slugs, device)


if __name__ == "__main__":
    main()
