"""Phase 4 - statistical significance (Mann-Whitney U) + results table + heatmap.

The paper compares 50 outcomes (10-fold x 5-run) per model with a Mann-Whitney U
test. Our analog: run leave-one-out CV under several random seeds for each model,
collect the per-seed macro-F1 (and accuracy), and test whether X-FAMHA-GNN's
scores stochastically dominate the GATConv baseline's. Honest caveat: with a tiny
sample and only a handful of seeds this is illustrative, not powered.
"""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from src import config as C
from src.baselines.run_baselines import build_gat
from src.model.features import load_dataset
from src.train.common import majority_baseline, train_and_eval_loo
from src.train.train_case_study import build_xfamha

N_SEEDS = 5
EPOCHS = 120


def _run_seeds(build_fn, dataset, slugs, device):
    accs, f1s = [], []
    for seed in range(N_SEEDS):
        r = train_and_eval_loo(build_fn, dataset, slugs, epochs=EPOCHS,
                               seed=seed, device=device)
        accs.append(r["accuracy"])
        f1s.append(r["f1_macro"])
        print(f"    seed {seed}: acc={r['accuracy']:.3f} f1={r['f1_macro']:.3f}")
    return np.array(accs), np.array(f1s)


def main():
    device = C.get_device()
    dataset = load_dataset()
    slugs = [c.slug for c in C.COMPANY_LIST]

    print(f"Running {N_SEEDS}-seed LOO for X-FAMHA-GNN...")
    xf_acc, xf_f1 = _run_seeds(build_xfamha, dataset, slugs, device)
    print(f"Running {N_SEEDS}-seed LOO for GATConv...")
    gat_acc, gat_f1 = _run_seeds(build_gat, dataset, slugs, device)

    maj = majority_baseline(dataset, slugs)
    maj_acc = np.full(N_SEEDS, maj["accuracy"])
    maj_f1 = np.full(N_SEEDS, maj["f1_macro"])

    def mwu(a, b):
        try:
            return float(mannwhitneyu(a, b, alternative="greater").pvalue)
        except ValueError:
            return 1.0

    p = {
        "xfamha_vs_gat_acc": mwu(xf_acc, gat_acc),
        "xfamha_vs_gat_f1": mwu(xf_f1, gat_f1),
        "xfamha_vs_majority_acc": mwu(xf_acc, maj_acc),
        "xfamha_vs_majority_f1": mwu(xf_f1, maj_f1),
    }

    sig = {
        "n_seeds": N_SEEDS, "epochs": EPOCHS,
        "xfamha": {"acc_mean": float(xf_acc.mean()), "acc_std": float(xf_acc.std()),
                   "f1_mean": float(xf_f1.mean()), "f1_std": float(xf_f1.std()),
                   "acc_scores": xf_acc.tolist(), "f1_scores": xf_f1.tolist()},
        "gat": {"acc_mean": float(gat_acc.mean()), "acc_std": float(gat_acc.std()),
                "f1_mean": float(gat_f1.mean()), "f1_std": float(gat_f1.std()),
                "acc_scores": gat_acc.tolist(), "f1_scores": gat_f1.tolist()},
        "majority": {"acc": maj["accuracy"], "f1": maj["f1_macro"]},
        "mann_whitney_u_pvalues_greater": p,
    }
    (C.ARTIFACTS / "significance.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
    print("\nMann-Whitney U (one-sided 'X-FAMHA greater') p-values:")
    for k, v in p.items():
        print(f"  {k:28s}: {v:.4f}")

    _plot_heatmap(p)
    _write_results_table(sig)
    print(f"\nSaved -> {C.ARTIFACTS/'significance.json'}, "
          f"{C.ARTIFACTS/'pvalue_heatmap.png'}, {C.DOCS/'results.csv'}")


def _plot_heatmap(p: dict):
    labels_row = ["X-FAMHA vs GAT", "X-FAMHA vs Majority"]
    labels_col = ["accuracy", "F1"]
    M = np.array([[p["xfamha_vs_gat_acc"], p["xfamha_vs_gat_f1"]],
                  [p["xfamha_vs_majority_acc"], p["xfamha_vs_majority_f1"]]])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    im = ax.imshow(M, cmap="viridis_r", vmin=0, vmax=0.2)
    ax.set_xticks(range(len(labels_col)), labels=labels_col)
    ax.set_yticks(range(len(labels_row)), labels=labels_row)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.1 else "black")
    ax.set_title("Mann-Whitney U p-values (alpha=0.05)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(C.ARTIFACTS / "pvalue_heatmap.png", dpi=140)
    plt.close(fig)


def _write_results_table(sig: dict):
    rows = [
        {"model": "X-FAMHA-GNN", "accuracy": f"{sig['xfamha']['acc_mean']:.4f}",
         "f1_macro": f"{sig['xfamha']['f1_mean']:.4f}",
         "acc_std": f"{sig['xfamha']['acc_std']:.4f}"},
        {"model": "GATConv", "accuracy": f"{sig['gat']['acc_mean']:.4f}",
         "f1_macro": f"{sig['gat']['f1_mean']:.4f}",
         "acc_std": f"{sig['gat']['acc_std']:.4f}"},
        {"model": "Majority-class", "accuracy": f"{sig['majority']['acc']:.4f}",
         "f1_macro": f"{sig['majority']['f1']:.4f}", "acc_std": "0.0000"},
    ]
    with (C.DOCS / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "accuracy", "f1_macro", "acc_std"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
