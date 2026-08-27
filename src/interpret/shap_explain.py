"""Phase 5 - SHAP interpretability (model-agnostic KernelExplainer).

Explains a company's graph-level risk prediction in terms of the individual
entities (nodes) that push it toward higher/lower vulnerability, mirroring the
paper's Fig. 9(a). Uses shap.KernelExplainer over a node-presence mask, which is
model-agnostic and works regardless of the GNN internals.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src import config as C
from src.interpret.common import load_model, masked_prob_fn, predict
from src.model.features import load_dataset

TOP_K = 10
NSAMPLES = 300


def explain_company(slug: str, model, dataset, device) -> dict:
    data = dataset[slug]
    labels = data["node_labels"]
    n = len(labels)
    pred, probs = predict(model, data, device)

    f = masked_prob_fn(model, data, target=pred, device=device)
    background = np.zeros((1, n), dtype=np.float32)     # "nothing present"
    explainer = shap.KernelExplainer(f, background)
    instance = np.ones((1, n), dtype=np.float32)        # full graph
    sv = explainer.shap_values(instance, nsamples=NSAMPLES, silent=True)
    sv = np.array(sv).reshape(-1)

    order = np.argsort(np.abs(sv))[::-1][:TOP_K]
    top = [{"entity": labels[i], "shap": float(sv[i]),
            "type": data["node_types"][i]} for i in order]

    _plot(slug, pred, probs, top)
    return {"slug": slug, "pred_class": pred,
            "pred_label": C.LABEL_SCHEME[pred],
            "pred_prob": float(probs[pred]),
            "top_entities": top}


def _plot(slug, pred, probs, top):
    ents = [t["entity"][:34] for t in top][::-1]
    vals = [t["shap"] for t in top][::-1]
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(ents)), vals, color=colors)
    ax.set_yticks(range(len(ents)), labels=ents, fontsize=8)
    ax.set_xlabel("SHAP value (-> higher predicted-class probability)")
    ax.set_title(f"{C.COMPANIES_BY_SLUG[slug].name} - top entities for "
                 f"'{C.LABEL_SCHEME[pred]}' (p={probs[pred]:.2f})", fontsize=10)
    ax.axvline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(C.ARTIFACTS / f"shap_{slug}.png", dpi=140)
    plt.close(fig)


def main(slugs=None):
    device = C.get_device()
    model = model_eval(load_model(device))
    dataset = load_dataset()
    slugs = slugs or ["uber", "capital_one"]
    results = {}
    for slug in slugs:
        print(f"SHAP explaining {slug}...")
        results[slug] = explain_company(slug, model, dataset, device)
        top = results[slug]["top_entities"][:5]
        for t in top:
            print(f"    {t['shap']:+.4f}  {t['entity']} ({t['type']})")
    (C.ARTIFACTS / "shap_results.json").write_text(json.dumps(results, indent=2),
                                                   encoding="utf-8")
    print(f"Saved SHAP charts + {C.ARTIFACTS/'shap_results.json'}")
    return results


def model_eval(model):
    model.eval()
    return model


if __name__ == "__main__":
    main()
