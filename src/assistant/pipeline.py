"""Phase 6 - end-to-end assistant pipeline for one company.

Ties everything together with no manual intervention:
  KG -> risk score -> SHAP + attention -> counterfactual re-score -> grounded
  narrative (exec + engineer views). Returns a single evidence-linked object.
"""

from __future__ import annotations

import json

from src import config as C
from src.assistant.counterfactual import counterfactual
from src.assistant.narrative import (engineer_ticket, exec_summary, generate)
from src.interpret.attention_heatmap import heatmap_company
from src.interpret.common import load_model, predict
from src.interpret.shap_explain import explain_company
from src.model.features import load_dataset


def run_company(slug: str, model=None, dataset=None, device=None) -> dict:
    device = device or C.get_device()
    model = model or load_model(device)
    model.eval()
    dataset = dataset or load_dataset()

    data = dataset[slug]
    pred, probs = predict(model, data, device)

    shap_res = explain_company(slug, model, dataset, device)
    attn_res = heatmap_company(slug, model, dataset, device)
    cf = counterfactual(slug, model, dataset, device)

    narr = generate(
        company=C.COMPANIES_BY_SLUG[slug].name,
        pred_label=C.LABEL_SCHEME[pred],
        pred_prob=float(probs[pred]),
        top_shap=shap_res["top_entities"],
        top_attention=attn_res["top_attention_pairs"],
        counterfactual=cf,
    )

    result = {
        "slug": slug,
        "company": C.COMPANIES_BY_SLUG[slug].name,
        "true_label": C.LABEL_SCHEME[data["y"]],
        "pred_label": C.LABEL_SCHEME[pred],
        "pred_prob": float(probs[pred]),
        "probs": {C.LABEL_SCHEME[i]: float(probs[i]) for i in range(len(probs))},
        "top_shap": shap_res["top_entities"][:5],
        "top_attention": attn_res["top_attention_pairs"][:5],
        "counterfactual": cf,
        "narrative": narr["narrative"],
        "narrative_source": narr["source"],
        "ungrounded_number_warnings": narr["ungrounded_number_warnings"],
        "exec_summary": exec_summary(narr),
        "engineer_ticket": engineer_ticket(narr),
        "artifacts": {
            "shap_png": str((C.ARTIFACTS / f"shap_{slug}.png")),
            "attention_png": str((C.ARTIFACTS / f"attention_{slug}.png")),
        },
    }
    return result


def main(slugs=None):
    device = C.get_device()
    model = load_model(device); model.eval()
    dataset = load_dataset()
    slugs = slugs or ["uber"]
    out = {}
    for slug in slugs:
        print(f"\n===== END-TO-END: {slug} =====")
        r = run_company(slug, model, dataset, device)
        out[slug] = r
        print(f"  true={r['true_label']}  pred={r['pred_label']} ({r['pred_prob']:.2f})")
        print(f"  counterfactual: {r['counterfactual']['risk_before']:.3f} -> "
              f"{r['counterfactual']['risk_after']:.3f} "
              f"({r['counterfactual']['pct_change']:+.1f}%)")
        print(f"  narrative source: {r['narrative_source']}")
        print(f"  verdict: {r['narrative']['one_line_verdict']}")
        print(f"  why    : {r['narrative']['why']}")
        print(f"  fix    : {r['narrative']['fix']}")
        print(f"  impact : {r['narrative']['impact']}")
        if r["ungrounded_number_warnings"]:
            print(f"  WARNING ungrounded numbers: {r['ungrounded_number_warnings']}")
    (C.ARTIFACTS / "assistant_results.json").write_text(json.dumps(out, indent=2),
                                                        encoding="utf-8")
    print(f"\nSaved -> {C.ARTIFACTS/'assistant_results.json'}")
    return out


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
