"""Phase 5 validation: SHAP + attention artifacts for >=2 companies, with a
sanity check that top entities relate to the company's known breach."""
import json
import sys

from src import config as C

COMPANIES = ["uber", "capital_one"]


def main() -> int:
    checks = []

    shap_res = json.loads((C.ARTIFACTS / "shap_results.json").read_text(encoding="utf-8"))
    attn_res = json.loads((C.ARTIFACTS / "attention_results.json").read_text(encoding="utf-8"))

    for slug in COMPANIES:
        png_shap = C.ARTIFACTS / f"shap_{slug}.png"
        png_attn = C.ARTIFACTS / f"attention_{slug}.png"
        checks.append((f"{slug}: SHAP chart saved", png_shap.exists(), ""))
        checks.append((f"{slug}: attention heatmap saved", png_attn.exists(), ""))
        checks.append((f"{slug}: SHAP produced >=3 ranked entities",
                       len(shap_res.get(slug, {}).get("top_entities", [])) >= 3, ""))
        checks.append((f"{slug}: attention produced >=1 policy-attack pair",
                       len(attn_res.get(slug, {}).get("top_attention_pairs", [])) >= 1, ""))

    # sanity: at least one top-SHAP or attention entity overlaps the company's
    # documented attack/policy vocabulary
    sane = []
    for slug in COMPANIES:
        comp = C.COMPANIES_BY_SLUG[slug]
        vocab = " ".join(comp.attack_entities + comp.policy_areas).lower()
        ents = [e["entity"].lower() for e in shap_res[slug]["top_entities"]]
        pairs = attn_res[slug]["top_attention_pairs"]
        pair_ents = " ".join(p["policy_entity"] + " " + p["attack_entity"]
                             for p in pairs).lower()
        hit = any(any(w in vocab for w in e.split()) for e in ents) or \
              any(w in vocab for w in pair_ents.split())
        sane.append((slug, hit))
    checks.append((f"top entities relate to known breach vocabulary "
                   f"({[s for s,h in sane if h]})", any(h for _, h in sane), str(sane)))

    allpass = all(ok for _, ok, _ in checks)
    for name, ok, info in checks:
        print(("PASS" if ok else "FAIL"), "-", name, ("" if ok else ":: " + info))
    print("\nPHASE 5", "PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
