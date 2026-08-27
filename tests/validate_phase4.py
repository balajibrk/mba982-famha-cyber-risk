"""Phase 4 validation checks."""
import json
import sys

from src import config as C


def main() -> int:
    checks = []
    cs = json.loads((C.ARTIFACTS / "case_study_results.json").read_text(encoding="utf-8"))
    base = json.loads((C.ARTIFACTS / "baseline_results.json").read_text(encoding="utf-8"))
    sig = json.loads((C.ARTIFACTS / "significance.json").read_text(encoding="utf-8"))

    # 1. training loss decreases (no NaN, not flatlined)
    lc = cs["loss_curve_fold1"]
    import math
    finite = all(math.isfinite(v) for v in lc)
    decreased = lc[0] - lc[-1] > 0.1
    checks.append(("training loss decreases, no NaN/flatline "
                   f"({lc[0]:.3f}->{lc[-1]:.3f})", finite and decreased, ""))

    # 2. X-FAMHA beats majority-class baseline
    xf_acc = cs["xfamha"]["accuracy"]
    maj_acc = cs["majority_baseline"]["accuracy"]
    xf_f1 = cs["xfamha"]["f1_macro"]
    maj_f1 = cs["majority_baseline"]["f1_macro"]
    checks.append((f"X-FAMHA acc {xf_acc:.3f} > majority {maj_acc:.3f} "
                   f"(and F1 {xf_f1:.3f} > {maj_f1:.3f})",
                   xf_acc > maj_acc and xf_f1 > maj_f1, ""))

    # 3. baseline trained and produced a comparable metrics table
    gat_ok = all(k in base["gat"] for k in ("accuracy", "f1_macro"))
    checks.append((f"GATConv baseline trained (acc {base['gat']['accuracy']:.3f}, "
                   f"f1 {base['gat']['f1_macro']:.3f})", gat_ok, ""))

    # 4. Mann-Whitney U p-value reported
    p = sig["mann_whitney_u_pvalues_greater"]
    has_p = "xfamha_vs_gat_f1" in p and isinstance(p["xfamha_vs_gat_f1"], float)
    checks.append((f"Mann-Whitney U p-value reported "
                   f"(X-FAMHA vs GAT F1 p={p.get('xfamha_vs_gat_f1'):.4f})", has_p, ""))

    # 5. FAMHA param reduction recorded (Theorem 3.1)
    fam, van = cs["famha_params"], cs["vanilla_mha_params"]
    checks.append((f"FAMHA params {fam} < vanilla-MHA {van}", fam < van, ""))

    allpass = all(ok for _, ok, _ in checks)
    print(f"X-FAMHA-GNN: acc={xf_acc:.3f} f1={xf_f1:.3f} | "
          f"GAT: acc={base['gat']['accuracy']:.3f} f1={base['gat']['f1_macro']:.3f} | "
          f"majority: acc={maj_acc:.3f}\n")
    for name, ok, info in checks:
        print(("PASS" if ok else "FAIL"), "-", name, ("" if ok else ":: " + info))
    print("\nPHASE 4", "PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
