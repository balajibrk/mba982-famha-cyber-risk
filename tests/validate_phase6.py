"""Phase 6 validation: real counterfactual delta, grounded JSON narrative,
and one full end-to-end company run with no mid-run intervention."""
import json
import sys

from src import config as C
from src.assistant.narrative import SCHEMA_KEYS


def main() -> int:
    checks = []
    assistant = json.loads((C.ARTIFACTS / "assistant_results.json").read_text(encoding="utf-8"))

    # at least one company ran end to end
    checks.append(("end-to-end assistant produced >=1 company result",
                   len(assistant) >= 1, ""))

    for slug, r in assistant.items():
        cf = r["counterfactual"]
        # 1. counterfactual delta is a real, model-computed number (nonzero)
        checks.append((f"{slug}: nonzero model-computed counterfactual delta "
                       f"({cf['risk_before']:.3f}->{cf['risk_after']:.3f}, "
                       f"delta {cf['delta']:+.3f})", abs(cf["delta"]) > 1e-6, ""))

        # 2. narrative JSON conforms to schema
        narr = r["narrative"]
        conforms = all(k in narr and isinstance(narr[k], str) and narr[k].strip()
                       for k in SCHEMA_KEYS)
        checks.append((f"{slug}: narrative JSON has all keys {SCHEMA_KEYS}", conforms, ""))

        # 3. no invented numbers flagged
        checks.append((f"{slug}: no ungrounded numbers in narrative",
                       not r["ungrounded_number_warnings"],
                       str(r["ungrounded_number_warnings"])))

        # 4. exec + engineer views generated
        checks.append((f"{slug}: exec summary + engineer ticket generated",
                       bool(r["exec_summary"]) and bool(r["engineer_ticket"]), ""))

    allpass = all(ok for _, ok, _ in checks)
    for name, ok, info in checks:
        print(("PASS" if ok else "FAIL"), "-", name, ("" if ok else ":: " + info))
    src = next(iter(assistant.values()))["narrative_source"]
    print(f"\nnarrative source: {src}")
    print("PHASE 6", "PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
