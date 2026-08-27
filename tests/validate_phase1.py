"""Phase 1 validation checks."""
import csv
import sys

from src import config as C


def main() -> int:
    rows = list(csv.DictReader(open(C.DOCS / "data_manifest.csv", encoding="utf-8")))
    checks = []

    missing = []
    for c in C.COMPANY_LIST:
        for kind in ("policy", "attack"):
            p = C.PROCESSED / f"{c.slug}_{kind}_clean.txt"
            if (not p.exists()) or p.stat().st_size == 0 or not p.read_text(encoding="utf-8").strip():
                missing.append(p.name)
    checks.append(("every company has non-empty clean policy+attack", not missing, f"missing={missing}"))

    checks.append(("manifest has 36 rows", len(rows) == 36, f"rows={len(rows)}"))

    zero = [r["slug"] + ":" + r["kind"] for r in rows
            if int(r["clean_word_count"]) == 0 or int(r["raw_word_count"]) == 0]
    checks.append(("no zero word counts", not zero, f"zero={zero}"))

    slugs = {r["slug"] for r in rows}
    checks.append(("all 18 companies in manifest", len(slugs) == 18, f"n={len(slugs)}"))

    allpass = all(ok for _, ok, _ in checks)
    for name, ok, info in checks:
        print(("PASS" if ok else "FAIL"), "-", name, ("" if ok else ":: " + info))

    raw = [int(r["raw_word_count"]) for r in rows]
    print(f"\nraw word-count range: {min(raw)}-{max(raw)}")
    print("PHASE 1", "PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
