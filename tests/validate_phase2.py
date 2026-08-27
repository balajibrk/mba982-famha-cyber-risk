"""Phase 2 validation checks."""
import csv
import sys

import networkx as nx

from src import config as C
from src.config import CANONICAL_RELATIONS


def main() -> int:
    checks = []

    # every company has a non-trivial graphml (>0 nodes, >0 edges)
    bad = []
    graphs = {}
    for c in C.COMPANY_LIST:
        p = C.KG_DIR / f"{c.slug}.graphml"
        if not p.exists():
            bad.append(c.slug + ":missing")
            continue
        G = nx.read_graphml(p)
        graphs[c.slug] = G
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            bad.append(f"{c.slug}:n={G.number_of_nodes()},e={G.number_of_edges()}")
    checks.append(("every company has non-trivial graphml (>0 nodes/edges)", not bad, f"bad={bad}"))

    # kg_stats.csv populated, no zero rows
    rows = list(csv.DictReader(open(C.DOCS / "kg_stats.csv", encoding="utf-8")))
    zero = [r["slug"] for r in rows if int(r["nodes"]) == 0 or int(r["edges"]) == 0]
    checks.append(("kg_stats.csv has 18 rows, none zero", len(rows) == 18 and not zero,
                   f"rows={len(rows)} zero={zero}"))

    # relations limited to the 8-type set
    leaked = set()
    canon = set(CANONICAL_RELATIONS)
    for slug, G in graphs.items():
        for _, _, d in G.edges(data=True):
            if d.get("relation") not in canon:
                leaked.add(d.get("relation"))
    checks.append(("all edge relations within 8-type canonical set", not leaked, f"leaked={leaked}"))

    # temporal attributes present on nodes
    no_ts = [s for s, G in graphs.items()
             if not all("timestamp" in d for _, d in G.nodes(data=True))]
    checks.append(("all nodes carry a temporal timestamp attribute", not no_ts, f"missing={no_ts}"))

    allpass = all(ok for _, ok, _ in checks)
    for name, ok, info in checks:
        print(("PASS" if ok else "FAIL"), "-", name, ("" if ok else ":: " + info))

    ncount = [int(r["nodes"]) for r in rows]
    ecount = [int(r["edges"]) for r in rows]
    print(f"\nnodes range {min(ncount)}-{max(ncount)}, edges range {min(ecount)}-{max(ecount)}")
    print("PHASE 2", "PASS" if allpass else "FAIL")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
