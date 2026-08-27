"""Spot-check utility: print node/edge lists for given company KGs."""
import sys

import networkx as nx

from src import config as C


def show(slug: str) -> None:
    G = nx.read_graphml(C.KG_DIR / f"{slug}.graphml")
    print(f"===== {slug}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges =====")
    print("sample nodes (label :: type @ timestamp):")
    for n, d in list(G.nodes(data=True))[:10]:
        print(f"   {d.get('label')!r} :: {d.get('ntype')} @ {d.get('timestamp')}")
    print("sample edges (subject -[relation]-> object) [source]:")
    for u, v, d in list(G.edges(data=True))[:14]:
        su = G.nodes[u].get("label")
        ov = G.nodes[v].get("label")
        print(f"   {su!r} -[{d.get('relation')}]-> {ov!r}  [{d.get('source')}]")
    print()


if __name__ == "__main__":
    slugs = sys.argv[1:] or ["uber", "capital_one"]
    for s in slugs:
        show(s)
