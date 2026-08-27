"""Phase 2 (vi) - assemble per-company temporal knowledge graphs.

Ties together coref -> OIE -> NER -> canonicalization -> temporal labeling into a
networkx.DiGraph per company, exports to data/processed/kg/{slug}.graphml, and
writes docs/kg_stats.csv (mirrors the paper's descriptive-stats table).
"""

from __future__ import annotations

import csv
import re

import networkx as nx

from src import config as C
from src.kg import coref, ner
from src.kg.canonicalize import build_canonicalizer
from src.kg.oie import Triple, extract
from src.kg.temporal import company_timeline

_DET_RE = re.compile(r"^(the|a|an|all|its|their)\s+", flags=re.IGNORECASE)


def _norm_node(label: str) -> str:
    key = _DET_RE.sub("", label.strip().lower())
    key = re.sub(r"\s+", " ", key).strip(" .")
    return key or label.strip().lower()


def _company_ref_keys(comp) -> set[str]:
    """Normalized keys that should all collapse to the company's central node."""
    name_norm = _norm_node(comp.name)
    keys = {name_norm}
    for a in comp.aliases:
        an = _norm_node(a)
        if an and an not in {"company", "firm"}:
            keys.add(an)
    # leading name token (e.g. "capital" for "Capital One", "jpmorgan" for
    # "JPMorgan Chase") - recovers proper-noun fragments split by the chunker.
    first = name_norm.split()[0]
    if len(first) > 3:
        keys.add(first)
    return keys


def _load_triples(slug: str):
    """coref + OIE for a company's policy and attack text, tagged by source."""
    comp = C.COMPANIES_BY_SLUG[slug]
    tagged: list[tuple[Triple, str]] = []
    for kind in ("policy", "attack"):
        sents = (C.PROCESSED / f"{slug}_{kind}_clean.txt").read_text(
            encoding="utf-8").splitlines()
        resolved = coref.resolve(sents, comp)
        for t in extract(resolved):
            tagged.append((t, kind))
    return tagged


def build_all() -> None:
    # 1. extract triples for every company
    per_company = {c.slug: _load_triples(c.slug) for c in C.COMPANY_LIST}

    # 2. build one canonicalizer from the whole corpus (paper clusters globally)
    all_triples = [t for tagged in per_company.values() for (t, _) in tagged]
    print(f"Extracted {len(all_triples)} raw triples across "
          f"{len(per_company)} companies.")
    canon = build_canonicalizer(all_triples)

    # 3. per-company graph
    stats_rows = []
    relations_seen: set[str] = set()
    for c in C.COMPANY_LIST:
        tagged = per_company[c.slug]
        attack_text = (C.PROCESSED / f"{c.slug}_attack_clean.txt").read_text(encoding="utf-8")
        timeline = company_timeline(c, attack_text)

        G = nx.DiGraph(name=c.slug, company=c.name, label=c.label,
                       label_name=C.LABEL_SCHEME[c.label],
                       occurred_at=timeline["occurred_at"] or -1,
                       published_at=timeline["published_at"] or -1)

        # ensure the company node exists
        comp_key = _norm_node(c.name)
        ref_keys = _company_ref_keys(c)
        G.add_node(comp_key, label=c.name, ntype=ner.ORG, source="policy",
                   timestamp=timeline["published_at"] or -1)

        def node_key(raw: str) -> str:
            k = _norm_node(raw)
            return comp_key if k in ref_keys else k

        for (t, kind) in tagged:
            rel = canon.relation_of(t.verb)
            relations_seen.add(rel)
            ts = timeline["occurred_at"] if kind == "attack" else timeline["published_at"]
            ts = ts if ts is not None else -1
            for raw in (t.subject, t.object):
                key = node_key(raw)
                if not key:
                    continue
                if key not in G:
                    G.add_node(key, label=raw.strip(), ntype=ner.entity_type(raw),
                               source=kind, timestamp=ts)
            sk, ok = node_key(t.subject), node_key(t.object)
            if sk and ok and sk != ok:
                G.add_edge(sk, ok, relation=rel, raw_verb=t.verb,
                           source=kind, timestamp=ts)

        out = C.KG_DIR / f"{c.slug}.graphml"
        nx.write_graphml(G, out)

        ntypes = Counter_types(G)
        density = nx.density(G) if G.number_of_nodes() > 1 else 0.0
        stats_rows.append({
            "slug": c.slug, "name": c.name,
            "label": c.label, "label_name": C.LABEL_SCHEME[c.label],
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "density": round(density, 4),
            "n_org": ntypes.get(ner.ORG, 0), "n_policy": ntypes.get(ner.POLICY, 0),
            "n_attack": ntypes.get(ner.ATTACK, 0), "n_asset": ntypes.get(ner.ASSET, 0),
            "n_entity": ntypes.get(ner.ENTITY, 0),
            "occurred_at": timeline["occurred_at"] or "",
            "published_at": timeline["published_at"] or "",
        })

    # 4. stats table
    stats_path = C.DOCS / "kg_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    print(f"Built {len(stats_rows)} graphs -> {C.KG_DIR}")
    print(f"Relations used: {sorted(relations_seen)}")
    print(f"Stats: {stats_path}")


def Counter_types(G) -> dict:
    d: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        d[data.get("ntype", ner.ENTITY)] = d.get(data.get("ntype", ner.ENTITY), 0) + 1
    return d


if __name__ == "__main__":
    build_all()
