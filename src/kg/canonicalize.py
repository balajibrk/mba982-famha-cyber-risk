"""Phase 2 (iv) - canonicalized relation construction (verb clustering).

Faithful reproduction of the paper's canonicalization step (Section 3.2.2):

  1. For each triple <s, v, o> build a context string "s-lemma v-lemma o-lemma".
  2. Embed with Sentence-BERT (all-MiniLM-L6-v2, 384-d) - matches the paper.
  3. Agglomerative clustering, average linkage, cosine distance, threshold
     tau = 0.30 (cos-sim >= 0.70).
  4. Map each cluster to the paper's compact 8-relation set via majority-in-
     cluster using a small seed lexicon, falling back to nearest-canonical by
     embedding similarity when a cluster contains no seeded verb.

Runs the transformer on GPU in fp16 and frees VRAM afterwards.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src import config as C
from src.config import CANONICAL_RELATIONS
from src.kg.oie import Triple

TAU = 0.30  # cosine distance threshold (paper)

# --------------------------------------------------------------------------- #
# Seed lexicon: verb lemma -> canonical relation
# --------------------------------------------------------------------------- #
SEED_LEXICON: dict[str, str] = {}


def _seed(rel: str, verbs: str) -> None:
    for v in verbs.split():
        SEED_LEXICON[v] = rel


_seed("implements", "implement enforce apply deploy maintain require conduct operate "
                     "follow adopt use protect segment isolate control manage govern "
                     "restrict encrypt hash sign rotate monitor review test secure "
                     "limit verify purge disable harden filter defend collect practice "
                     "centralize run scan tune store issue perform validate")
_seed("mitigates", "mitigate remediate investigate respond contain strengthen patch "
                    "block detect prevent")
_seed("aligns-with", "align comply conform adhere meet")
_seed("violates", "violate breach expose leak fail weaken")
_seed("causes", "cause exploit enable allow let lead trigger weaponize insert subvert "
                "distribute combine base")
_seed("impacts", "impact affect force scrape steal obtain access reach pivot abuse "
                  "alter bypass crack unlock exfiltrate extend hush conceal grant "
                  "provide widen remain compromise hardcoded")
_seed("reports", "report disclose notify inform")
_seed("regulates", "regulate oversee")

# Descriptive phrases for canonical-relation fallback embeddings.
_REL_PHRASES = {
    "implements": "implements or enforces a security control",
    "aligns-with": "aligns with or complies with a standard",
    "violates": "violates or breaches, exposing data",
    "mitigates": "mitigates or remediates a threat",
    "causes": "causes or enables an attack",
    "impacts": "impacts or affects by accessing data",
    "reports": "reports or discloses an incident",
    "regulates": "regulates or oversees",
}


class Canonicalizer:
    def __init__(self, verb_to_rel: dict[str, str]):
        self.verb_to_rel = verb_to_rel

    def relation_of(self, verb: str) -> str:
        return self.verb_to_rel.get(verb, "impacts")

    def apply(self, triples: list[Triple]) -> list[tuple[str, str, str]]:
        return [(t.subject, self.relation_of(t.verb), t.object) for t in triples]


def _encode(texts: list[str]) -> np.ndarray:
    """Encode with MiniLM on GPU (fp16) then free VRAM."""
    from sentence_transformers import SentenceTransformer

    device = "cuda" if _cuda() else "cpu"
    model = SentenceTransformer(C.SBERT_MODEL, device=device)
    if device == "cuda":
        model = model.half()
    emb = model.encode(texts, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    C.free_gpu(model)
    return emb.astype(np.float32)


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def build_canonicalizer(all_triples: list[Triple]) -> Canonicalizer:
    """Cluster verbs across the whole corpus and map clusters to 8 relations."""
    # unique verbs with a representative context string (paper uses s-v-o context)
    rep_context: dict[str, str] = {}
    for t in all_triples:
        if t.verb not in rep_context:
            rep_context[t.verb] = f"{t.subject} {t.verb} {t.object}".lower()
    verbs = sorted(rep_context)
    if not verbs:
        return Canonicalizer({})

    ctx_emb = _encode([rep_context[v] for v in verbs])
    rel_emb = _encode([_REL_PHRASES[r] for r in CANONICAL_RELATIONS])

    # agglomerative clustering on cosine distance, average linkage, tau=0.30
    if len(verbs) == 1:
        labels = np.array([0])
    else:
        clu = AgglomerativeClustering(
            n_clusters=None, distance_threshold=TAU,
            metric="cosine", linkage="average",
        )
        labels = clu.fit_predict(ctx_emb)

    verb_to_rel: dict[str, str] = {}
    for cid in sorted(set(labels)):
        members = [verbs[i] for i in range(len(verbs)) if labels[i] == cid]
        votes = [SEED_LEXICON[v] for v in members if v in SEED_LEXICON]
        if votes:
            rel = Counter(votes).most_common(1)[0][0]
        else:
            # nearest canonical relation by centroid similarity
            idx = [i for i in range(len(verbs)) if labels[i] == cid]
            centroid = ctx_emb[idx].mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-9)
            rel = CANONICAL_RELATIONS[int(np.argmax(rel_emb @ centroid))]
        for v in members:
            # a seeded verb keeps its own seed mapping; unseeded inherit cluster rel
            verb_to_rel[v] = SEED_LEXICON.get(v, rel)

    n_clusters = len(set(labels))
    print(f"  canonicalize: {len(verbs)} unique verbs -> {n_clusters} clusters "
          f"-> {len(set(verb_to_rel.values()))} of 8 canonical relations used")
    return Canonicalizer(verb_to_rel)
