"""Phase 1b - cleaning & preprocessing pipeline (NLTK-based).

Replicates the paper's Phase 1 (Section 3.2.1): tokenize -> strip noise
(stopwords, URLs, emoji, repeated punctuation, contractions) -> spellcheck ->
lemmatize (WordNetLemmatizer, exactly as the paper specifies).

Because spaCy's compiled DLLs are blocked by Smart App Control on the target
machine, this uses NLTK, which is pure Python. NLTK also provides the exact
WordNetLemmatizer the paper cites, so this stage is arguably more faithful to
the paper than a spaCy lemmatizer would be.

Two outputs are produced per source file:
  * ``{slug}_{kind}_clean.txt``  - lightly cleaned, one sentence per line. This
    preserves sentence structure and is the input to the Phase 2 KG pipeline
    (coref / OIE need readable sentences).
  * ``{slug}_{kind}_lemmas.txt`` - the fully normalized (stopword-removed,
    lemmatized) token stream, demonstrating the paper's normalization step and
    used for word-count statistics.

Also writes ``docs/data_manifest.csv``.
"""

from __future__ import annotations

import csv
import json
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from symspellpy import SymSpell, Verbosity
import importlib.resources as ir

from src import config as C

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]",
    flags=re.UNICODE,
)
_MULTI_PUNCT_RE = re.compile(r"([!?.,;:])\1{1,}")
_WS_RE = re.compile(r"\s+")
_NONWORD_KEEP = re.compile(r"[^a-z0-9\-\s]")

CONTRACTIONS = {
    "don't": "do not", "can't": "cannot", "won't": "will not",
    "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
    "'ve": " have", "'d": " would", "'m": " am",
}

_LEMMATIZER = WordNetLemmatizer()
_STOP = set(stopwords.words("english"))


# --------------------------------------------------------------------------- #
# Conservative SymSpell spellchecker (protects domain vocabulary)
# --------------------------------------------------------------------------- #
def _build_symspell() -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    with ir.as_file(ir.files("symspellpy") / "frequency_dictionary_en_82_765.txt") as p:
        sym.load_dictionary(str(p), term_index=0, count_index=1)
    return sym


def _protected_vocab() -> set[str]:
    vocab: set[str] = set()
    for c in C.COMPANY_LIST:
        for a in c.aliases:
            vocab.update(w.lower() for w in a.split())
        for area in c.policy_areas:
            vocab.update(area.lower().split())
        for ent in c.attack_entities:
            vocab.update(ent.lower().split())
        vocab.update(c.name.lower().split())
        vocab.update(c.slug.split("_"))
    # common cyber terms we never want "corrected"
    vocab.update({
        "cybersecurity", "vpn", "mfa", "waf", "iam", "aws", "s3", "sql",
        "api", "ransomware", "phishing", "malware", "ssrf", "idor", "rasp",
        "cspm", "sbom", "dast", "zero", "trust", "backdoor", "subprocessor",
    })
    return vocab


_SYM = _build_symspell()
_PROTECTED = _protected_vocab()


def _spellcheck_token(tok: str) -> str:
    low = tok.lower()
    if not low.isalpha() or len(low) < 4 or low in _PROTECTED:
        return tok
    sugg = _SYM.lookup(low, Verbosity.CLOSEST, max_edit_distance=1, include_unknown=True)
    if sugg and sugg[0].term != low and sugg[0].distance > 0:
        return sugg[0].term
    return tok


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _strip_header(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def _normalize(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    text = _MULTI_PUNCT_RE.sub(r"\1", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def clean_sentences(raw: str) -> list[str]:
    """Return lightly cleaned sentences (readable, for the KG pipeline)."""
    text = _normalize(_strip_header(raw))
    out = []
    for sent in sent_tokenize(text):
        toks = word_tokenize(sent)
        toks = [_spellcheck_token(t) for t in toks]
        cleaned = _WS_RE.sub(" ", " ".join(toks)).strip()
        # tidy spacing before punctuation
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        if cleaned:
            out.append(cleaned)
    return out


def lemmatize_tokens(raw: str) -> list[str]:
    """Return the fully normalized token stream (paper-style)."""
    text = _normalize(_strip_header(raw)).lower()
    text = _NONWORD_KEEP.sub(" ", text)
    lemmas = []
    for tok in word_tokenize(text):
        if tok in _STOP or len(tok) < 2 or not any(ch.isalnum() for ch in tok):
            continue
        lemmas.append(_LEMMATIZER.lemmatize(tok))
    return lemmas


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run() -> None:
    labels = json.loads((C.LABELS_DIR / "labels.json").read_text(encoding="utf-8"))
    rows = []
    for c in C.COMPANY_LIST:
        for kind, raw_dir in (("policy", C.RAW_POLICIES), ("attack", C.RAW_ATTACKS)):
            raw = (raw_dir / f"{c.slug}.txt").read_text(encoding="utf-8")
            sents = clean_sentences(raw)
            lemmas = lemmatize_tokens(raw)

            clean_path = C.PROCESSED / f"{c.slug}_{kind}_clean.txt"
            lemma_path = C.PROCESSED / f"{c.slug}_{kind}_lemmas.txt"
            clean_path.write_text("\n".join(sents) + "\n", encoding="utf-8")
            lemma_path.write_text(" ".join(lemmas) + "\n", encoding="utf-8")

            raw_wc = len(_strip_header(raw).split())
            rows.append({
                "slug": c.slug,
                "name": c.name,
                "kind": kind,
                "source": "curated proof-of-concept (grounded in documented public facts)",
                "generated_date": "2026-08-25",
                "raw_word_count": raw_wc,
                "clean_sentence_count": len(sents),
                "clean_word_count": sum(len(s.split()) for s in sents),
                "lemma_count": len(lemmas),
                "label": labels[c.slug]["label"],
                "label_name": labels[c.slug]["label_name"],
            })

    manifest = C.DOCS / "data_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Cleaned {len(rows)} files -> {C.PROCESSED}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    run()
