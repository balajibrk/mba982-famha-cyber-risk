"""Phase 2 (ii) - Open Information Extraction (SVO triples).

The paper uses Stanford OpenIE. To avoid a JVM dependency (and because spaCy's
dependency parser is blocked by Smart App Control on this machine), we use a
lightweight NLTK POS-tag + noun-phrase-chunk SVO extractor. Our curated text is
written with clear subject-verb-object structure, so this recovers meaningful
triples reliably at this scale.

A triple is (subject_phrase, verb_lemma, object_phrase).
"""

from __future__ import annotations

from dataclasses import dataclass

import nltk
from nltk import RegexpParser
from nltk.stem import WordNetLemmatizer

_LEMM = WordNetLemmatizer()

# Noun-phrase grammar: proper-noun runs OR (det? adj* noun+).
_GRAMMAR = r"""
  NP: {<NNP|NNPS>+}
      {<DT>?<JJ.*|VBG>*<NN.*>+}
"""
_PARSER = RegexpParser(_GRAMMAR)

# Verbs we skip as "light"/auxiliary so they don't dominate the main verb.
_AUX = {"be", "is", "are", "was", "were", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "can", "could", "may",
        "might", "must", "should"}

_STOP_OBJECT_HEADS = {"basis", "schedule", "period"}


@dataclass(frozen=True)
class Triple:
    subject: str
    verb: str          # lemmatized
    object: str
    sentence: str


def _phrase(tree) -> str:
    return " ".join(w for w, _ in tree.leaves()).strip()


def _flatten(sentence: str):
    """Return an ordered list of ('NP', phrase) and ('V', (word, lemma)) items."""
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    tree = _PARSER.parse(tagged)
    seq = []
    for node in tree:
        if isinstance(node, nltk.Tree) and node.label() == "NP":
            seq.append(("NP", _phrase(node)))
        else:
            word, tag = node
            if tag.startswith("VB"):
                lemma = _LEMM.lemmatize(word.lower(), pos="v")
                seq.append(("V", (word.lower(), lemma)))
            else:
                seq.append(("X", word))
    return seq


def extract(sentences: list[str]) -> list[Triple]:
    triples: list[Triple] = []
    for sent in sentences:
        seq = _flatten(sent)
        # scan for NP (X* V) (X* NP) patterns
        i = 0
        last_np = None
        while i < len(seq):
            kind, val = seq[i]
            if kind == "NP":
                last_np = val
                i += 1
                continue
            if kind == "V" and last_np is not None:
                word, lemma = val
                if lemma in _AUX:
                    i += 1
                    continue
                # find the next NP as object, allowing intervening X tokens
                j = i + 1
                obj = None
                while j < len(seq) and j < i + 5:
                    if seq[j][0] == "NP":
                        obj = seq[j][1]
                        break
                    if seq[j][0] == "V":  # another verb before an object -> stop
                        break
                    j += 1
                if obj is not None:
                    obj_head = obj.split()[-1].lower() if obj else ""
                    if obj_head not in _STOP_OBJECT_HEADS:
                        triples.append(Triple(
                            subject=last_np.strip(),
                            verb=lemma,
                            object=obj.strip(),
                            sentence=sent,
                        ))
                    last_np = obj  # allow chaining
                    i = j + 1
                    continue
            i += 1
    return triples
