"""Phase 2 (i) - coreference resolution (rule-based, CPU).

The paper uses a neural coref model. Here we use a lightweight rule-based
resolver: at our scale the referents are almost always the company itself
("the company", "the firm", "the bank", pronouns "it/its/they/their"), so we
resolve those generic references to the canonical company name. This mirrors
the paper's intent (link references such as 'Uber', 'the company', 'the firm'
to the same entity) without a heavyweight neural model or GPU/VRAM cost.
"""

from __future__ import annotations

import re

from src.config import Company

# Generic references that resolve to the company subject.
_GENERIC_SUBJECTS = [
    "the company", "the firm", "the organization", "the organisation",
    "the bank", "the retailer", "the vendor", "the platform",
    "the hotel group", "the credit bureau", "the pipeline operator",
    "the identity provider", "the password manager", "the security firm",
    "the payments firm", "the internet company", "the ride-hailing firm",
]


def resolve(sentences: list[str], company: Company) -> list[str]:
    """Replace generic references + aliases with the canonical company name."""
    name = company.name
    name_low = name.lower()
    resolved = []
    # Longer aliases first to avoid partial clobbering. Skip aliases that are a
    # substring of the canonical name (e.g. "JPMorgan" inside "JPMorgan Chase"),
    # which would otherwise duplicate a name token.
    aliases = [
        a for a in sorted(set(company.aliases + _GENERIC_SUBJECTS), key=len, reverse=True)
        if a.lower() != name_low and a.lower() not in name_low
    ]
    for sent in sentences:
        s = sent
        for alias in aliases:
            # word-boundary, case-insensitive, keep sentence-initial capital
            pattern = re.compile(rf"\b{re.escape(alias)}\b", flags=re.IGNORECASE)
            s = pattern.sub(name, s)
        # bare possessive/subject pronouns referring to the company subject
        s = re.sub(r"\bthey\b", name, s, flags=re.IGNORECASE)
        s = re.sub(r"\btheir\b", f"{name} 's", s, flags=re.IGNORECASE)
        # collapse any consecutive repeated words introduced by replacements
        s = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip()
        resolved.append(s)
    return resolved
