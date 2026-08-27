"""Phase 2 (v) - temporal labeling.

Attach time attributes to the KG: attack-event nodes/edges get an ``occurred_at``
year (extracted from the breach article via dateparser / regex, falling back to
the documented breach year), and policy nodes get a ``published_at`` proxy. This
is what makes the graph a *temporal* knowledge graph (paper Section 3.2.2 (v)).
"""

from __future__ import annotations

import re

import dateparser
from dateparser.search import search_dates

from src.config import Company

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def extract_years(text: str) -> list[int]:
    years = [int(m.group()) for m in _YEAR_RE.finditer(text)]
    # also try dateparser search for phrases like "November 2017"
    try:
        found = search_dates(text, languages=["en"]) or []
        for _, dt in found:
            if 1990 <= dt.year <= 2035:
                years.append(dt.year)
    except Exception:
        pass
    return sorted(set(years))


def company_timeline(company: Company, attack_text: str) -> dict:
    """Return {'occurred_at': year, 'published_at': year} for the company."""
    years = extract_years(attack_text)
    if company.breached and company.breach_year:
        occurred = company.breach_year
    elif years:
        occurred = min(years)
    else:
        occurred = None
    # policy "published" proxy: breach timeframe for breached firms, else 2023
    published = company.breach_year if company.breached else 2023
    return {"occurred_at": occurred, "published_at": published,
            "years_mentioned": years}
