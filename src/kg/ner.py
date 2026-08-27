"""Phase 2 (iii) - Cybersecurity NER (gazetteer-based).

The paper uses CyNER (a SecureBERT-based model). Following the plan's documented
fallback (and because compiled transformer NER pipelines add VRAM pressure while
spaCy is blocked here), we use a gazetteer + POS-based typer built from the
project's own controlled vocabulary. It assigns each node one of four types that
mirror the paper's typing (organization / policy-area / attack-method / asset).

Node types:
  ORG    - companies / organizations
  POLICY - policy areas / controls
  ATTACK - attack methods / adversary actions
  ASSET  - data/systems/assets
  ENTITY - generic fallback
"""

from __future__ import annotations

import re

from src import config as C

ORG, POLICY, ATTACK, ASSET, ENTITY = "ORG", "POLICY", "ATTACK", "ASSET", "ENTITY"

# --------------------------------------------------------------------------- #
# Build gazetteers from the controlled vocabulary in config.
# --------------------------------------------------------------------------- #
def _build_gazetteers():
    orgs, policies, attacks = set(), set(), set()
    for c in C.COMPANY_LIST:
        orgs.add(c.name.lower())
        for a in c.aliases:
            if a.lower() not in {"the company", "the firm"}:
                orgs.add(a.lower())
        for area in c.policy_areas:
            policies.add(area.lower())
        for ent in c.attack_entities:
            attacks.add(ent.lower())
    return orgs, policies, attacks


_ORGS, _POLICIES, _ATTACKS = _build_gazetteers()

# Additional cyber lexicons for typing free entities.
_POLICY_TERMS = {
    "access control", "multi-factor authentication", "encryption", "incident response",
    "vendor risk", "patch management", "network segmentation", "least privilege",
    "identity and access management", "web application firewall", "zero trust",
    "key management", "key rotation", "session management", "monitoring", "logging",
    "privileged access management", "security policy", "cybersecurity policy",
    "policy framework", "control", "controls", "authentication", "governance",
    "data retention", "business continuity", "code signing", "api security",
    "data minimization", "password hashing", "endpoint security", "due diligence",
}
_ATTACK_TERMS = {
    "cyber-attack", "cyberattack", "attack", "attacker", "attackers", "intruder",
    "intruders", "breach", "ransomware", "malware", "phishing", "backdoor",
    "sql injection", "zero day", "server side request forgery", "exploit",
    "compromise", "insider", "credential theft", "enumeration", "scraping",
    "session hijacking", "supply chain compromise", "mass exploitation",
    "url tampering", "forged cookies", "misconfiguration",
}
_ASSET_TERMS = {
    "data", "records", "customer data", "confidential data", "credit records",
    "payment cards", "passport numbers", "vehicle data", "email addresses",
    "credentials", "access keys", "session token", "vault", "vaults", "database",
    "reservation database", "cloud storage", "aws s3", "s3", "server", "systems",
    "network", "account", "accounts", "customer contact data", "backup", "backups",
}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def entity_type(phrase: str) -> str:
    """Type a node phrase into one of the five categories."""
    p = phrase.strip().lower()
    p = re.sub(r"^(the|a|an|all|its|their)\s+", "", p)
    p_head = p.split()[-1] if p.split() else p

    if p in _ORGS or any(p == o or o in p for o in _ORGS):
        return ORG
    if p in _POLICIES or p in _POLICY_TERMS or any(t in p for t in _POLICY_TERMS):
        return POLICY
    if p in _ATTACKS or p in _ATTACK_TERMS or any(t in p for t in _ATTACK_TERMS):
        return ATTACK
    if p in _ASSET_TERMS or any(t == p_head for t in _ASSET_TERMS) or any(t in p for t in _ASSET_TERMS):
        return ASSET
    return ENTITY


def is_year(phrase: str) -> bool:
    return bool(_YEAR_RE.search(phrase))
