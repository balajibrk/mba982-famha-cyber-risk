"""Central configuration: paths, company list, labeling heuristic, GPU helpers.

This is a scoped, honest reproduction of Bag, Sarkar & Bose (2025),
"Enhancing cybersecurity risk assessment using temporal knowledge graph-based
explainable decision support system" (Decision Support Systems 198, 114526).

Scope deviations from the paper (flagged for hackathon credibility):
  * ~18 companies (proof-of-concept), not the paper's 190 train / 154 test corpus.
  * Labels come from a documented heuristic (see docs/labeling_methodology.md),
    not the paper's idtheftcenter.org / upguard.com severity data.
  * Policy + breach text is a curated seed corpus grounded in documented public
    facts, not live-scraped corporate pages.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_POLICIES = DATA / "raw" / "policies"
RAW_ATTACKS = DATA / "raw" / "attacks"
LABELS_DIR = DATA / "labels"
PROCESSED = DATA / "processed"
KG_DIR = PROCESSED / "kg"
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"

for _d in (RAW_POLICIES, RAW_ATTACKS, LABELS_DIR, PROCESSED, KG_DIR, DOCS, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Label scheme (paper uses 4 vulnerability/risk classes)
# --------------------------------------------------------------------------- #
LABEL_SCHEME = {0: "low_risk", 1: "medium_risk", 2: "high_risk", 3: "critical_risk"}
NUM_CLASSES = len(LABEL_SCHEME)

# --------------------------------------------------------------------------- #
# The paper's 8 canonical relation types (Section 3.2.2, canonicalization step)
# --------------------------------------------------------------------------- #
CANONICAL_RELATIONS = [
    "implements",
    "aligns-with",
    "violates",
    "mitigates",
    "causes",
    "impacts",
    "reports",
    "regulates",
]

# Shared sentence-transformer used for canonicalization + node text embeddings.
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_DIM = 384

# Embedding / graph settings
MINILM_TEXT = SBERT_MODEL
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Company metadata
# --------------------------------------------------------------------------- #
@dataclass
class Company:
    """Facts + labeling factors for one company.

    The four labeling factors are the inputs to the documented heuristic in
    docs/labeling_methodology.md. ``records_exposed``, ``disclosure_delay_days``
    and ``recurrence_count`` are objective public facts; ``root_severity`` is a
    small (0-2) documented judgment factor for root-cause seriousness / systemic
    or operational impact that a pure record-count proxy misses (e.g. critical
    infrastructure, supply-chain, deliberate concealment).
    """

    name: str
    slug: str
    sector: str
    breached: bool
    breach_year: Optional[int]
    records_exposed: int
    disclosure_delay_days: int
    recurrence_count: int  # number of distinct major breaches (>=1 if breached)
    root_severity: int  # 0-2 documented judgment factor
    root_cause: str
    aliases: list[str] = field(default_factory=list)
    policy_areas: list[str] = field(default_factory=list)
    attack_entities: list[str] = field(default_factory=list)

    # -- labeling factor computations (see docs/labeling_methodology.md) ------ #
    @property
    def factor_scope(self) -> int:
        """log10 bucket of records exposed -> 0..4."""
        r = self.records_exposed
        if r <= 0:
            return 0
        if r < 1_000_000:
            return 1
        if r < 10_000_000:
            return 2
        if r < 100_000_000:
            return 3
        return 4

    @property
    def factor_delay(self) -> int:
        d = self.disclosure_delay_days
        if d <= 30:
            return 0
        if d <= 180:
            return 1
        return 2

    @property
    def factor_recurrence(self) -> int:
        if not self.breached:
            return 0
        return max(0, min(self.recurrence_count - 1, 2))

    @property
    def factor_rootsev(self) -> int:
        return max(0, min(self.root_severity, 2))

    @property
    def risk_score(self) -> int:
        return (
            self.factor_scope
            + self.factor_delay
            + self.factor_recurrence
            + self.factor_rootsev
        )

    @property
    def label(self) -> int:
        """Map composite score (0..10) to a 4-class label."""
        s = self.risk_score
        if s <= 1:
            return 0  # low
        if s <= 4:
            return 1  # medium
        if s <= 7:
            return 2  # high
        return 3  # critical


# 15 breached + 3 comparatively clean = 18 companies.
# Records / delays are approximate public figures used only as a labeling proxy.
COMPANY_LIST: list[Company] = [
    # ---- breached ---------------------------------------------------------- #
    Company(
        name="Uber", slug="uber", sector="mobility", breached=True, breach_year=2016,
        records_exposed=57_000_000, disclosure_delay_days=365, recurrence_count=2,
        root_severity=2,
        root_cause="Hardcoded AWS credentials in a private code repository let attackers reach an S3 datastore; the breach was concealed for a year and hushed with a payout.",
        aliases=["Uber", "Uber Technologies", "the company", "the firm", "the ride-hailing firm"],
        policy_areas=["access control", "credential management", "vendor risk management", "incident response", "data encryption"],
        attack_entities=["hardcoded credentials", "private repository", "aws s3", "concealed breach", "insider access"],
    ),
    Company(
        name="Capital One", slug="capital_one", sector="banking", breached=True, breach_year=2019,
        records_exposed=106_000_000, disclosure_delay_days=120, recurrence_count=1,
        root_severity=1,
        root_cause="A misconfigured web application firewall allowed a server-side request forgery that exfiltrated data from cloud storage buckets.",
        aliases=["Capital One", "Capital One Financial", "the bank", "the company"],
        policy_areas=["cloud security posture", "access control", "identity and access management", "web application firewall", "least privilege"],
        attack_entities=["misconfigured waf", "server side request forgery", "cloud storage", "over-broad iam role"],
    ),
    Company(
        name="Equifax", slug="equifax", sector="credit_bureau", breached=True, breach_year=2017,
        records_exposed=147_000_000, disclosure_delay_days=40, recurrence_count=1,
        root_severity=2,
        root_cause="An unpatched known vulnerability in a public web application framework was exploited to access consumer credit records.",
        aliases=["Equifax", "the credit bureau", "the company"],
        policy_areas=["patch management", "vulnerability management", "data encryption", "incident response", "network segmentation"],
        attack_entities=["unpatched vulnerability", "web framework", "credit records", "delayed patch"],
    ),
    Company(
        name="Target", slug="target", sector="retail", breached=True, breach_year=2013,
        records_exposed=70_000_000, disclosure_delay_days=20, recurrence_count=1,
        root_severity=1,
        root_cause="Stolen HVAC vendor credentials gave a foothold that spread to point-of-sale systems infected with card-scraping malware.",
        aliases=["Target", "Target Corporation", "the retailer", "the company"],
        policy_areas=["vendor risk management", "network segmentation", "access control", "endpoint security", "incident response"],
        attack_entities=["vendor credentials", "point of sale malware", "payment cards", "network pivot"],
    ),
    Company(
        name="Marriott", slug="marriott", sector="hospitality", breached=True, breach_year=2018,
        records_exposed=383_000_000, disclosure_delay_days=1460, recurrence_count=1,
        root_severity=1,
        root_cause="Unauthorized access persisted for years in an acquired reservation database, exposing guest records including passport numbers.",
        aliases=["Marriott", "Starwood", "Marriott International", "the hotel group", "the company"],
        policy_areas=["access control", "data encryption", "third-party due diligence", "data retention", "monitoring and logging"],
        attack_entities=["unauthorized access", "reservation database", "passport numbers", "prolonged dwell time"],
    ),
    Company(
        name="JPMorgan Chase", slug="jpmorgan", sector="banking", breached=True, breach_year=2014,
        records_exposed=83_000_000, disclosure_delay_days=75, recurrence_count=1,
        root_severity=1,
        root_cause="Reused privileged VPN credentials without multi-factor authentication, combined with a slow patch cadence, exposed customer contact data.",
        aliases=["JPMorgan", "JPMorgan Chase", "the bank", "the company"],
        policy_areas=["multi-factor authentication", "access control", "patch management", "network segmentation", "privileged access management"],
        attack_entities=["single-factor vpn", "outdated server patch", "flat network", "privileged credentials"],
    ),
    Company(
        name="Citigroup", slug="citigroup", sector="banking", breached=True, breach_year=2011,
        records_exposed=360_000, disclosure_delay_days=25, recurrence_count=1,
        root_severity=1,
        root_cause="URL parameter tampering (an insecure direct object reference) in the customer card portal exposed account details.",
        aliases=["Citigroup", "Citi", "the bank", "the company"],
        policy_areas=["secure application development", "session management", "access control", "dynamic application security testing"],
        attack_entities=["insecure direct object reference", "url tampering", "weak session cookie", "customer portal"],
    ),
    Company(
        name="Wells Fargo", slug="wells_fargo", sector="banking", breached=True, breach_year=2008,
        records_exposed=200_000, disclosure_delay_days=60, recurrence_count=1,
        root_severity=1,
        root_cause="A third-party vendor credential leak allowed unauthorized access to customer records held by an external partner.",
        aliases=["Wells Fargo", "the bank", "the company"],
        policy_areas=["vendor risk management", "zero trust", "key rotation", "third-party due diligence", "access control"],
        attack_entities=["third party data share", "unrotated vendor key", "vendor credential leak"],
    ),
    Company(
        name="SolarWinds", slug="solarwinds", sector="software", breached=True, breach_year=2020,
        records_exposed=18_000, disclosure_delay_days=270, recurrence_count=1,
        root_severity=2,
        root_cause="A nation-state compromised the software build pipeline, inserting a backdoor into signed product updates distributed to thousands of organizations.",
        aliases=["SolarWinds", "Orion", "the vendor", "the company"],
        policy_areas=["software supply chain security", "build pipeline integrity", "code signing", "monitoring and logging", "incident response"],
        attack_entities=["supply chain compromise", "malicious update", "backdoor", "build pipeline"],
    ),
    Company(
        name="Colonial Pipeline", slug="colonial_pipeline", sector="critical_infrastructure", breached=True, breach_year=2021,
        records_exposed=5_800, disclosure_delay_days=5, recurrence_count=1,
        root_severity=2,
        root_cause="A ransomware crew logged in through a legacy VPN account lacking multi-factor authentication, forcing a shutdown of fuel distribution.",
        aliases=["Colonial Pipeline", "Colonial", "the pipeline operator", "the company"],
        policy_areas=["multi-factor authentication", "account lifecycle management", "network segmentation", "business continuity", "incident response"],
        attack_entities=["legacy vpn account", "ransomware", "no mfa", "operational shutdown"],
    ),
    Company(
        name="Okta", slug="okta", sector="identity", breached=True, breach_year=2022,
        records_exposed=366, disclosure_delay_days=90, recurrence_count=2,
        root_severity=1,
        root_cause="A support subprocessor's environment was compromised, giving attackers limited access to some customer administrative data.",
        aliases=["Okta", "the identity provider", "the company"],
        policy_areas=["vendor risk management", "least privilege", "session management", "monitoring and logging", "incident response"],
        attack_entities=["subprocessor compromise", "support system", "session hijacking", "administrative access"],
    ),
    Company(
        name="LastPass", slug="lastpass", sector="software", breached=True, breach_year=2022,
        records_exposed=25_000_000, disclosure_delay_days=120, recurrence_count=2,
        root_severity=2,
        root_cause="Attackers chained an initial developer compromise to steal cloud storage keys and exfiltrate encrypted customer password vault backups.",
        aliases=["LastPass", "the password manager", "the company"],
        policy_areas=["credential management", "data encryption", "key management", "developer endpoint security", "cloud security posture"],
        attack_entities=["developer compromise", "cloud storage keys", "encrypted vaults", "chained attack"],
    ),
    Company(
        name="Progress MOVEit", slug="moveit", sector="software", breached=True, breach_year=2023,
        records_exposed=60_000_000, disclosure_delay_days=10, recurrence_count=1,
        root_severity=2,
        root_cause="A ransomware group mass-exploited a SQL injection zero-day in a managed file transfer product to steal data from many downstream organizations.",
        aliases=["MOVEit", "Progress", "Progress Software", "the vendor", "the company"],
        policy_areas=["vulnerability management", "secure application development", "software supply chain security", "incident response", "data encryption"],
        attack_entities=["sql injection", "zero day", "managed file transfer", "mass exploitation"],
    ),
    Company(
        name="Twitter", slug="twitter", sector="social_media", breached=True, breach_year=2022,
        records_exposed=5_400_000, disclosure_delay_days=150, recurrence_count=2,
        root_severity=1,
        root_cause="An API flaw allowed enumeration that linked accounts to email addresses and phone numbers, later scraped at scale.",
        aliases=["Twitter", "X", "the platform", "the company"],
        policy_areas=["api security", "access control", "data minimization", "abuse monitoring", "vulnerability management"],
        attack_entities=["api vulnerability", "account enumeration", "data scraping", "email exposure"],
    ),
    Company(
        name="Yahoo", slug="yahoo", sector="internet", breached=True, breach_year=2013,
        records_exposed=3_000_000_000, disclosure_delay_days=1095, recurrence_count=2,
        root_severity=1,
        root_cause="Forged authentication cookies and weak password hashing led to the compromise of every user account, disclosed years after the fact.",
        aliases=["Yahoo", "the company", "the internet company"],
        policy_areas=["data encryption", "password hashing", "session management", "monitoring and logging", "incident response"],
        attack_entities=["forged cookies", "weak hashing", "account compromise", "delayed disclosure"],
    ),
    # ---- comparatively clean (no major catastrophic customer-data breach) --- #
    Company(
        name="Apple", slug="apple", sector="technology", breached=False, breach_year=None,
        records_exposed=0, disclosure_delay_days=0, recurrence_count=0,
        root_severity=0,
        root_cause="No major publicly confirmed customer-data breach of the kind studied here; strong platform security posture.",
        aliases=["Apple", "Apple Inc", "the company"],
        policy_areas=["data encryption", "multi-factor authentication", "secure application development", "access control", "vulnerability management"],
        attack_entities=[],
    ),
    Company(
        name="Cloudflare", slug="cloudflare", sector="security", breached=False, breach_year=None,
        records_exposed=0, disclosure_delay_days=0, recurrence_count=0,
        root_severity=0,
        root_cause="No major customer-data breach; transparent security engineering and rapid incident disclosure practices.",
        aliases=["Cloudflare", "the company", "the security firm"],
        policy_areas=["zero trust", "network security", "incident response", "monitoring and logging", "access control"],
        attack_entities=[],
    ),
    Company(
        name="Stripe", slug="stripe", sector="fintech", breached=False, breach_year=None,
        records_exposed=0, disclosure_delay_days=0, recurrence_count=0,
        root_severity=0,
        root_cause="No major publicly confirmed data breach; PCI-focused security program.",
        aliases=["Stripe", "the company", "the payments firm"],
        policy_areas=["data encryption", "access control", "key management", "secure application development", "vendor risk management"],
        attack_entities=[],
    ),
]

COMPANIES_BY_SLUG = {c.slug: c for c in COMPANY_LIST}


def label_distribution() -> dict[str, int]:
    dist = {name: 0 for name in LABEL_SCHEME.values()}
    for c in COMPANY_LIST:
        dist[LABEL_SCHEME[c.label]] += 1
    return dist


# --------------------------------------------------------------------------- #
# GPU helpers
# --------------------------------------------------------------------------- #
def get_device():
    """Return the best available torch device (GPU-first, per project brief)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_gpu(*objs) -> None:
    """Delete objects and empty the CUDA cache between pipeline stages.

    Keeps transformer models from stacking up on the 8 GB card, as instructed
    in the build brief (``del model; torch.cuda.empty_cache()``).
    """
    import torch

    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"{len(COMPANY_LIST)} companies "
          f"({sum(c.breached for c in COMPANY_LIST)} breached, "
          f"{sum(not c.breached for c in COMPANY_LIST)} clean)")
    print("Label distribution:", label_distribution())
    print()
    for c in COMPANY_LIST:
        print(f"  {c.name:20s} score={c.risk_score:2d} "
              f"(scope={c.factor_scope} delay={c.factor_delay} "
              f"recur={c.factor_recurrence} root={c.factor_rootsev}) "
              f"-> {LABEL_SCHEME[c.label]}")
