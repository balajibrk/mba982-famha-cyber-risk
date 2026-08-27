"""Phase 1a - build a curated seed corpus of policy + breach text per company.

This is NOT scraped text. Each file is a proof-of-concept reconstruction grounded
in documented public facts about the company (see ``src/config.py`` for the
facts and ``docs/labeling_methodology.md`` for how labels are derived). The text
is written with clear subject-verb-object sentence structure so the downstream
KG pipeline (coref -> OIE -> NER -> canonicalization) can extract meaningful
triples without a heavyweight dependency parser.

Every raw file starts with a ``#``-prefixed provenance header flagging it as a
proof-of-concept reconstruction; the cleaning stage strips those header lines.

Outputs:
  data/raw/policies/{slug}.txt
  data/raw/attacks/{slug}.txt
  data/labels/labels.json          (slug -> {label, label_name, factors, score})
"""

from __future__ import annotations

import json
from datetime import date

from src import config as C
from src.config import Company

# --------------------------------------------------------------------------- #
# Natural-language predicates for each policy area (subject = the company)
# --------------------------------------------------------------------------- #
AREA_CLAUSES: dict[str, list[str]] = {
    "access control": [
        "enforces a strict access control policy",
        "restricts access to sensitive systems through role based access control",
    ],
    "credential management": [
        "manages credentials through a centralized secrets vault",
        "prohibits hardcoded credentials in source code",
    ],
    "multi-factor authentication": [
        "requires multi-factor authentication for all privileged accounts",
        "enforces multi-factor authentication on remote access",
    ],
    "data encryption": [
        "encrypts confidential data at rest and in transit",
        "applies strong encryption to protect customer data",
    ],
    "incident response": [
        "maintains an incident response plan",
        "operates an incident response team that investigates security incidents",
    ],
    "vendor risk management": [
        "assesses vendor risk before onboarding third parties",
        "governs third party access through a vendor risk management program",
    ],
    "patch management": [
        "follows a patch management policy with defined service level agreements",
        "remediates known vulnerabilities through timely patching",
    ],
    "vulnerability management": [
        "runs a vulnerability management program",
        "scans systems for vulnerabilities on a regular schedule",
    ],
    "network segmentation": [
        "segments its network to contain lateral movement",
        "isolates critical systems through network segmentation",
    ],
    "cloud security posture": [
        "monitors its cloud security posture continuously",
        "audits cloud configurations to prevent misconfiguration",
    ],
    "identity and access management": [
        "governs permissions through an identity and access management policy",
        "applies least privilege to identity and access management roles",
    ],
    "least privilege": [
        "applies the principle of least privilege to all roles",
        "limits privileged access to the minimum necessary",
    ],
    "web application firewall": [
        "protects public applications with a web application firewall",
        "tunes web application firewall rules to block malicious requests",
    ],
    "secure application development": [
        "follows secure application development practices",
        "reviews application code for security defects before release",
    ],
    "session management": [
        "enforces secure session management for customer portals",
        "issues randomized session tokens to protect user sessions",
    ],
    "dynamic application security testing": [
        "performs dynamic application security testing on customer facing applications",
        "tests running applications for exploitable flaws",
    ],
    "monitoring and logging": [
        "centralizes monitoring and logging across its infrastructure",
        "monitors security logs to detect suspicious activity",
    ],
    "privileged access management": [
        "controls administrative access through privileged access management",
        "rotates and monitors privileged credentials",
    ],
    "zero trust": [
        "adopts a zero trust security model",
        "verifies every access request under a zero trust architecture",
    ],
    "key rotation": [
        "rotates cryptographic keys on a defined schedule",
        "requires periodic rotation of vendor and service keys",
    ],
    "key management": [
        "protects encryption keys through a key management system",
        "stores encryption keys in a hardware security module",
    ],
    "third-party due diligence": [
        "conducts third party due diligence on partners",
        "reviews the security posture of acquired companies",
    ],
    "data retention": [
        "limits data retention to what is operationally necessary",
        "purges stale records under a data retention policy",
    ],
    "business continuity": [
        "maintains a business continuity and disaster recovery plan",
        "tests business continuity procedures periodically",
    ],
    "account lifecycle management": [
        "manages account lifecycle from provisioning to deprovisioning",
        "disables dormant and legacy accounts promptly",
    ],
    "software supply chain security": [
        "secures its software supply chain",
        "verifies the integrity of third party software components",
    ],
    "build pipeline integrity": [
        "protects the integrity of its software build pipeline",
        "signs build artifacts to prevent tampering",
    ],
    "code signing": [
        "signs released software with code signing certificates",
        "validates code signatures before deployment",
    ],
    "developer endpoint security": [
        "hardens developer endpoints against compromise",
        "monitors developer workstations for malicious activity",
    ],
    "endpoint security": [
        "deploys endpoint security controls on all devices",
        "monitors endpoints for malware and intrusion",
    ],
    "api security": [
        "secures its public application programming interfaces",
        "rate limits and authenticates api requests",
    ],
    "data minimization": [
        "practices data minimization across its services",
        "collects only the data required to operate",
    ],
    "abuse monitoring": [
        "monitors its platform for abuse and enumeration",
        "detects and blocks automated scraping",
    ],
    "password hashing": [
        "hashes stored passwords with a strong algorithm",
        "salts and hashes user credentials",
    ],
    "network security": [
        "defends its network with layered security controls",
        "filters malicious traffic at the network edge",
    ],
}

# --------------------------------------------------------------------------- #
# Attack-entity phrasing (used to build the breach narrative)
# --------------------------------------------------------------------------- #
ENTITY_CLAUSES: dict[str, str] = {
    "hardcoded credentials": "The attackers obtained hardcoded credentials from a private repository.",
    "private repository": "A private code repository exposed sensitive access keys.",
    "aws s3": "The intruders accessed data stored in an aws s3 bucket.",
    "concealed breach": "The company concealed the breach from regulators and victims.",
    "insider access": "An insider actor abused legitimate access.",
    "misconfigured waf": "A misconfigured web application firewall failed to block the request.",
    "server side request forgery": "The attacker used server side request forgery to reach internal metadata.",
    "cloud storage": "Confidential records were exfiltrated from cloud storage.",
    "over-broad iam role": "An over broad identity and access management role widened the blast radius.",
    "unpatched vulnerability": "The attackers exploited an unpatched vulnerability.",
    "web framework": "A flaw in a public web framework enabled remote code execution.",
    "credit records": "The breach exposed sensitive consumer credit records.",
    "delayed patch": "A delayed patch left the system exposed for weeks.",
    "vendor credentials": "Stolen vendor credentials provided the initial foothold.",
    "point of sale malware": "Point of sale malware scraped payment card data.",
    "payment cards": "The breach compromised millions of payment cards.",
    "network pivot": "The attackers pivoted from a vendor network into core systems.",
    "unauthorized access": "Unauthorized access persisted undetected for a long period.",
    "reservation database": "The intruders accessed a guest reservation database.",
    "passport numbers": "The breach exposed passport numbers of guests.",
    "prolonged dwell time": "A prolonged dwell time allowed extensive data theft.",
    "single-factor vpn": "A single factor virtual private network account was compromised.",
    "outdated server patch": "An outdated server patch left a known hole open.",
    "flat network": "A flat network let the attackers move laterally.",
    "privileged credentials": "Reused privileged credentials granted broad access.",
    "insecure direct object reference": "An insecure direct object reference exposed account data.",
    "url tampering": "The attacker altered url parameters to enumerate accounts.",
    "weak session cookie": "A weak session cookie enabled session abuse.",
    "customer portal": "The customer portal leaked account details.",
    "third party data share": "A third party data share leaked customer information.",
    "unrotated vendor key": "An unrotated vendor key remained valid for too long.",
    "vendor credential leak": "A vendor credential leak enabled unauthorized access.",
    "supply chain compromise": "A software supply chain compromise inserted a backdoor.",
    "malicious update": "A malicious update was distributed to customers.",
    "backdoor": "The backdoor granted persistent remote access.",
    "build pipeline": "The attackers subverted the software build pipeline.",
    "legacy vpn account": "A legacy virtual private network account lacked multi-factor authentication.",
    "ransomware": "Ransomware encrypted critical systems.",
    "no mfa": "The account was not protected by multi-factor authentication.",
    "operational shutdown": "The incident forced an operational shutdown.",
    "subprocessor compromise": "A support subprocessor environment was compromised.",
    "support system": "Attackers reached a customer support system.",
    "session hijacking": "Session hijacking extended the attackers reach.",
    "administrative access": "The attackers gained limited administrative access.",
    "developer compromise": "A developer endpoint was compromised first.",
    "cloud storage keys": "Stolen cloud storage keys unlocked backup archives.",
    "encrypted vaults": "Encrypted customer vault backups were exfiltrated.",
    "chained attack": "The attackers chained two incidents together.",
    "sql injection": "A sql injection flaw was exploited at scale.",
    "zero day": "The attackers weaponized a zero day vulnerability.",
    "managed file transfer": "A managed file transfer product was the entry point.",
    "mass exploitation": "Mass exploitation affected many downstream organizations.",
    "api vulnerability": "An application programming interface vulnerability enabled enumeration.",
    "account enumeration": "Account enumeration linked identities to contact details.",
    "data scraping": "The exposed data was scraped at scale.",
    "email exposure": "The breach exposed email addresses and phone numbers.",
    "forged cookies": "Forged authentication cookies bypassed login.",
    "weak hashing": "Weak password hashing accelerated credential cracking.",
    "account compromise": "Virtually every user account was compromised.",
    "delayed disclosure": "The company disclosed the breach years later.",
}

PROV_HEADER = (
    "# PROOF-OF-CONCEPT reconstruction - not a verbatim corporate document.\n"
    "# Grounded in documented public facts about {name} ({kind}).\n"
    "# Generated for a scoped reproduction of Bag, Sarkar & Bose (2025).\n"
)


def _disclosure_phrase(delay: int) -> str:
    if delay <= 30:
        return "disclosed the incident promptly to regulators and affected customers"
    if delay <= 180:
        return "disclosed the incident after an internal investigation of several months"
    return "disclosed the incident only after a prolonged delay, drawing regulatory criticism"


def build_policy_text(c: Company) -> str:
    lines = [f"{c.name} maintains a comprehensive cybersecurity policy framework."]
    for area in c.policy_areas:
        clauses = AREA_CLAUSES.get(area)
        if not clauses:
            lines.append(f"{c.name} maintains controls for {area}.")
            continue
        lines.append(f"{c.name} {clauses[0]}.")
        if len(clauses) > 1:
            # vary the subject a little for coref practice
            lines.append(f"The company also {clauses[1]}.")
    lines.append(f"{c.name} reports its security posture to executive leadership.")
    lines.append(f"{c.name} aligns its controls with recognized cybersecurity standards.")
    lines.append("The security team reviews these policies on a regular basis.")
    return " ".join(lines)


def build_attack_text(c: Company) -> str:
    if not c.breached:
        return (
            f"As of the study period, {c.name} has not experienced a major publicly "
            f"confirmed data breach of the type analyzed in this study. "
            f"Organizations in the {c.sector.replace('_', ' ')} sector nonetheless face "
            f"threats such as phishing, ransomware, and credential theft. "
            f"{c.name} monitors emerging threats and updates its controls accordingly. "
            f"The security team conducts regular tabletop exercises to test its incident "
            f"response readiness."
        )
    lines = [f"In {c.breach_year}, a significant cyber-attack occurred on {c.name}."]
    lines.append(c.root_cause)  # company-specific factual prose
    for ent in c.attack_entities:
        clause = ENTITY_CLAUSES.get(ent)
        if clause:
            lines.append(clause)
    if c.records_exposed >= 1000:
        lines.append(f"The breach exposed approximately {c.records_exposed:,} records.")
    lines.append(f"{c.name} {_disclosure_phrase(c.disclosure_delay_days)}.")
    lines.append("The incident response team investigated the intrusion and mitigated the threat.")
    if c.policy_areas:
        lines.append(f"{c.name} later strengthened its {c.policy_areas[0]} controls.")
    return " ".join(lines)


def write_corpus() -> None:
    labels: dict[str, dict] = {}
    for c in C.COMPANY_LIST:
        pol = PROV_HEADER.format(name=c.name, kind="cybersecurity policy") + build_policy_text(c) + "\n"
        atk = PROV_HEADER.format(name=c.name, kind="breach / threat article") + build_attack_text(c) + "\n"
        (C.RAW_POLICIES / f"{c.slug}.txt").write_text(pol, encoding="utf-8")
        (C.RAW_ATTACKS / f"{c.slug}.txt").write_text(atk, encoding="utf-8")
        labels[c.slug] = {
            "name": c.name,
            "label": c.label,
            "label_name": C.LABEL_SCHEME[c.label],
            "score": c.risk_score,
            "factors": {
                "scope": c.factor_scope,
                "delay": c.factor_delay,
                "recurrence": c.factor_recurrence,
                "root_severity": c.factor_rootsev,
            },
            "breached": c.breached,
            "breach_year": c.breach_year,
        }
    (C.LABELS_DIR / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Wrote {len(C.COMPANY_LIST)} policy files, "
          f"{len(C.COMPANY_LIST)} attack files, and labels.json "
          f"(generated {date.today().isoformat()}).")


if __name__ == "__main__":
    write_corpus()
