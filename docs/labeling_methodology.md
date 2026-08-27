# Labeling Methodology

This document defines exactly how each company in the case-study set is assigned
one of four cybersecurity-policy risk classes. It exists so the labels are
transparent and defensible: judges can see precisely why a company got its class.

## Why a heuristic

The paper (Bag, Sarkar & Bose, 2025) labels companies using breach severity and
scope sourced from `idtheftcenter.org` and `upguard.com`. We do **not** have
access to those proprietary severity scales, so we derive labels from a small,
explicit, reproducible heuristic built from **documented public facts** about
each breach. This is a proof-of-concept labeling scheme, not a claim of parity
with the paper's data sources.

## Risk classes

| Label | Name            |
|------:|-----------------|
| 0     | `low_risk`      |
| 1     | `medium_risk`   |
| 2     | `high_risk`     |
| 3     | `critical_risk` |

## Scoring factors

Each company gets a composite score from four factors. The first three are
objective public facts; the fourth is a small, documented judgment factor for
root-cause seriousness that a pure record count would miss.

### 1. Scope (records exposed) — `factor_scope`, range 0–4

A log10 bucket of the number of records/accounts exposed:

| Records exposed        | Score |
|------------------------|------:|
| 0 (no major breach)    | 0     |
| < 1,000,000            | 1     |
| 1M – < 10M             | 2     |
| 10M – < 100M           | 3     |
| >= 100M                | 4     |

### 2. Disclosure delay — `factor_delay`, range 0–2

Time from breach/detection to public disclosure. Regulatory and reputational
risk rises with concealment.

| Delay (days) | Score |
|--------------|------:|
| <= 30        | 0     |
| 31 – 180     | 1     |
| > 180        | 2     |

### 3. Recurrence — `factor_recurrence`, range 0–2

`min(recurrence_count - 1, 2)` for breached companies (0 for clean ones), where
`recurrence_count` is the number of distinct major breaches. Repeat offenders
carry more residual policy risk.

### 4. Root-cause severity / systemic impact — `factor_rootsev`, range 0–2

A documented 0–2 judgment factor capturing seriousness that record counts miss:
critical-infrastructure disruption, software supply-chain compromise, deliberate
concealment, or exposure of especially sensitive material (e.g. password vaults,
SSNs). Each company's value is justified inline in `src/config.py` via its
`root_cause` text.

## Composite score and mapping

```
risk_score = factor_scope + factor_delay + factor_recurrence + factor_rootsev   # 0..10
```

| Composite score | Label            |
|-----------------|------------------|
| 0 – 1           | `low_risk` (0)   |
| 2 – 4           | `medium_risk` (1)|
| 5 – 7           | `high_risk` (2)  |
| 8 – 10          | `critical_risk` (3) |

## Resulting label distribution (18 companies)

| Class         | Count | Companies |
|---------------|------:|-----------|
| low_risk      | 3 | Apple, Cloudflare, Stripe |
| medium_risk   | 5 | Target, Citigroup, Wells Fargo, Colonial Pipeline, Okta |
| high_risk     | 8 | Capital One, Equifax, Marriott, JPMorgan Chase, SolarWinds, LastPass, Progress MOVEit, Twitter |
| critical_risk | 2 | Uber, Yahoo |

## Known limitations (stated honestly)

- **Small N.** 18 companies with a skewed distribution (only 2 critical). Results
  are illustrative, not statistically powered like the paper's 190-company study.
- **Records-count proxy.** Operationally severe but low-record incidents (e.g.
  Colonial Pipeline) are partly under-scored by the scope factor; the
  `root_severity` factor is the deliberate correction for this.
- **Public-fact approximations.** Record counts and disclosure delays are rounded
  public figures used only as a labeling proxy, not audited values.
- **Clean != invulnerable.** "Clean" companies are those without a major publicly
  confirmed catastrophic customer-data breach of the type studied here; it is not
  a claim that they have never had any security incident.

Labels are computed programmatically from these factors in `src/config.py`
(`Company.label`); regenerate the per-company label files with
`python -m src.preprocess.build_corpus`.
