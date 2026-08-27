"""Phase 6b - grounded narrative generation (local Ollama, template fallback).

The LLM is given ONLY precomputed evidence (risk class, top SHAP entities, top
attention pairs, and the real counterfactual delta). A hard system rule forbids
referencing anything not in that evidence, keeping the narrative defensible if a
judge fact-checks it. Output is a strict JSON object
{one_line_verdict, why, fix, impact}. If Ollama is unavailable, a deterministic
template built from the same evidence is used instead (clearly flagged).
"""

from __future__ import annotations

import json
from typing import Optional

DEFAULT_MODEL = "llama3.1:8b"
SCHEMA_KEYS = ["one_line_verdict", "why", "fix", "impact"]

SYSTEM_PROMPT = (
    "You are a cybersecurity risk analyst. You will be given a JSON object of "
    "PRECOMPUTED evidence about one company. Write a short, grounded assessment.\n"
    "HARD RULES:\n"
    "1. Only reference entities, edges, controls, or numbers that appear in the "
    "input evidence. Never invent a policy area, control, statistic, or number "
    "that is not provided.\n"
    "2. The risk-reduction number must be exactly the counterfactual delta given.\n"
    "3. Respond with ONLY a JSON object with keys: one_line_verdict, why, fix, impact.\n"
)


def build_evidence(company: str, pred_label: str, pred_prob: float,
                   top_shap: list[dict], top_attention: list[dict],
                   counterfactual: dict) -> dict:
    return {
        "company": company,
        "risk_class": pred_label,
        "confidence": round(float(pred_prob), 3),
        "top_shap_entities": [
            {"entity": t["entity"], "shap": round(float(t["shap"]), 4)}
            for t in top_shap[:3]
        ],
        "top_attention_pairs": [
            {"policy_entity": p["policy_entity"], "attack_entity": p["attack_entity"],
             "weight": round(float(p["weight"]), 3)}
            for p in top_attention[:3]
        ],
        "counterfactual": {
            "risk_before": counterfactual["risk_before"],
            "risk_after": counterfactual["risk_after"],
            "delta": counterfactual["delta"],
            "pct_change": counterfactual["pct_change"],
            "what_changed": counterfactual["what_changed"],
        },
    }


def ollama_available(model: str = DEFAULT_MODEL) -> Optional[str]:
    """Return an available model name (prefer `model`) or None."""
    try:
        import ollama
        tags = ollama.list()
        names = [m.get("model") or m.get("name") for m in tags.get("models", [])]
        if not names:
            return None
        for n in names:
            if n and n.startswith(model.split(":")[0]):
                return n
        return names[0]
    except Exception:
        return None


def _generate_ollama(evidence: dict, model: str) -> dict:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(evidence, indent=2)},
        ],
        format="json",
        options={"temperature": 0.2},
    )
    content = resp["message"]["content"]
    data = json.loads(content)
    return {k: data.get(k, "") for k in SCHEMA_KEYS}


def _template(evidence: dict) -> dict:
    c = evidence
    cf = c["counterfactual"]
    shap_ents = ", ".join(e["entity"] for e in c["top_shap_entities"]) or "n/a"
    pair = c["top_attention_pairs"][0] if c["top_attention_pairs"] else None
    pair_txt = (f"the link between '{pair['policy_entity']}' and "
                f"'{pair['attack_entity']}'") if pair else "the flagged policy-attack link"
    direction = "reduces" if cf["delta"] < 0 else "changes"
    return {
        "one_line_verdict": (f"{c['company']} is classified {c['risk_class']} "
                             f"(confidence {c['confidence']:.0%})."),
        "why": (f"The model attributes this most to: {shap_ents}. Attention "
                f"highlights {pair_txt}."),
        "fix": cf["what_changed"],
        "impact": (f"Simulating this fix {direction} predicted high/critical risk "
                   f"from {cf['risk_before']:.2f} to {cf['risk_after']:.2f} "
                   f"({cf['pct_change']:+.1f}%), per model re-scoring."),
    }


def _validate_grounded(narr: dict, evidence: dict) -> list[str]:
    """Flag any number in the narrative not present in the evidence (light check)."""
    import re
    allowed = set()
    for e in evidence["top_shap_entities"]:
        allowed.add(f"{e['shap']}")
    cf = evidence["counterfactual"]
    for v in (cf["risk_before"], cf["risk_after"], cf["delta"], cf["pct_change"],
              evidence["confidence"]):
        allowed.add(f"{v}")
    text = " ".join(str(v) for v in narr.values())
    warns = []
    for num in re.findall(r"\d+\.\d+", text):
        if not any(num in a for a in allowed):
            warns.append(num)
    return warns


def generate(company: str, pred_label: str, pred_prob: float,
             top_shap: list[dict], top_attention: list[dict],
             counterfactual: dict, model: str = DEFAULT_MODEL) -> dict:
    evidence = build_evidence(company, pred_label, pred_prob, top_shap,
                              top_attention, counterfactual)
    avail = ollama_available(model)
    source = "template"
    if avail:
        try:
            narr = _generate_ollama(evidence, avail)
            if all(narr.get(k) for k in SCHEMA_KEYS):
                source = f"ollama:{avail}"
            else:
                narr = _template(evidence)
        except Exception:
            narr = _template(evidence)
    else:
        narr = _template(evidence)

    return {"narrative": narr, "evidence": evidence, "source": source,
            "ungrounded_number_warnings": _validate_grounded(narr, evidence)}


def exec_summary(narr: dict) -> str:
    n = narr["narrative"]
    return f"{n['one_line_verdict']} {n['impact']}"


def engineer_ticket(narr: dict) -> dict:
    n = narr["narrative"]
    ev = narr["evidence"]
    return {
        "title": f"[{ev['risk_class'].upper()}] Harden cybersecurity policy for {ev['company']}",
        "severity": ev["risk_class"],
        "why": n["why"],
        "remediation": n["fix"],
        "expected_impact": n["impact"],
    }
