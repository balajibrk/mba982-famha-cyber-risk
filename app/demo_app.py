"""Phase 6 (stretch) - Streamlit demo.

Pick a company -> see its temporal KG snapshot, the model's risk class, SHAP top
entities, FAMHA attention heatmap, the real counterfactual re-score, and the
grounded LLM narrative. Run with:

    .venv\\Scripts\\streamlit run app\\demo_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C            # noqa: E402
from src.assistant.pipeline import run_company  # noqa: E402
from src.interpret.common import load_model     # noqa: E402
from src.model.features import load_dataset      # noqa: E402


@st.cache_resource
def _load():
    device = C.get_device()
    model = load_model(device)
    model.eval()
    return model, load_dataset(), device


st.set_page_config(page_title="X-FAMHA-GNN Cyber Risk", layout="wide")
st.title("Temporal KG + X-FAMHA-GNN — Cybersecurity Risk Assessment")
st.caption("Scoped reproduction of Bag, Sarkar & Bose (2025) + a grounded LLM assistant layer. "
           "Data is a proof-of-concept curated corpus.")

model, dataset, device = _load()
slugs = [c.slug for c in C.COMPANY_LIST]
slug = st.sidebar.selectbox("Company", slugs,
                            format_func=lambda s: C.COMPANIES_BY_SLUG[s].name)

if st.sidebar.button("Analyze", type="primary") or slug:
    with st.spinner("Running KG -> risk -> SHAP/attention -> counterfactual -> narrative..."):
        r = run_company(slug, model, dataset, device)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted risk", r["pred_label"], f"{r['pred_prob']:.0%} conf.")
    c2.metric("Ground-truth label", r["true_label"])
    cf = r["counterfactual"]
    c3.metric("Counterfactual risk (high+crit)",
              f"{cf['risk_after']:.2f}", f"{cf['delta']:+.2f} vs {cf['risk_before']:.2f}")

    st.subheader("Grounded assessment")
    n = r["narrative"]
    st.markdown(f"**Verdict.** {n['one_line_verdict']}")
    st.markdown(f"**Why.** {n['why']}")
    st.markdown(f"**Fix.** {n['fix']}")
    st.markdown(f"**Impact.** {n['impact']}")
    st.caption(f"narrative source: {r['narrative_source']}")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Top entities (SHAP)")
        p = Path(r["artifacts"]["shap_png"])
        if p.exists():
            st.image(str(p))
    with colB:
        st.subheader("FAMHA attention (policy x attack)")
        p = Path(r["artifacts"]["attention_png"])
        if p.exists():
            st.image(str(p))

    st.subheader("Knowledge-graph snapshot")
    G = nx.read_graphml(C.KG_DIR / f"{slug}.graphml")
    st.write(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
             f"breach year {G.graph.get('occurred_at', 'n/a')}")
    edges = [{"subject": G.nodes[u].get("label"), "relation": d.get("relation"),
              "object": G.nodes[v].get("label"), "source": d.get("source")}
             for u, v, d in list(G.edges(data=True))[:25]]
    st.dataframe(edges, use_container_width=True)

    with st.expander("Engineer ticket (JSON)"):
        st.json(r["engineer_ticket"])
