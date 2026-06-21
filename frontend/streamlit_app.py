"""
Streamlit frontend for the AI-Powered Healthcare Prior Authorization Platform.

Run with:  streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1 = f"{API_BASE}/api/v1"
TIMEOUT = 120  # seconds

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prior Auth AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None) -> dict | None:
    try:
        r = httpx.get(f"{API_V1}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to API. Is the FastAPI server running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data: dict = None, files=None) -> dict | None:
    try:
        if files:
            r = httpx.post(f"{API_V1}{path}", data=data, files=files, timeout=TIMEOUT)
        else:
            r = httpx.post(f"{API_V1}{path}", json=data, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot connect to API. Is the FastAPI server running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def decision_badge(decision: str) -> str:
    colors = {
        "approved": "🟢",
        "denied": "🔴",
        "pending_info": "🟡",
        "unknown": "⚪",
    }
    labels = {
        "approved": "APPROVED",
        "denied": "DENIED",
        "pending_info": "PENDING INFO",
        "unknown": "UNKNOWN",
    }
    icon = colors.get(decision, "⚪")
    label = labels.get(decision, decision.upper())
    return f"{icon} {label}"


def severity_badge(severity: str) -> str:
    mapping = {"low": "🟢 Low", "medium": "🟡 Medium", "high": "🔴 High", "critical": "🚨 Critical"}
    return mapping.get(severity, f"⚪ {severity}")


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/ios-filled/100/4A90D9/hospital.png", width=60)
st.sidebar.title("Prior Auth AI")
st.sidebar.caption("AI-Powered Healthcare Authorization")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Authorization Analysis", "📄 Policy Upload", "🔒 PHI Scrubber", "📋 Audit History", "❤️ System Health"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(f"API: `{API_BASE}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Authorization Analysis
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Authorization Analysis":
    st.title("🔍 Prior Authorization Analysis")
    st.caption("Submit a clinical note to run the full 4-agent AI authorization pipeline.")

    col1, col2 = st.columns([2, 1])

    with col1:
        clinical_note = st.text_area(
            "Clinical Note",
            height=200,
            placeholder=(
                "Patient John Doe (DOB: 01/15/1965, SSN: 123-45-6789) presents with "
                "severe right knee osteoarthritis (M17.11). Patient has failed conservative "
                "treatments including PT for 6 months and NSAIDs. Requesting authorization "
                "for total knee arthroplasty (CPT 27447). Treating physician: Dr. Smith, "
                "contact: dr.smith@orthoclinic.com, (555) 234-5678."
            ),
        )

    with col2:
        procedure_code = st.text_input("Procedure Code (CPT)", value="27447", placeholder="e.g. 27447")
        diagnosis_code = st.text_input("Diagnosis Code (ICD-10)", value="M17.11", placeholder="e.g. M17.11")
        insurance_plan = st.text_input("Insurance Plan", value="BlueCross PPO Gold", placeholder="Plan name")

        # Optional: select from uploaded policies
        policies = api_get("/policies/list") or []
        policy_options = {f"{p['filename']} ({p['insurance_plan']})": p["policy_id"] for p in policies}
        selected_policy_label = st.selectbox(
            "Restrict to Policy (optional)",
            ["All policies"] + list(policy_options.keys()),
        )
        policy_id = policy_options.get(selected_policy_label) if selected_policy_label != "All policies" else None

    run_btn = st.button("🚀 Run Authorization Analysis", type="primary", use_container_width=True)

    if run_btn:
        if not clinical_note.strip():
            st.warning("Please enter a clinical note.")
        elif not procedure_code or not diagnosis_code or not insurance_plan:
            st.warning("Please fill in all required fields.")
        else:
            with st.spinner("Running 4-agent AI pipeline… (Clinical → Policy → Gap Detector → Recommendation)"):
                result = api_post("/authorize", {
                    "clinical_note": clinical_note,
                    "procedure_code": procedure_code,
                    "diagnosis_code": diagnosis_code,
                    "insurance_plan": insurance_plan,
                    "policy_id": policy_id,
                })

            if result:
                # ── Decision banner ─────────────────────────────────────────
                decision = result.get("authorization_decision", "unknown")
                confidence = result.get("confidence_score", 0.0)

                banner_color = {"approved": "#d4edda", "denied": "#f8d7da", "pending_info": "#fff3cd"}.get(decision, "#e2e3e5")
                st.markdown(
                    f"""<div style="background:{banner_color};padding:16px 24px;border-radius:8px;margin-bottom:16px">
                    <h2 style="margin:0">{decision_badge(decision)}</h2>
                    <p style="margin:4px 0 0 0;color:#555">Confidence: {confidence:.0%} &nbsp;|&nbsp;
                    Request ID: <code>{result.get('request_id', '')[:12]}…</code> &nbsp;|&nbsp;
                    Gap Severity: {severity_badge(result.get('gap_severity', 'unknown'))}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # ── Rationale ───────────────────────────────────────────────
                st.subheader("Recommendation Rationale")
                st.info(result.get("recommendation_rationale", ""))

                # ── Tabs for detailed results ────────────────────────────────
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📋 Policy Analysis", "🩺 Clinical Analysis", "⚠️ Gap Analysis", "🔧 Agent Trace"]
                )

                with tab1:
                    st.markdown("**Policy Summary**")
                    st.write(result.get("policy_summary", ""))
                    st.markdown("**Coverage Criteria**")
                    for c in result.get("coverage_criteria", []):
                        st.markdown(f"- {c}")
                    st.caption(f"Policy clauses retrieved: {result.get('policy_clauses_retrieved', 0)}")

                with tab2:
                    st.markdown("**Clinical Summary**")
                    st.write(result.get("clinical_summary", ""))
                    col_ev, col_fl = st.columns(2)
                    with col_ev:
                        st.markdown("**Clinical Evidence**")
                        for e in result.get("clinical_evidence", []):
                            st.markdown(f"✅ {e}")
                    with col_fl:
                        st.markdown("**Clinical Flags**")
                        for f in result.get("clinical_flags", []):
                            st.markdown(f"⚠️ {f}")

                with tab3:
                    col_cg, col_dg = st.columns(2)
                    with col_cg:
                        st.markdown("**Coverage Gaps**")
                        for g in result.get("coverage_gaps", []):
                            st.markdown(f"❌ {g}")
                    with col_dg:
                        st.markdown("**Documentation Gaps**")
                        for d in result.get("documentation_gaps", []):
                            st.markdown(f"📄 {d}")

                    if result.get("required_actions"):
                        st.markdown("**Required Actions**")
                        for i, a in enumerate(result.get("required_actions", []), 1):
                            st.markdown(f"{i}. {a}")

                    if result.get("appeal_guidance"):
                        st.markdown("**Appeal Guidance**")
                        st.warning(result.get("appeal_guidance"))

                with tab4:
                    st.markdown("**Agent Execution Trace**")
                    for step in result.get("agent_trace", []):
                        st.code(step, language=None)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Policy Upload
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Policy Upload":
    st.title("📄 Insurance Policy Upload")
    st.caption("Upload insurance policy PDFs to index them into the RAG knowledge base (ChromaDB).")

    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Select Policy PDF", type=["pdf"])
        insurance_plan_name = st.text_input("Insurance Plan Name", placeholder="e.g. BlueCross PPO Gold")
        policy_id_override = st.text_input("Policy ID (optional, auto-generated if blank)")
        submit = st.form_submit_button("📤 Upload & Index", type="primary")

    if submit:
        if not uploaded_file:
            st.warning("Please select a PDF file.")
        elif not insurance_plan_name:
            st.warning("Please enter the insurance plan name.")
        else:
            with st.spinner("Extracting text and indexing into ChromaDB…"):
                result = api_post(
                    "/policies/upload",
                    data={"insurance_plan": insurance_plan_name, "policy_id": policy_id_override or ""},
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                )
            if result:
                st.success(f"✅ {result.get('message')}")
                st.json(result)

    st.divider()
    st.subheader("Indexed Policies")

    policies = api_get("/policies/list")
    if policies:
        df = pd.DataFrame(policies)
        df["uploaded_at"] = pd.to_datetime(df["uploaded_at"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No policies uploaded yet. Upload a policy PDF above to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PHI Scrubber
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔒 PHI Scrubber":
    st.title("🔒 HIPAA PHI Scrubber")
    st.caption("Test the standalone PII/PHI redaction layer (Microsoft Presidio). Detects 10+ entity types.")

    sample_note = (
        "Patient Jane Smith (DOB: 03/22/1980, SSN: 987-65-4321) was seen on January 5, 2024. "
        "Contact: jane.smith@email.com or (555) 123-4567. Address: 123 Main St, Springfield. "
        "Medical License: ML123456. IP: 192.168.1.1. Card: 4111 1111 1111 1111."
    )

    col1, col2 = st.columns(2)
    with col1:
        text_input = st.text_area("Input Text (with PHI)", value=sample_note, height=200)
        scrub_btn = st.button("🔒 Scrub PHI", type="primary")

    with col2:
        if scrub_btn and text_input:
            result = api_post("/clinical/scrub", {"text": text_input})
            if result:
                st.text_area("Scrubbed Output", value=result.get("scrubbed_text", ""), height=200)

                st.markdown("**Detection Summary**")
                st.success(result.get("summary", ""))

                if result.get("entity_counts"):
                    df = pd.DataFrame(
                        [{"Entity Type": k, "Count": v} for k, v in result["entity_counts"].items()]
                    )
                    fig = px.bar(df, x="Entity Type", y="Count", title="PHI Entities Detected",
                                 color="Entity Type", height=300)
                    st.plotly_chart(fig, use_container_width=True)

                st.caption(f"Engine: `{result.get('scrubbing_engine', 'unknown')}`")
        elif not scrub_btn:
            st.info("Enter text on the left and click **Scrub PHI** to see results.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Audit History
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Audit History":
    st.title("📋 Authorization Audit History")
    st.caption("Full audit trail of all prior authorization analysis runs (SQLite-persisted).")

    col1, col2, col3 = st.columns(3)
    with col1:
        decision_filter = st.selectbox("Filter by Decision", ["All", "approved", "denied", "pending_info"])
    with col2:
        page_size = st.selectbox("Records per page", [10, 20, 50], index=1)
    with col3:
        skip = st.number_input("Skip (offset)", min_value=0, value=0, step=page_size)

    params = {"skip": int(skip), "limit": page_size}
    if decision_filter != "All":
        params["decision"] = decision_filter

    data = api_get("/history", params=params)

    if data:
        st.caption(f"Total records: **{data['total']}**")

        records = data.get("records", [])
        if records:
            df = pd.DataFrame(records)
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            df["confidence_score"] = df["confidence_score"].apply(
                lambda x: f"{x:.0%}" if x is not None else "—"
            )

            # Colour-code decision column
            def colour_decision(val):
                colours = {"approved": "background-color: #d4edda", "denied": "background-color: #f8d7da",
                           "pending_info": "background-color: #fff3cd"}
                return colours.get(val, "")

            styled = df.style.applymap(colour_decision, subset=["authorization_decision"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # ── Drill-down ──────────────────────────────────────────────────
            st.divider()
            selected_id = st.text_input("Enter a Request ID to view full details:")
            if selected_id:
                detail = api_get(f"/history/{selected_id.strip()}")
                if detail:
                    st.subheader(f"Result: {decision_badge(detail.get('authorization_decision', 'unknown'))}")
                    st.write(detail.get("recommendation_rationale", ""))
                    with st.expander("Full JSON"):
                        st.json(detail)
        else:
            st.info("No records found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — System Health
# ══════════════════════════════════════════════════════════════════════════════
elif page == "❤️ System Health":
    st.title("❤️ System Health")

    if st.button("🔄 Refresh"):
        st.rerun()

    health = api_get("/health".replace("/api/v1", "").replace("//", "/"))
    if not health:
        # Try absolute path
        try:
            r = httpx.get(f"{API_BASE}/health", timeout=10)
            health = r.json()
        except Exception:
            health = None

    if health:
        status = health.get("status", "unknown")
        col1, col2, col3 = st.columns(3)
        col1.metric("API Status", "🟢 Healthy" if status == "healthy" else "🟡 Degraded")
        col2.metric("DB Status", health.get("db_status", "unknown"))
        col3.metric("Version", health.get("version", "—"))

        st.subheader("ChromaDB")
        chroma = health.get("chroma_stats", {})
        st.json(chroma)
    else:
        st.error("Could not reach the health endpoint. Ensure `uvicorn app.main:app` is running.")

    st.divider()
    st.subheader("Architecture")
    st.markdown("""
    ```
    Streamlit UI
         │
         ▼
    FastAPI (REST API)
         │
         ├── POST /api/v1/authorize ──► LangGraph Pipeline
         │                                    │
         │                          ┌─────────┴───────────┐
         │                          ▼                     ▼
         │                   ClinicalAgent          PolicyAgent
         │                   (PHI Scrub +           (ChromaDB RAG
         │                    Clinical NLP)          + LLM Summary)
         │                          │                     │
         │                          └─────────┬───────────┘
         │                                    ▼
         │                           GapDetectorAgent
         │                           (Criteria vs Evidence)
         │                                    │
         │                                    ▼
         │                         RecommendationAgent
         │                         (Final Decision)
         │
         ├── POST /api/v1/policies/upload ──► ChromaDB + SQLite
         ├── POST /api/v1/clinical/scrub  ──► Presidio PII Scrubber
         └── GET  /api/v1/history         ──► SQLite Audit Log
    ```
    """)
