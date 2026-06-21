"""
Clinical Agent — analyses the PHI-scrubbed clinical note to extract
medical evidence and flag any deficiencies relevant to authorization.
"""
from __future__ import annotations

import json
from loguru import logger

from app.agents.state import AuthState
from app.agents.llm import get_llm, StubLLM
from app.pii.scrubber import scrub_phi

SYSTEM_PROMPT = """You are a clinical documentation specialist reviewing healthcare prior authorization requests.

Your task:
1. Analyse the clinical note provided.
2. Extract evidence that supports medical necessity for the requested procedure.
3. Identify any clinical flags — missing information, outdated records, or concerns.

Output ONLY valid JSON in this exact schema:
{
  "clinical_summary": "<structured 2–4 sentence summary of the patient's condition and care context>",
  "clinical_evidence": ["<evidence item 1>", "<evidence item 2>", ...],
  "clinical_flags": ["<flag 1>", "<flag 2>", ...]
}
"""


def clinical_agent(state: AuthState) -> AuthState:
    """
    LangGraph node: Clinical Agent.

    Reads: clinical_note, procedure_code, diagnosis_code
    Writes: scrubbed_note, clinical_summary, clinical_evidence, clinical_flags
    """
    logger.info(f"[ClinicalAgent] Starting for request_id={state.get('request_id')}")
    trace = list(state.get("agent_trace", []))
    trace.append("ClinicalAgent: started")

    try:
        raw_note = state.get("clinical_note", "")

        # ── Step 1: HIPAA scrub ──────────────────────────────────────────────
        scrub_result = scrub_phi(raw_note)
        scrubbed = scrub_result.scrubbed_text
        trace.append(
            f"ClinicalAgent: PHI scrubbed — {scrub_result.summary} "
            f"(engine: {scrub_result.scrubbing_engine})"
        )

        # ── Step 2: LLM clinical analysis ────────────────────────────────────
        llm = get_llm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Procedure Requested: {state.get('procedure_code', 'Unknown')}\n"
                    f"Diagnosis: {state.get('diagnosis_code', 'Unknown')}\n\n"
                    f"Clinical Note:\n{scrubbed}"
                ),
            },
        ]

        if isinstance(llm, StubLLM):
            response = llm.invoke(messages, agent_type="clinical")
        else:
            response = llm.invoke(messages)

        parsed = json.loads(response.content)
        trace.append("ClinicalAgent: LLM clinical analysis complete")

        return {
            **state,
            "scrubbed_note": scrubbed,
            "clinical_summary": parsed.get("clinical_summary", ""),
            "clinical_evidence": parsed.get("clinical_evidence", []),
            "clinical_flags": parsed.get("clinical_flags", []),
            "agent_trace": trace,
        }

    except Exception as e:
        logger.error(f"[ClinicalAgent] Error: {e}")
        trace.append(f"ClinicalAgent: ERROR — {e}")
        return {
            **state,
            "scrubbed_note": state.get("clinical_note", ""),
            "clinical_summary": f"Clinical analysis failed: {e}",
            "clinical_evidence": [],
            "clinical_flags": [f"Analysis error: {e}"],
            "agent_trace": trace,
            "error": str(e),
        }
