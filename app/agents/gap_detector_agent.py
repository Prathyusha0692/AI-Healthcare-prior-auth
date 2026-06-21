"""
Gap Detector Agent — compares policy coverage criteria against
clinical evidence to identify documentation and coverage gaps.
"""
from __future__ import annotations

import json
from loguru import logger

from app.agents.state import AuthState
from app.agents.llm import get_llm, StubLLM

SYSTEM_PROMPT = """You are a prior authorization gap analyst for a healthcare insurance company.

You will receive:
- A list of policy coverage criteria that must be satisfied
- A list of clinical evidence extracted from the patient's clinical note
- Any clinical flags raised by the clinical documentation reviewer

Your task:
1. Map each policy criterion to the available clinical evidence.
2. Identify COVERAGE GAPS — criteria not met or not documented.
3. Identify DOCUMENTATION GAPS — specific records or data that are missing.
4. Rate the overall gap severity.

Output ONLY valid JSON in this exact schema:
{
  "coverage_gaps": ["<unmet criterion 1>", "<unmet criterion 2>", ...],
  "documentation_gaps": ["<missing document 1>", "<missing document 2>", ...],
  "gap_severity": "<low|medium|high|critical>"
}

Severity guide:
  low      — minor issues unlikely to affect authorization
  medium   — some gaps that may delay approval
  high     — significant gaps likely to result in denial
  critical — fundamental requirements unmet
"""


def gap_detector_agent(state: AuthState) -> AuthState:
    """
    LangGraph node: Gap Detector Agent.

    Reads: coverage_criteria, clinical_evidence, clinical_flags
    Writes: coverage_gaps, documentation_gaps, gap_severity
    """
    logger.info(f"[GapDetectorAgent] Starting for request_id={state.get('request_id')}")
    trace = list(state.get("agent_trace", []))
    trace.append("GapDetectorAgent: started")

    try:
        criteria = state.get("coverage_criteria", [])
        evidence = state.get("clinical_evidence", [])
        flags = state.get("clinical_flags", [])

        llm = get_llm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "=== POLICY COVERAGE CRITERIA ===\n"
                    + "\n".join(f"- {c}" for c in criteria)
                    + "\n\n=== CLINICAL EVIDENCE AVAILABLE ===\n"
                    + "\n".join(f"- {e}" for e in evidence)
                    + "\n\n=== CLINICAL FLAGS ===\n"
                    + "\n".join(f"- {f}" for f in flags)
                ),
            },
        ]

        if isinstance(llm, StubLLM):
            response = llm.invoke(messages, agent_type="gap")
        else:
            response = llm.invoke(messages)

        parsed = json.loads(response.content)
        severity = parsed.get("gap_severity", "medium")
        trace.append(
            f"GapDetectorAgent: found {len(parsed.get('coverage_gaps', []))} coverage gaps, "
            f"severity={severity}"
        )

        return {
            **state,
            "coverage_gaps": parsed.get("coverage_gaps", []),
            "documentation_gaps": parsed.get("documentation_gaps", []),
            "gap_severity": severity,
            "agent_trace": trace,
        }

    except Exception as e:
        logger.error(f"[GapDetectorAgent] Error: {e}")
        trace.append(f"GapDetectorAgent: ERROR — {e}")
        return {
            **state,
            "coverage_gaps": [f"Gap analysis failed: {e}"],
            "documentation_gaps": [],
            "gap_severity": "high",
            "agent_trace": trace,
            "error": str(e),
        }
