"""
Recommendation Agent — synthesises all upstream agent outputs to produce
a final prior authorization decision with rationale and next steps.
"""
from __future__ import annotations

import json
from loguru import logger

from app.agents.state import AuthState
from app.agents.llm import get_llm, StubLLM

SYSTEM_PROMPT = """You are a senior prior authorization decision-maker at a health insurance company.

You will receive the full analysis produced by three specialist agents:
- Policy Agent: applicable coverage criteria
- Clinical Agent: evidence from the clinical note
- Gap Detector Agent: unmet criteria and missing documentation

Your task: synthesise all findings into a final authorization recommendation.

Output ONLY valid JSON in this exact schema:
{
  "authorization_decision": "<approved|denied|pending_info>",
  "confidence_score": <float 0.0–1.0>,
  "recommendation_rationale": "<clear, professional 3–5 sentence rationale>",
  "required_actions": ["<action 1>", "<action 2>", ...],
  "appeal_guidance": "<guidance if decision is denial or pending, otherwise empty string>"
}

Decision rules:
  approved     — all criteria met, no critical gaps
  pending_info — criteria partially met; specific information needed
  denied       — fundamental coverage criteria unmet regardless of documentation
"""


def recommendation_agent(state: AuthState) -> AuthState:
    """
    LangGraph node: Recommendation Agent.

    Reads: policy_summary, coverage_criteria, clinical_summary,
           clinical_evidence, clinical_flags, coverage_gaps,
           documentation_gaps, gap_severity
    Writes: authorization_decision, confidence_score,
            recommendation_rationale, required_actions, appeal_guidance
    """
    logger.info(f"[RecommendationAgent] Starting for request_id={state.get('request_id')}")
    trace = list(state.get("agent_trace", []))
    trace.append("RecommendationAgent: started")

    try:
        llm = get_llm()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"=== REQUEST DETAILS ===\n"
                    f"Procedure: {state.get('procedure_code', 'Unknown')}\n"
                    f"Diagnosis: {state.get('diagnosis_code', 'Unknown')}\n"
                    f"Insurance Plan: {state.get('insurance_plan', 'Unknown')}\n\n"
                    f"=== POLICY ANALYSIS ===\n"
                    f"Summary: {state.get('policy_summary', '')}\n"
                    f"Coverage Criteria:\n"
                    + "\n".join(f"- {c}" for c in state.get("coverage_criteria", []))
                    + f"\n\n=== CLINICAL ANALYSIS ===\n"
                    f"Summary: {state.get('clinical_summary', '')}\n"
                    f"Evidence:\n"
                    + "\n".join(f"- {e}" for e in state.get("clinical_evidence", []))
                    + f"\nFlags:\n"
                    + "\n".join(f"- {f}" for f in state.get("clinical_flags", []))
                    + f"\n\n=== GAP ANALYSIS ===\n"
                    f"Coverage Gaps:\n"
                    + "\n".join(f"- {g}" for g in state.get("coverage_gaps", []))
                    + f"\nDocumentation Gaps:\n"
                    + "\n".join(f"- {d}" for d in state.get("documentation_gaps", []))
                    + f"\nOverall Severity: {state.get('gap_severity', 'unknown')}"
                ),
            },
        ]

        if isinstance(llm, StubLLM):
            response = llm.invoke(messages, agent_type="recommendation")
        else:
            response = llm.invoke(messages)

        parsed = json.loads(response.content)
        decision = parsed.get("authorization_decision", "pending_info")
        confidence = float(parsed.get("confidence_score", 0.5))
        trace.append(f"RecommendationAgent: decision={decision}, confidence={confidence:.2f}")

        return {
            **state,
            "authorization_decision": decision,
            "confidence_score": confidence,
            "recommendation_rationale": parsed.get("recommendation_rationale", ""),
            "required_actions": parsed.get("required_actions", []),
            "appeal_guidance": parsed.get("appeal_guidance", ""),
            "agent_trace": trace,
        }

    except Exception as e:
        logger.error(f"[RecommendationAgent] Error: {e}")
        trace.append(f"RecommendationAgent: ERROR — {e}")
        return {
            **state,
            "authorization_decision": "pending_info",
            "confidence_score": 0.0,
            "recommendation_rationale": f"Recommendation generation failed: {e}",
            "required_actions": ["Manual review required"],
            "appeal_guidance": "",
            "agent_trace": trace,
            "error": str(e),
        }
