"""
Policy Agent — retrieves relevant insurance policy clauses from ChromaDB
and summarises coverage criteria applicable to the authorization request.
"""
from __future__ import annotations

import json
from loguru import logger

from app.agents.state import AuthState
from app.agents.llm import get_llm, StubLLM
from app.rag.retriever import retrieve_policy_clauses

SYSTEM_PROMPT = """You are a healthcare insurance policy analysis expert.

Given a prior authorization request, your job is to:
1. Review the retrieved insurance policy clauses provided in the context.
2. Summarize the policy coverage rules relevant to the requested procedure.
3. Extract a clear list of criteria the patient/clinician must satisfy for authorization.

Output ONLY valid JSON in this exact schema:
{
  "policy_summary": "<brief summary of applicable policy rules>",
  "coverage_criteria": ["<criterion 1>", "<criterion 2>", ...]
}
"""


def policy_agent(state: AuthState) -> AuthState:
    """
    LangGraph node: Policy Agent.

    Reads: clinical_note, procedure_code, diagnosis_code, insurance_plan, policy_id
    Writes: policy_clauses, policy_summary, coverage_criteria
    """
    logger.info(f"[PolicyAgent] Starting for request_id={state.get('request_id')}")
    trace = list(state.get("agent_trace", []))
    trace.append("PolicyAgent: started")

    try:
        # Build a rich query combining procedure + diagnosis context
        query = (
            f"Insurance coverage policy for procedure {state.get('procedure_code', '')} "
            f"diagnosis {state.get('diagnosis_code', '')} "
            f"plan {state.get('insurance_plan', '')}. "
            f"{state.get('scrubbed_note', '')[:300]}"
        )

        retrieval = retrieve_policy_clauses(
            query=query,
            n_results=5,
            policy_id=state.get("policy_id"),
        )

        policy_clauses = [
            {
                "text": c.clause_text,
                "source": c.source,
                "page": c.page,
                "policy_id": c.policy_id,
                "relevance": round(1 - c.relevance_score, 3),
            }
            for c in retrieval.clauses
        ]

        llm = get_llm()
        context_block = retrieval.as_context_block()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Insurance Plan: {state.get('insurance_plan', 'Unknown')}\n"
                    f"Procedure Code: {state.get('procedure_code', 'Unknown')}\n"
                    f"Diagnosis Code: {state.get('diagnosis_code', 'Unknown')}\n\n"
                    f"{context_block}"
                ),
            },
        ]

        if isinstance(llm, StubLLM):
            response = llm.invoke(messages, agent_type="policy")
        else:
            response = llm.invoke(messages)

        parsed = json.loads(response.content)
        trace.append(f"PolicyAgent: retrieved {len(policy_clauses)} clauses, parsed LLM response")

        return {
            **state,
            "policy_clauses": policy_clauses,
            "policy_summary": parsed.get("policy_summary", ""),
            "coverage_criteria": parsed.get("coverage_criteria", []),
            "agent_trace": trace,
        }

    except Exception as e:
        logger.error(f"[PolicyAgent] Error: {e}")
        trace.append(f"PolicyAgent: ERROR — {e}")
        return {
            **state,
            "policy_clauses": [],
            "policy_summary": f"Policy analysis failed: {e}",
            "coverage_criteria": [],
            "agent_trace": trace,
            "error": str(e),
        }
