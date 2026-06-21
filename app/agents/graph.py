"""
LangGraph orchestration — wires the 4 agents into a sequential StateGraph.

Execution order:
  START → clinical_agent → policy_agent → gap_detector_agent → recommendation_agent → END

clinical and policy agents run first (independent; clinical also does PHI scrub),
then gap detector combines both outputs, then recommendation synthesises everything.
"""
from __future__ import annotations

import uuid
from typing import Optional

from langgraph.graph import StateGraph, END
from loguru import logger

from app.agents.state import AuthState
from app.agents.clinical_agent import clinical_agent
from app.agents.policy_agent import policy_agent
from app.agents.gap_detector_agent import gap_detector_agent
from app.agents.recommendation_agent import recommendation_agent


def _init_state(state: AuthState) -> AuthState:
    """Ensure required fields are initialised before the pipeline starts."""
    return {
        **state,
        "request_id": state.get("request_id") or str(uuid.uuid4()),
        "agent_trace": state.get("agent_trace") or [],
        "error": None,
    }


def build_graph() -> StateGraph:
    """Build and compile the LangGraph prior-authorization workflow."""
    graph = StateGraph(AuthState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("init", _init_state)
    graph.add_node("clinical_agent", clinical_agent)
    graph.add_node("policy_agent", policy_agent)
    graph.add_node("gap_detector_agent", gap_detector_agent)
    graph.add_node("recommendation_agent", recommendation_agent)

    # ── Edges (sequential pipeline) ───────────────────────────────────────────
    graph.set_entry_point("init")
    graph.add_edge("init", "clinical_agent")
    graph.add_edge("clinical_agent", "policy_agent")
    graph.add_edge("policy_agent", "gap_detector_agent")
    graph.add_edge("gap_detector_agent", "recommendation_agent")
    graph.add_edge("recommendation_agent", END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("LangGraph prior-authorization graph compiled")
    return _compiled_graph


def run_authorization(
    clinical_note: str,
    procedure_code: str,
    diagnosis_code: str,
    insurance_plan: str,
    patient_id: str = "PATIENT_001",
    policy_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> AuthState:
    """
    Entry point for running a full prior authorization analysis.

    Args:
        clinical_note: Raw clinical note text (PHI will be scrubbed inside the pipeline).
        procedure_code: CPT code for the requested procedure.
        diagnosis_code: ICD-10 diagnosis code.
        insurance_plan: Insurance plan name or ID.
        patient_id: Patient identifier (will be used for tracking only).
        policy_id: Optional ChromaDB policy document ID to restrict retrieval.
        request_id: Optional explicit request ID; auto-generated if omitted.

    Returns:
        Completed AuthState dict with all agent outputs.
    """
    graph = get_graph()

    initial_state: AuthState = {
        "request_id": request_id or str(uuid.uuid4()),
        "patient_id": patient_id,
        "clinical_note": clinical_note,
        "procedure_code": procedure_code,
        "diagnosis_code": diagnosis_code,
        "insurance_plan": insurance_plan,
        "policy_id": policy_id,
        "agent_trace": [],
    }

    logger.info(
        f"Running authorization pipeline: request_id={initial_state['request_id']} "
        f"procedure={procedure_code} diagnosis={diagnosis_code}"
    )

    final_state = graph.invoke(initial_state)
    logger.info(
        f"Pipeline complete: decision={final_state.get('authorization_decision')} "
        f"confidence={final_state.get('confidence_score', 0):.2f}"
    )
    return final_state
