"""
POST /api/v1/authorize — run the full multi-agent prior authorization pipeline.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.orm import Session

from app.agents.graph import run_authorization
from app.database import AuthorizationRun, get_db
from app.models import AuthorizationRequest, AuthorizationResponse

router = APIRouter(tags=["Authorization"])


@router.post("/authorize", response_model=AuthorizationResponse)
def authorize(request: AuthorizationRequest, db: Session = Depends(get_db)):
    """
    Run the 4-agent prior authorization pipeline:
      1. Clinical Agent — PHI scrub + clinical analysis
      2. Policy Agent — RAG retrieval + policy summary
      3. Gap Detector Agent — coverage gap analysis
      4. Recommendation Agent — final decision + rationale

    Every run is persisted to SQLite for audit purposes.
    """
    logger.info(
        f"Authorization request: procedure={request.procedure_code} "
        f"diagnosis={request.diagnosis_code} plan={request.insurance_plan}"
    )

    try:
        state = run_authorization(
            clinical_note=request.clinical_note,
            procedure_code=request.procedure_code,
            diagnosis_code=request.diagnosis_code,
            insurance_plan=request.insurance_plan,
            patient_id=request.patient_id or "ANONYMOUS",
            policy_id=request.policy_id,
            request_id=request.request_id,
        )
    except Exception as e:
        logger.error(f"Authorization pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Authorization pipeline error: {e}")

    # ── Persist to SQLite ─────────────────────────────────────────────────────
    run = AuthorizationRun(
        request_id=state["request_id"],
        patient_id=state.get("patient_id"),
        procedure_code=request.procedure_code,
        diagnosis_code=request.diagnosis_code,
        insurance_plan=request.insurance_plan,
        policy_id=request.policy_id,
        authorization_decision=state.get("authorization_decision"),
        confidence_score=state.get("confidence_score"),
        recommendation_rationale=state.get("recommendation_rationale"),
        gap_severity=state.get("gap_severity"),
        policy_summary=state.get("policy_summary"),
        clinical_summary=state.get("clinical_summary"),
        coverage_criteria=state.get("coverage_criteria", []),
        clinical_evidence=state.get("clinical_evidence", []),
        clinical_flags=state.get("clinical_flags", []),
        coverage_gaps=state.get("coverage_gaps", []),
        documentation_gaps=state.get("documentation_gaps", []),
        required_actions=state.get("required_actions", []),
        appeal_guidance=state.get("appeal_guidance", ""),
        agent_trace=state.get("agent_trace", []),
        error=state.get("error"),
        created_at=datetime.utcnow(),
    )
    db.merge(run)
    db.commit()

    return AuthorizationResponse(
        request_id=state["request_id"],
        authorization_decision=state.get("authorization_decision", "pending_info"),
        confidence_score=state.get("confidence_score", 0.0),
        recommendation_rationale=state.get("recommendation_rationale", ""),
        policy_summary=state.get("policy_summary", ""),
        coverage_criteria=state.get("coverage_criteria", []),
        clinical_summary=state.get("clinical_summary", ""),
        clinical_evidence=state.get("clinical_evidence", []),
        clinical_flags=state.get("clinical_flags", []),
        coverage_gaps=state.get("coverage_gaps", []),
        documentation_gaps=state.get("documentation_gaps", []),
        gap_severity=state.get("gap_severity", "unknown"),
        required_actions=state.get("required_actions", []),
        appeal_guidance=state.get("appeal_guidance", ""),
        policy_clauses_retrieved=len(state.get("policy_clauses", [])),
        agent_trace=state.get("agent_trace", []),
        created_at=datetime.utcnow(),
    )
