"""
GET /api/v1/history        — paginated audit history of all authorization runs
GET /api/v1/history/{id}   — fetch a single run by request_id
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import AuthorizationRun, get_db
from app.models import AuditHistoryResponse, AuditRecord, AuthorizationResponse

router = APIRouter(prefix="/history", tags=["Audit History"])


@router.get("", response_model=AuditHistoryResponse)
def get_history(
    skip: int = 0,
    limit: int = 20,
    decision: str = None,
    db: Session = Depends(get_db),
):
    """
    Return paginated audit history.

    Query params:
      skip     — offset (default 0)
      limit    — max records (default 20, max 100)
      decision — filter by authorization_decision (approved/denied/pending_info)
    """
    limit = min(limit, 100)
    query = db.query(AuthorizationRun)

    if decision:
        query = query.filter(AuthorizationRun.authorization_decision == decision)

    total = query.count()
    runs = query.order_by(AuthorizationRun.created_at.desc()).offset(skip).limit(limit).all()

    return AuditHistoryResponse(
        total=total,
        records=[
            AuditRecord(
                request_id=r.request_id,
                patient_id=r.patient_id,
                procedure_code=r.procedure_code,
                diagnosis_code=r.diagnosis_code,
                insurance_plan=r.insurance_plan,
                authorization_decision=r.authorization_decision,
                confidence_score=r.confidence_score,
                gap_severity=r.gap_severity,
                created_at=r.created_at,
            )
            for r in runs
        ],
    )


@router.get("/{request_id}", response_model=AuthorizationResponse)
def get_run(request_id: str, db: Session = Depends(get_db)):
    """Retrieve the full analysis result for a specific request_id."""
    run = db.query(AuthorizationRun).filter(AuthorizationRun.request_id == request_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"No run found for request_id={request_id}")

    return AuthorizationResponse(
        request_id=run.request_id,
        authorization_decision=run.authorization_decision or "unknown",
        confidence_score=run.confidence_score or 0.0,
        recommendation_rationale=run.recommendation_rationale or "",
        policy_summary=run.policy_summary or "",
        coverage_criteria=run.coverage_criteria or [],
        clinical_summary=run.clinical_summary or "",
        clinical_evidence=run.clinical_evidence or [],
        clinical_flags=run.clinical_flags or [],
        coverage_gaps=run.coverage_gaps or [],
        documentation_gaps=run.documentation_gaps or [],
        gap_severity=run.gap_severity or "unknown",
        required_actions=run.required_actions or [],
        appeal_guidance=run.appeal_guidance or "",
        policy_clauses_retrieved=0,
        agent_trace=run.agent_trace or [],
        created_at=run.created_at,
    )


@router.delete("/{request_id}", tags=["Audit History"])
def delete_run(request_id: str, db: Session = Depends(get_db)):
    """Delete a single audit run by request_id."""
    deleted = db.query(AuthorizationRun).filter(
        AuthorizationRun.request_id == request_id
    ).delete()
    db.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No run found for request_id={request_id}")
    return {"message": f"Deleted run {request_id}"}
