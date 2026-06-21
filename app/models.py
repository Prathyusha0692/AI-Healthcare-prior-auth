"""
Pydantic request / response models for the FastAPI API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────

class AuthorizationRequest(BaseModel):
    clinical_note: str = Field(..., description="Full clinical note text (PHI will be auto-scrubbed)")
    procedure_code: str = Field(..., description="CPT code for the requested procedure", examples=["27447"])
    diagnosis_code: str = Field(..., description="ICD-10 diagnosis code", examples=["M17.11"])
    insurance_plan: str = Field(..., description="Insurance plan name or ID", examples=["BlueCross PPO Gold"])
    patient_id: Optional[str] = Field(None, description="Optional patient identifier for tracking")
    policy_id: Optional[str] = Field(None, description="Restrict policy retrieval to this policy ID")
    request_id: Optional[str] = Field(None, description="Optional explicit request ID")


class ScrubRequest(BaseModel):
    text: str = Field(..., description="Text to scrub for PHI/PII")


# ── Response models ───────────────────────────────────────────────────────────

class ScrubResponse(BaseModel):
    scrubbed_text: str
    phi_detected: bool
    entity_counts: dict[str, int]
    scrubbing_engine: str
    summary: str


class PolicyClause(BaseModel):
    text: str
    source: str
    page: int
    policy_id: str
    relevance: float


class AuthorizationResponse(BaseModel):
    request_id: str
    authorization_decision: str  # "approved" | "denied" | "pending_info"
    confidence_score: float
    recommendation_rationale: str

    # Policy
    policy_summary: str
    coverage_criteria: list[str]

    # Clinical
    clinical_summary: str
    clinical_evidence: list[str]
    clinical_flags: list[str]

    # Gaps
    coverage_gaps: list[str]
    documentation_gaps: list[str]
    gap_severity: str

    # Actions
    required_actions: list[str]
    appeal_guidance: str

    # Meta
    policy_clauses_retrieved: int
    agent_trace: list[str]
    created_at: Optional[datetime] = None


class PolicyUploadResponse(BaseModel):
    policy_id: str
    filename: str
    chunks_indexed: int
    message: str


class AuditRecord(BaseModel):
    request_id: str
    patient_id: Optional[str]
    procedure_code: str
    diagnosis_code: str
    insurance_plan: str
    authorization_decision: Optional[str]
    confidence_score: Optional[float]
    gap_severity: Optional[str]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class AuditHistoryResponse(BaseModel):
    total: int
    records: list[AuditRecord]


class HealthResponse(BaseModel):
    status: str
    version: str
    chroma_stats: dict
    db_status: str
