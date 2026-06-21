"""
Shared LangGraph state schema for the prior authorization workflow.
All 4 agents read from and write to this TypedDict.
"""
from __future__ import annotations

from typing import Optional, TypedDict


class AuthState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    request_id: str                  # Unique ID for this authorization run
    patient_id: str                  # (scrubbed) patient identifier
    clinical_note: str               # Raw clinical note (pre-scrub)
    scrubbed_note: str               # PHI-redacted clinical note
    procedure_code: str              # CPT / ICD code being requested
    diagnosis_code: str              # ICD-10 diagnosis code
    insurance_plan: str              # Insurance plan name / ID
    policy_id: Optional[str]         # ChromaDB policy document ID (if known)

    # ── Policy Agent output ───────────────────────────────────────────────────
    policy_clauses: list[dict]       # Retrieved policy clauses from ChromaDB
    policy_summary: str              # LLM-generated summary of applicable rules
    coverage_criteria: list[str]     # Explicit criteria that must be met

    # ── Clinical Agent output ─────────────────────────────────────────────────
    clinical_summary: str            # Structured summary of the clinical note
    clinical_evidence: list[str]     # Evidence supporting medical necessity
    clinical_flags: list[str]        # Red flags or missing information

    # ── Gap Detector output ───────────────────────────────────────────────────
    coverage_gaps: list[str]         # Criteria not met by clinical documentation
    documentation_gaps: list[str]    # Missing documents / info
    gap_severity: str                # "low" | "medium" | "high" | "critical"

    # ── Recommendation Agent output ───────────────────────────────────────────
    authorization_decision: str      # "approved" | "denied" | "pending_info"
    confidence_score: float          # 0.0 – 1.0
    recommendation_rationale: str    # Human-readable rationale
    required_actions: list[str]      # Next steps (e.g., "submit MRI report")
    appeal_guidance: str             # Guidance if decision is denial

    # ── Metadata ──────────────────────────────────────────────────────────────
    error: Optional[str]             # Any error message from the pipeline
    agent_trace: list[str]           # Execution log for debugging
