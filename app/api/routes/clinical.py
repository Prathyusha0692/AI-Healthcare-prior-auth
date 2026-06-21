"""
POST /api/v1/clinical/scrub — standalone PHI scrubbing endpoint.
POST /api/v1/clinical/notes  — ingest and analyse a standalone clinical note.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.models import ScrubRequest, ScrubResponse
from app.pii.scrubber import scrub_phi

router = APIRouter(prefix="/clinical", tags=["Clinical"])


@router.post("/scrub", response_model=ScrubResponse)
def scrub_clinical_text(request: ScrubRequest):
    """
    HIPAA PII/PHI scrubbing — detects and redacts 10+ PHI entity types
    from clinical note text without running the full authorization pipeline.

    Useful for pre-processing or testing the scrubbing layer independently.
    """
    result = scrub_phi(request.text)
    return ScrubResponse(
        scrubbed_text=result.scrubbed_text,
        phi_detected=result.phi_detected,
        entity_counts=result.entity_counts,
        scrubbing_engine=result.scrubbing_engine,
        summary=result.summary,
    )


@router.post("/notes")
def ingest_clinical_note(request: ScrubRequest):
    """
    Ingest a clinical note: scrub PHI and return structured metadata.
    For a full authorization analysis, use POST /authorize instead.
    """
    result = scrub_phi(request.text)
    return {
        "original_length": len(request.text),
        "scrubbed_length": len(result.scrubbed_text),
        "scrubbed_text": result.scrubbed_text,
        "phi_summary": result.summary,
        "entity_counts": result.entity_counts,
        "scrubbing_engine": result.scrubbing_engine,
        "ready_for_analysis": True,
    }
