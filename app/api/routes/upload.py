"""
POST /api/v1/policies/upload — ingest a policy PDF into ChromaDB.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from app.database import PolicyDocument, get_db
from app.models import PolicyUploadResponse
from app.rag.retriever import chunk_text
from app.rag.vector_store import add_policy_chunks
from app.utils.pdf_parser import extract_pages_from_bytes

router = APIRouter(prefix="/policies", tags=["Policy Upload"])


@router.post("/upload", response_model=PolicyUploadResponse)
async def upload_policy_pdf(
    file: UploadFile = File(..., description="Insurance policy PDF"),
    insurance_plan: str = Form(..., description="Insurance plan name for this policy"),
    policy_id: str = Form(None, description="Optional explicit policy ID"),
    db: Session = Depends(get_db),
):
    """
    Upload an insurance policy PDF and index it into ChromaDB for RAG retrieval.

    - Extracts text page-by-page
    - Splits into overlapping chunks (512 chars, 64-char overlap)
    - Embeds and stores chunks in ChromaDB with source metadata
    - Persists policy metadata to SQLite
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    policy_id = policy_id or str(uuid.uuid4())
    logger.info(f"Uploading policy PDF: {file.filename} → policy_id={policy_id}")

    try:
        pages = extract_pages_from_bytes(content, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF extraction failed: {e}")

    # ── Chunk and index ───────────────────────────────────────────────────────
    chunks: list[str] = []
    metadatas: list[dict] = []

    for page_data in pages:
        page_text = page_data["text"]
        if not page_text.strip():
            continue

        page_chunks = chunk_text(page_text, chunk_size=512, overlap=64)
        for chunk in page_chunks:
            chunks.append(chunk)
            metadatas.append({
                "policy_id": policy_id,
                "source": file.filename,
                "page": page_data["page"],
                "insurance_plan": insurance_plan,
            })

    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF")

    add_policy_chunks(chunks, metadatas)

    # ── Persist metadata to SQLite ────────────────────────────────────────────
    doc = PolicyDocument(
        policy_id=policy_id,
        filename=file.filename,
        insurance_plan=insurance_plan,
        chunk_count=str(len(chunks)),
        uploaded_at=datetime.utcnow(),
    )
    db.merge(doc)
    db.commit()

    logger.info(f"Indexed {len(chunks)} chunks for policy_id={policy_id}")
    return PolicyUploadResponse(
        policy_id=policy_id,
        filename=file.filename,
        chunks_indexed=len(chunks),
        message=f"Policy '{file.filename}' indexed successfully with {len(chunks)} chunks.",
    )


@router.get("/list", tags=["Policy Upload"])
def list_policies(db: Session = Depends(get_db)):
    """List all uploaded policy documents."""
    docs = db.query(PolicyDocument).order_by(PolicyDocument.uploaded_at.desc()).all()
    return [
        {
            "policy_id": d.policy_id,
            "filename": d.filename,
            "insurance_plan": d.insurance_plan,
            "chunk_count": d.chunk_count,
            "uploaded_at": d.uploaded_at,
        }
        for d in docs
    ]


@router.delete("/{policy_id}", tags=["Policy Upload"])
def delete_policy(policy_id: str, db: Session = Depends(get_db)):
    """Delete a policy and all its ChromaDB chunks."""
    from app.rag.vector_store import delete_policy as chroma_delete

    deleted = chroma_delete(policy_id)
    db.query(PolicyDocument).filter(PolicyDocument.policy_id == policy_id).delete()
    db.commit()
    return {"message": f"Deleted {deleted} chunks for policy_id={policy_id}"}
