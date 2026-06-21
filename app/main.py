"""
FastAPI application entry point.

Run with:  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.database import init_db
from app.models import HealthResponse
from app.rag.vector_store import collection_stats

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting AI-Powered Prior Authorization Platform…")
    init_db()
    logger.info("SQLite database initialised")
    yield
    logger.info("Shutting down…")


app = FastAPI(
    title="AI-Powered Healthcare Prior Authorization Platform",
    description=(
        "Multi-agent LangGraph system for automated insurance prior authorization analysis. "
        "Features: RAG with ChromaDB, HIPAA PII scrubbing (Presidio), "
        "4 specialized agents (Policy, Clinical, Gap Detector, Recommendation), "
        "and full audit persistence."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.routes.upload import router as upload_router
from app.api.routes.analysis import router as analysis_router
from app.api.routes.clinical import router as clinical_router
from app.api.routes.history import router as history_router

PREFIX = "/api/v1"
app.include_router(upload_router, prefix=PREFIX)
app.include_router(analysis_router, prefix=PREFIX)
app.include_router(clinical_router, prefix=PREFIX)
app.include_router(history_router, prefix=PREFIX)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """System health check — confirms DB and ChromaDB are accessible."""
    try:
        stats = collection_stats()
        chroma_ok = True
    except Exception as e:
        stats = {"error": str(e)}
        chroma_ok = False

    try:
        from app.database import SessionLocal
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    return HealthResponse(
        status="healthy" if chroma_ok and db_status == "ok" else "degraded",
        version="1.0.0",
        chroma_stats=stats,
        db_status=db_status,
    )


@app.get("/", tags=["System"])
def root():
    return {
        "message": "AI-Powered Healthcare Prior Authorization Platform",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }
