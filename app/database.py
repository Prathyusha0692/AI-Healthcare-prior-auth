"""
SQLite database setup via SQLAlchemy.
Every analysis run is persisted to the `authorization_runs` table.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

_settings = get_settings()

engine = create_engine(
    _settings.database_url,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class AuthorizationRun(Base):
    """Persists every prior authorization analysis run."""

    __tablename__ = "authorization_runs"

    request_id = Column(String, primary_key=True, index=True)
    patient_id = Column(String, nullable=True)
    procedure_code = Column(String, nullable=False)
    diagnosis_code = Column(String, nullable=False)
    insurance_plan = Column(String, nullable=False)
    policy_id = Column(String, nullable=True)

    # Agent outputs
    authorization_decision = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    recommendation_rationale = Column(Text, nullable=True)
    gap_severity = Column(String, nullable=True)
    policy_summary = Column(Text, nullable=True)
    clinical_summary = Column(Text, nullable=True)

    # JSON columns for list fields
    coverage_criteria = Column(JSON, nullable=True)
    clinical_evidence = Column(JSON, nullable=True)
    clinical_flags = Column(JSON, nullable=True)
    coverage_gaps = Column(JSON, nullable=True)
    documentation_gaps = Column(JSON, nullable=True)
    required_actions = Column(JSON, nullable=True)
    appeal_guidance = Column(Text, nullable=True)
    agent_trace = Column(JSON, nullable=True)

    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PolicyDocument(Base):
    """Metadata for uploaded insurance policy PDFs."""

    __tablename__ = "policy_documents"

    policy_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    insurance_plan = Column(String, nullable=True)
    chunk_count = Column(String, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
