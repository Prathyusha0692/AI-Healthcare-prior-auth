"""
RAG retriever: given a clinical note or authorization request,
retrieve the most relevant insurance policy clauses from ChromaDB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from app.rag.vector_store import similarity_search


@dataclass
class RetrievedClause:
    clause_text: str
    source: str
    page: int
    policy_id: str
    relevance_score: float  # 0 (identical) → 1 (distant) for cosine distance


@dataclass
class RetrievalResult:
    query: str
    clauses: list[RetrievedClause] = field(default_factory=list)
    total_retrieved: int = 0

    def as_context_block(self) -> str:
        """Format retrieved clauses into a single LLM context string."""
        if not self.clauses:
            return "No relevant policy clauses found."

        lines = ["=== RETRIEVED POLICY CLAUSES ==="]
        for i, c in enumerate(self.clauses, 1):
            lines.append(
                f"\n[Clause {i}] Source: {c.source} | Page: {c.page} | "
                f"Policy ID: {c.policy_id} | Relevance: {1 - c.relevance_score:.2%}"
            )
            lines.append(c.clause_text)
            lines.append("─" * 60)
        return "\n".join(lines)


def retrieve_policy_clauses(
    query: str,
    n_results: int = 5,
    policy_id: Optional[str] = None,
    min_relevance: float = 0.7,  # cosine similarity threshold (1 - distance)
) -> RetrievalResult:
    """
    Retrieve the most relevant policy clauses for a given query.

    Args:
        query: The clinical note or authorization request text.
        n_results: Number of chunks to retrieve.
        policy_id: Optionally filter to a specific policy.
        min_relevance: Minimum cosine similarity (0–1) to include a result.

    Returns:
        RetrievalResult with ranked clauses.
    """
    where = {"policy_id": policy_id} if policy_id else None
    raw = similarity_search(query, n_results=n_results, where=where)

    clauses = []
    for r in raw:
        similarity = 1.0 - r["distance"]
        if similarity < min_relevance:
            logger.debug(f"Skipping chunk (sim={similarity:.3f} < threshold {min_relevance})")
            continue

        clauses.append(
            RetrievedClause(
                clause_text=r["document"],
                source=r["metadata"].get("source", "unknown"),
                page=int(r["metadata"].get("page", 0)),
                policy_id=r["metadata"].get("policy_id", "unknown"),
                relevance_score=r["distance"],
            )
        )

    logger.info(f"Retrieved {len(clauses)} clauses above threshold for query: {query[:80]}…")
    return RetrievalResult(query=query, clauses=clauses, total_retrieved=len(clauses))


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split a long document into overlapping chunks for indexing.

    Args:
        text: Raw document text.
        chunk_size: Target characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += chunk_size - overlap
    return [c for c in chunks if c]
