"""
ChromaDB vector store for insurance policy documents.
"""
from __future__ import annotations

import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from app.config import get_settings
from app.rag.embeddings import get_embedding_function

_settings = get_settings()

# ── Singleton client ──────────────────────────────────────────────────────────
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=_settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    """Return (creating if needed) the insurance-policy collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=_settings.chroma_collection_name,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{_settings.chroma_collection_name}' ready — "
                    f"{_collection.count()} docs")
    return _collection


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def add_policy_chunks(
    chunks: list[str],
    metadata_list: list[dict],
    ids: Optional[list[str]] = None,
) -> list[str]:
    """
    Upsert text chunks into ChromaDB.

    Args:
        chunks: List of text chunks.
        metadata_list: Parallel list of metadata dicts (source, page, policy_id, …).
        ids: Optional explicit IDs; auto-generated if omitted.

    Returns:
        List of IDs that were upserted.
    """
    collection = get_collection()
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in chunks]

    collection.upsert(documents=chunks, metadatas=metadata_list, ids=ids)
    logger.info(f"Upserted {len(chunks)} chunks into ChromaDB")
    return ids


def similarity_search(
    query: str,
    n_results: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for a query.

    Returns a list of dicts with keys: id, document, metadata, distance.
    """
    collection = get_collection()
    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — returning no results")
        return []

    kwargs: dict = {"query_texts": [query], "n_results": min(n_results, collection.count())}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    out = []
    for i, doc in enumerate(results["documents"][0]):
        out.append({
            "id": results["ids"][0][i],
            "document": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return out


def delete_policy(policy_id: str) -> int:
    """Delete all chunks belonging to a policy_id. Returns number deleted."""
    collection = get_collection()
    existing = collection.get(where={"policy_id": policy_id})
    if not existing["ids"]:
        return 0
    collection.delete(ids=existing["ids"])
    logger.info(f"Deleted {len(existing['ids'])} chunks for policy_id={policy_id}")
    return len(existing["ids"])


def collection_stats() -> dict:
    """Return basic stats about the collection."""
    collection = get_collection()
    return {
        "collection_name": collection.name,
        "total_chunks": collection.count(),
    }
