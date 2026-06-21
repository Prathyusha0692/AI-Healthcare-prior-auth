"""
OpenAI embedding wrapper with stub fallback for when no API key is set.
"""
from __future__ import annotations

from loguru import logger
from app.config import get_settings

settings = get_settings()

# ── Stub embeddings (used when OPENAI_API_KEY is a placeholder) ───────────────
import hashlib


def _stub_embed(text: str) -> list[float]:
    """Deterministic 1536-dim stub embedding based on text hash."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    import random
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(1536)]


def _is_real_key(key: str) -> bool:
    return key.startswith("sk-") and "placeholder" not in key and len(key) > 20


# ── Embedding function factory ────────────────────────────────────────────────

def get_embedding_function():
    """Return a ChromaDB-compatible embedding function."""
    if _is_real_key(settings.openai_api_key):
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name=settings.openai_embedding_model,
            )
        except Exception as e:
            logger.warning(f"OpenAI embedding init failed: {e}. Falling back to stub.")

    logger.warning("Using stub embeddings — add a real OPENAI_API_KEY for production")

    class StubEmbeddingFunction:
        def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
            return [_stub_embed(t) for t in input]

    return StubEmbeddingFunction()


def embed_text(text: str) -> list[float]:
    """Embed a single string, returning a float list."""
    fn = get_embedding_function()
    return fn([text])[0]
