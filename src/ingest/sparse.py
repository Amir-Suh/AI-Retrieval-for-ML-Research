"""BM25 sparse vectors via FastEmbed — the keyword channel for hybrid search.

FastEmbed's `Qdrant/bm25` produces sparse vectors (token-id -> weight) that pair with
Qdrant's IDF modifier on the collection's sparse vector. The model is loaded lazily and
reused across calls (it downloads a small vocabulary on first use).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastembed import SparseTextEmbedding

from config import settings

_model: SparseTextEmbedding | None = None


def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name=settings.sparse_model)
    return _model


def embed_sparse_documents(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """Return (indices, values) pairs for each document, ready for Qdrant SparseVector."""
    model = _get_model()
    out: list[tuple[list[int], list[float]]] = []
    for emb in model.embed(texts):
        out.append((emb.indices.tolist(), emb.values.tolist()))
    return out


def embed_sparse_query(query: str) -> tuple[list[int], list[float]]:
    """Sparse-embed a single query (uses the query-side method for correct weighting)."""
    model = _get_model()
    emb = next(model.query_embed(query))
    return emb.indices.tolist(), emb.values.tolist()
