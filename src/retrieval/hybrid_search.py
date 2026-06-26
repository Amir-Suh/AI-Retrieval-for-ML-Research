"""Hybrid retrieval over the Qdrant collection.

Three search functions, all returning a uniform list of `Hit`:
  - dense_search   : semantic (Gemini embedding) nearest neighbours
  - sparse_search  : BM25 keyword matches
  - fused_search   : both channels merged with Qdrant's native Reciprocal Rank Fusion

The dense/sparse functions exist so the UI can show each channel separately; the
production pipeline uses fused_search + reranking.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qdrant_client import QdrantClient, models

from config import settings
from src.ingest.embedder import embed_query
from src.ingest.sparse import embed_sparse_query


@dataclass
class Hit:
    arxiv_id: str
    title: str
    abstract: str
    categories: list[str]
    year: int | None
    score: float       # channel-specific score (cosine / BM25 / RRF)
    rank: int          # 1-based position within this channel

    def to_dict(self) -> dict:
        return asdict(self)


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _client


def _to_hits(points) -> list[Hit]:
    hits: list[Hit] = []
    for i, p in enumerate(points, start=1):
        pl = p.payload or {}
        hits.append(
            Hit(
                arxiv_id=pl.get("arxiv_id", ""),
                title=pl.get("title", ""),
                abstract=pl.get("abstract", ""),
                categories=pl.get("categories", []),
                year=pl.get("year"),
                score=float(p.score),
                rank=i,
            )
        )
    return hits


def dense_search(query: str, limit: int | None = None) -> list[Hit]:
    limit = limit or settings.fusion_candidates
    qvec = embed_query(query)
    res = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=qvec,
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return _to_hits(res.points)


def sparse_search(query: str, limit: int | None = None) -> list[Hit]:
    limit = limit or settings.fusion_candidates
    idx, val = embed_sparse_query(query)
    res = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=models.SparseVector(indices=idx, values=val),
        using="bm25",
        limit=limit,
        with_payload=True,
    )
    return _to_hits(res.points)


def fused_search(query: str, limit: int | None = None) -> list[Hit]:
    """Dense + sparse prefetch, merged with Qdrant's native RRF."""
    limit = limit or settings.rerank_candidates
    prefetch_n = settings.fusion_candidates
    qvec = embed_query(query)
    idx, val = embed_sparse_query(query)
    res = get_client().query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(query=qvec, using="dense", limit=prefetch_n),
            models.Prefetch(
                query=models.SparseVector(indices=idx, values=val),
                using="bm25",
                limit=prefetch_n,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return _to_hits(res.points)
