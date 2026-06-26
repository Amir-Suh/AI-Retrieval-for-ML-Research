"""Cross-encoder reranking with bge-reranker-v2-m3 (local, via sentence-transformers).

Unlike the bi-encoder retrieval channels (which embed query and document separately),
the cross-encoder reads (query, document) together and scores true relevance. It's slow,
so it only runs on the fused candidate shortlist, not the whole corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import settings
from src.retrieval.hybrid_search import Hit

_model = None


def _get_model():
    global _model
    if _model is None:
        # Imported lazily so the heavy torch import only happens when reranking is used.
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(settings.reranker_model, max_length=settings.reranker_max_length)
    return _model


def rerank(query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
    """Re-score `hits` with the cross-encoder and return the top_k, re-ranked.

    The returned Hits carry the cross-encoder score in `.score` and a fresh 1-based
    `.rank`; original retrieval order is available via the input list.
    """
    top_k = top_k or settings.top_k
    if not hits:
        return []

    model = _get_model()
    pairs = [(query, f"{h.title}\n\n{h.abstract}") for h in hits]
    scores = model.predict(pairs)

    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    reranked: list[Hit] = []
    for new_rank, i in enumerate(order[:top_k], start=1):
        h = hits[i]
        reranked.append(
            Hit(
                arxiv_id=h.arxiv_id,
                title=h.title,
                abstract=h.abstract,
                categories=h.categories,
                year=h.year,
                score=float(scores[i]),
                rank=new_rank,
            )
        )
    return reranked
