"""Retrieval pipeline orchestration: query -> stages -> top-k.

Returns every intermediate stage (dense, sparse, RRF-fused, reranked) so the UI can
visualize how the ranking evolves. The production "answer" is the reranked top-k.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import settings
from src.retrieval import hybrid_search
from src.retrieval.rerank import rerank


def search(query: str, use_rerank: bool = True, display_limit: int = 10) -> dict:
    """Run all retrieval stages and return them as JSON-serializable dicts."""
    timings: dict[str, float] = {}

    t = time.perf_counter()
    dense = hybrid_search.dense_search(query, limit=settings.fusion_candidates)
    timings["dense_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    sparse = hybrid_search.sparse_search(query, limit=settings.fusion_candidates)
    timings["sparse_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    fused = hybrid_search.fused_search(query, limit=settings.rerank_candidates)
    timings["fused_ms"] = (time.perf_counter() - t) * 1000

    fused_rank = {h.arxiv_id: h.rank for h in fused}

    reranked_out: list[dict] = []
    if use_rerank:
        t = time.perf_counter()
        reranked = rerank(query, fused, top_k=settings.top_k)
        timings["rerank_ms"] = (time.perf_counter() - t) * 1000
        for h in reranked:
            d = h.to_dict()
            # how far this paper moved when the cross-encoder re-scored it
            d["fused_rank"] = fused_rank.get(h.arxiv_id)
            reranked_out.append(d)
    else:
        for h in fused[: settings.top_k]:
            d = h.to_dict()
            d["fused_rank"] = h.rank
            reranked_out.append(d)

    return {
        "query": query,
        "use_rerank": use_rerank,
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "config": {
            "fusion_candidates": settings.fusion_candidates,
            "rerank_candidates": settings.rerank_candidates,
            "top_k": settings.top_k,
        },
        "stages": {
            "dense": [h.to_dict() for h in dense[:display_limit]],
            "sparse": [h.to_dict() for h in sparse[:display_limit]],
            "fused": [h.to_dict() for h in fused[:display_limit]],
            "reranked": reranked_out,
        },
    }


if __name__ == "__main__":
    import json

    q = " ".join(sys.argv[1:]) or "self-supervised vision transformers for dense prediction"
    result = search(q)
    print(json.dumps(result["timings_ms"], indent=2))
    print(f"\nQuery: {q}\n")
    print("=== Reranked top-5 ===")
    for h in result["stages"]["reranked"]:
        print(f"  #{h['rank']} (was #{h['fused_rank']} fused)  "
              f"[{h['score']:.3f}]  {h['arxiv_id']}  {h['title'][:70]}")
