"""Central configuration for the arXiv RAG pipeline.

Single source of truth for model names, dimensions, retrieval constants, the
category filter, and the Qdrant connection. Values are read from environment
variables (loaded from a local `.env`) with sensible defaults.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets / connection ---
    gemini_api_key: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "arxiv_papers"

    # --- Corpus filter (Phase 1) ---
    # An arXiv paper is kept if ANY of its categories is in this set.
    ml_categories: frozenset[str] = frozenset(
        {"cs.CV", "cs.LG", "cs.AI", "cs.CL", "cs.NE", "stat.ML"}
    )

    # --- Embeddings (dense channel) ---
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768          # MRL-truncated from the 3072 default
    embedding_batch_size: int = 100   # texts per embed request

    # --- Sparse channel ---
    sparse_model: str = "Qdrant/bm25"

    # --- Reranker (Phase 2) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Generation (Phase 4) ---
    map_model: str = "gemini-3-flash"   # per-paper structured extraction
    reduce_model: str = "gemini-3-pro"  # final literature-review synthesis

    # --- Retrieval constants ---
    fusion_candidates: int = 100   # top-N pulled from EACH channel before RRF
    rerank_candidates: int = 50    # fused candidates fed to the cross-encoder
    top_k: int = 5                 # final papers sent to synthesis

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()
