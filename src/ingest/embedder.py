"""Gemini dense embeddings for the arXiv RAG pipeline.

Wraps `gemini-embedding-001` with:
  - batched requests (config.embedding_batch_size)
  - MRL truncation to config.embedding_dim (768)
  - L2 normalization (required: truncated dims are NOT pre-normalized by Gemini,
    and we use cosine distance in Qdrant)
  - simple exponential-backoff retry on transient/rate-limit errors

Two task types matter:
  - RETRIEVAL_DOCUMENT  -> indexing abstracts (default here)
  - RETRIEVAL_QUERY     -> embedding a user query at search time
"""

from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from google import genai
from google.genai import types

from config import settings

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise SystemExit(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


_last_request_finish = 0.0


def _throttle(n_items: int) -> None:
    """Pace requests so we stay under embedding_rpm. Each text counts as one request,
    so a batch of n items must be spaced by at least n * 60/rpm seconds."""
    global _last_request_finish
    min_interval = n_items * 60.0 / settings.embedding_rpm
    wait = _last_request_finish + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request_finish = time.monotonic()


_RETRY_DELAY_RE = re.compile(r"retry(?:Delay|\s+in)['\":\s]*([0-9.]+)s", re.IGNORECASE)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract the server-suggested retry delay (RetryInfo) from a 429 error, if present."""
    match = _RETRY_DELAY_RE.search(str(exc))
    return float(match.group(1)) if match else None


def _embed_batch(texts: list[str], task_type: str, max_retries: int = 8) -> list[list[float]]:
    client = _get_client()
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=settings.embedding_dim,
    )
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            _throttle(len(texts))
            resp = client.models.embed_content(
                model=settings.embedding_model,
                contents=texts,
                config=config,
            )
            return [_l2_normalize(e.values) for e in resp.embeddings]
        except Exception as exc:  # noqa: BLE001 - retry on any transient API error
            if attempt == max_retries:
                raise
            # Prefer the server's RetryInfo delay; otherwise exponential backoff.
            server_delay = _retry_after_seconds(exc)
            wait = max(server_delay + 1.0, delay) if server_delay else delay
            print(f"  embed batch failed (attempt {attempt}/{max_retries}), "
                  f"waiting {wait:.0f}s (server asked {server_delay}s)...")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
    return []  # unreachable


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of texts, batching by config.embedding_batch_size."""
    out: list[list[float]] = []
    bs = settings.embedding_batch_size
    for i in range(0, len(texts), bs):
        out.extend(_embed_batch(texts[i : i + bs], task_type))
    return out


def embed_query(query: str) -> list[float]:
    """Embed a single user query (RETRIEVAL_QUERY task type)."""
    return _embed_batch([query], task_type="RETRIEVAL_QUERY")[0]
