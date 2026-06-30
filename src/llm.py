"""Shared Gemini client + structured-generation helper.

Centralizes the genai client and a retry wrapper that honors the API's RetryInfo
delay (we learned in Phase 1 that Gemini's 429s carry a server-suggested wait).
Used by the synthesis stages; embeddings have their own throttled path.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import settings

T = TypeVar("T", bound=BaseModel)

_client: genai.Client | None = None
_RETRY_DELAY_RE = re.compile(r"retry(?:Delay|\s+in)['\":\s]*([0-9.]+)s", re.IGNORECASE)


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise SystemExit("GEMINI_API_KEY is not set (see .env).")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _retry_after(exc: Exception) -> float | None:
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) if m else None


def generate_structured(prompt: str, schema: type[T], model: str | None = None,
                        temperature: float = 0.2, max_retries: int = 6) -> T:
    """Generate JSON conforming to `schema` (a Pydantic model) and return it parsed."""
    client = get_client()
    model = model or settings.map_model
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
    )
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            return resp.parsed
        except Exception as exc:  # noqa: BLE001 - retry transient/rate-limit errors
            if attempt == max_retries:
                raise
            server = _retry_after(exc)
            wait = max(server + 1.0, delay) if server else delay
            print(f"  generate failed (attempt {attempt}/{max_retries}), waiting {wait:.0f}s...")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")
