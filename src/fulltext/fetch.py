"""Fetch a paper's full text from arXiv — LaTeX source first, PDF fallback.

Only ever called on the handful of papers that survive reranking, so this stays
deliberately simple: polite single-threaded fetching with a descriptive User-Agent,
on-disk caching, and light rate limiting.

arXiv endpoints:
  - source: https://arxiv.org/e-print/{id}   (tar.gz, gzipped .tex, or sometimes a PDF)
  - pdf:    https://arxiv.org/pdf/{id}
"""

from __future__ import annotations

import gzip
import io
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

CACHE_ROOT = Path("fulltext_cache")
USER_AGENT = "arxiv-rag-pipeline/0.1 (https://github.com/Amir-Suh/AI-Retrieval-for-ML-Research; mailto:amirsyedsuhail@gmail.com)"
EPRINT_URL = "https://arxiv.org/e-print/{id}"
PDF_URL = "https://arxiv.org/pdf/{id}"

_MIN_INTERVAL = 3.0  # seconds between network fetches (arXiv etiquette)
_last_fetch = 0.0


@dataclass
class FetchResult:
    arxiv_id: str
    source_type: str          # "latex" | "pdf"
    cache_dir: Path
    tex_files: list[Path]     # populated when source_type == "latex"
    pdf_path: Path | None     # populated when source_type == "pdf"
    note: str = ""


def _throttle() -> None:
    global _last_fetch
    wait = _last_fetch + _MIN_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_fetch = time.monotonic()


def _safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _download(url: str) -> bytes:
    _throttle()
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return resp.content


def _extract_latex(raw: bytes, src_dir: Path) -> list[Path]:
    """Materialize .tex files from an e-print payload (tar.gz or single gzipped .tex)."""
    src_dir.mkdir(parents=True, exist_ok=True)

    # Case 1: a tar archive (possibly gzip-compressed).
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:  # auto-detects gz
            tex_files = []
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                # guard against path traversal
                name = Path(member.name).name
                if name.lower().endswith((".tex", ".ltx")):
                    data = tar.extractfile(member).read()
                    out = src_dir / name
                    out.write_bytes(data)
                    tex_files.append(out)
            if tex_files:
                return tex_files
    except (tarfile.TarError, OSError):
        pass

    # Case 2: a single gzipped .tex file.
    try:
        data = gzip.decompress(raw)
        out = src_dir / "main.tex"
        out.write_bytes(data)
        return [out]
    except (OSError, EOFError):
        pass

    # Case 3: plain text already.
    out = src_dir / "main.tex"
    out.write_bytes(raw)
    return [out]


def fetch_paper(arxiv_id: str, refresh: bool = False) -> FetchResult:
    """Fetch + cache a paper's source (or PDF). Idempotent: reuses the cache."""
    cache_dir = CACHE_ROOT / _safe_id(arxiv_id)
    src_dir = cache_dir / "src"
    pdf_path = cache_dir / "paper.pdf"

    # Reuse cache if present.
    if not refresh:
        if src_dir.exists():
            tex = sorted(src_dir.glob("*.tex")) + sorted(src_dir.glob("*.ltx"))
            if tex:
                return FetchResult(arxiv_id, "latex", cache_dir, tex, None, "cached")
        if pdf_path.exists():
            return FetchResult(arxiv_id, "pdf", cache_dir, [], pdf_path, "cached")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try LaTeX source first.
    try:
        raw = _download(EPRINT_URL.format(id=arxiv_id))
        if raw[:4] == b"%PDF":
            pdf_path.write_bytes(raw)
            return FetchResult(arxiv_id, "pdf", cache_dir, [], pdf_path, "e-print returned PDF")
        tex_files = _extract_latex(raw, src_dir)
        if tex_files:
            return FetchResult(arxiv_id, "latex", cache_dir, tex_files, None, "fetched source")
    except requests.HTTPError as exc:
        note = f"e-print failed ({exc.response.status_code}); using PDF"
    except requests.RequestException as exc:
        note = f"e-print error ({exc}); using PDF"
    else:
        note = "no tex in source; using PDF"

    # Fallback: PDF.
    raw = _download(PDF_URL.format(id=arxiv_id))
    pdf_path.write_bytes(raw)
    return FetchResult(arxiv_id, "pdf", cache_dir, [], pdf_path, note)
