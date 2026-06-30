"""Phase 4 Map stage: per-section summaries for each top-k paper.

For each parsed paper, send its detected sections to Gemini in a single call and get
back a concise summary per section that PRESERVES concrete specifics (dataset sizes,
metric names + values, loss functions, model names). Papers are summarized
concurrently and results are cached.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel

from config import settings
from src.fulltext.parse import ParsedPaper, parse_paper
from src.llm import generate_structured


class SectionSummary(BaseModel):
    section: str   # canonical section name (echoed back)
    summary: str   # 2-4 sentence summary preserving concrete details


class PaperSummary(BaseModel):
    summaries: list[SectionSummary]


_PROMPT = """You are summarizing one machine-learning research paper, section by section.

For EACH section provided below, write a concise 2-4 sentence summary. CRITICAL: preserve
concrete technical specifics verbatim where present — dataset names and sizes (e.g. "69,000
images"), metric names and values (e.g. "APCER 2.3%"), loss functions, model/architecture
names, and hyperparameters. Do not generalize these away. For each section, set `section`
to the plain section name shown in brackets below, WITHOUT the surrounding brackets.

Paper title: {title}

Sections:
{sections}
"""


def _build_sections_block(paper: ParsedPaper) -> tuple[str, list[str]]:
    """Render the sections for the prompt; fall back to full_text if none detected."""
    cap = settings.summary_max_section_chars
    if paper.sections:
        names = list(paper.sections.keys())
        block = "\n\n".join(
            f"[{name}]\n{text[:cap]}" for name, text in paper.sections.items()
        )
        return block, names
    # fallback: no sections parsed -> summarize the whole text as "overview"
    body = (paper.full_text or paper.abstract)[:cap]
    return f"[overview]\n{body}", ["overview"]


def summarize_paper(paper: ParsedPaper) -> dict:
    """Summarize one parsed paper's sections. Returns a JSON-serializable dict."""
    if not paper.available and not paper.abstract:
        return {
            "arxiv_id": paper.arxiv_id, "title": paper.title,
            "source_type": paper.source_type, "section_summaries": [],
            "note": "no text available",
        }

    block, names = _build_sections_block(paper)
    prompt = _PROMPT.format(title=paper.title or paper.arxiv_id, sections=block)
    try:
        result = generate_structured(prompt, PaperSummary)
        summaries = [{"section": s.section, "summary": s.summary} for s in result.summaries]
        note = f"{len(summaries)} section summaries"
    except Exception as exc:  # noqa: BLE001 - one paper failing shouldn't sink the batch
        summaries = []
        note = f"summarize error: {exc}"

    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "source_type": paper.source_type,
        "detected_sections": names,
        "section_summaries": summaries,
        "note": note,
    }


def summarize_papers(hits: list[dict], refresh: bool = False) -> list[dict]:
    """Fetch+parse (Phase 3) then summarize (Phase 4) each retrieval hit, concurrently.

    `hits` are dicts with at least {arxiv_id, title, abstract}. Order is preserved.
    """
    def _one(hit: dict) -> dict:
        cache = Path("fulltext_cache") / hit["arxiv_id"].replace("/", "_") / "summary.json"
        if cache.exists() and not refresh:
            return json.loads(cache.read_text(encoding="utf-8"))
        paper = parse_paper(hit["arxiv_id"], hit.get("title", ""), hit.get("abstract", ""))
        out = summarize_paper(paper)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out

    with ThreadPoolExecutor(max_workers=settings.summary_workers) as pool:
        return list(pool.map(_one, hits))


if __name__ == "__main__":
    import argparse

    # Windows consoles default to cp1252; force UTF-8 so unicode summaries print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Summarize sections of arXiv papers.")
    ap.add_argument("arxiv_ids", nargs="+")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    hits = [{"arxiv_id": a, "title": "", "abstract": ""} for a in args.arxiv_ids]
    for res in summarize_papers(hits, refresh=args.refresh):
        print(f"\n=== {res['arxiv_id']} [{res['source_type']}] {res['note']} ===")
        for s in res["section_summaries"]:
            print(f"  [{s['section']}] {s['summary']}")
