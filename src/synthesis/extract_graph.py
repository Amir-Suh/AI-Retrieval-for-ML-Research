"""Phase 5a: per-paper entity/relation extraction for the knowledge graph.

For each top-k paper we send its (cached) parsed sections to Gemini once and get back
a small typed graph: the datasets / metrics / methods / models / tasks it names, and
how they relate to the paper. We only ever emit *paper -> entity* (or entity -> entity)
edges here — a single-paper call cannot see the other papers, so cross-paper structure
is left to emerge at merge time (build_graph.py) from shared entities.

Reuses Phase 3's parse cache (no re-fetch) and mirrors summarize.py's concurrency +
per-paper caching so one failing paper never sinks the batch.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import settings
from src.fulltext.parse import ParsedPaper, parse_paper
from src.llm import generate_structured
from src.synthesis.graph_schema import PaperGraph


_PROMPT = """You are building a knowledge graph from ONE machine-learning paper.
Extract the key entities it names and the relationships between them. Be precise and
only include entities that actually appear in the text — do NOT invent or infer items,
and do NOT compare this paper to any other paper.

ENTITY TYPES (use exactly one of these for each entity's `type`):
- dataset : a named dataset or benchmark            (e.g. "ImageNet", "COCO", "GLUE")
- metric  : an evaluation metric NAME only, no number (e.g. "accuracy", "AUC", "APCER", "BLEU")
- method  : a named technique / algorithm / loss / training strategy (e.g. "contrastive learning", "LoRA")
- model   : a named model or architecture            (e.g. "ResNet-50", "BERT", "ViT-B/16")
- task    : the problem being solved                 (e.g. "image classification", "face anti-spoofing")

RELATION TYPES (the subject is the paper unless noted):
- uses-method    : the paper proposes or uses a method/model  -> source "__paper__", target = method/model
- evaluated-on   : the paper evaluates on a dataset           -> source "__paper__", target = dataset
- reports-metric : the paper reports a metric value           -> source "__paper__", target = metric;
                   set `value` to the number (e.g. "2.3%", "81.4") and `dataset` to the
                   dataset it was measured on, whenever the text states them
- based-on       : one method/model builds on another         -> source = method/model, target = method/model

Use the literal string "__paper__" as `source` when the subject is the paper itself.
Prefer canonical short names (e.g. "ImageNet" not "the ImageNet dataset").

Paper title: {title}

Text:
{body}
"""


def _body(paper: ParsedPaper) -> str:
    """Build the extraction context: abstract + each detected section (capped)."""
    cap = settings.graph_max_section_chars
    parts: list[str] = []
    if paper.abstract:
        parts.append("[abstract]\n" + paper.abstract)
    if paper.sections:
        for name, text in paper.sections.items():
            parts.append(f"[{name}]\n{text[:cap]}")
    elif paper.full_text:
        parts.append("[body]\n" + paper.full_text[: cap * 2])
    return "\n\n".join(parts) if parts else (paper.abstract or "")


def extract_paper_graph(paper: ParsedPaper) -> dict:
    """Extract one paper's entities/relations. Returns a JSON-serializable dict."""
    base = {"arxiv_id": paper.arxiv_id, "title": paper.title, "source_type": paper.source_type}
    body = _body(paper)
    if not body.strip():
        return {**base, "entities": [], "relations": [], "note": "no text available"}

    prompt = _PROMPT.format(title=paper.title or paper.arxiv_id, body=body)
    try:
        g = generate_structured(prompt, PaperGraph, temperature=0.0)
        entities = [{"name": e.name, "type": e.type} for e in g.entities if e.name.strip()]
        relations = [
            {"source": r.source, "target": r.target, "type": r.type,
             "value": r.value, "dataset": r.dataset}
            for r in g.relations if r.target.strip()
        ]
        note = f"{len(entities)} entities, {len(relations)} relations"
    except Exception as exc:  # noqa: BLE001 - one paper failing shouldn't sink the batch
        entities, relations, note = [], [], f"extract error: {exc}"

    return {**base, "entities": entities, "relations": relations, "note": note}


def extract_graphs(hits: list[dict], refresh: bool = False) -> list[dict]:
    """Parse (Phase 3) then extract a graph (Phase 5a) for each hit, concurrently.

    `hits` are dicts with at least {arxiv_id, title, abstract}. Order is preserved.
    Each paper's graph is cached to fulltext_cache/{id}/graph.json.
    """
    def _one(hit: dict) -> dict:
        cache = Path("fulltext_cache") / hit["arxiv_id"].replace("/", "_") / "graph.json"
        if cache.exists() and not refresh:
            return json.loads(cache.read_text(encoding="utf-8"))
        paper = parse_paper(hit["arxiv_id"], hit.get("title", ""), hit.get("abstract", ""))
        out = extract_paper_graph(paper)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out

    with ThreadPoolExecutor(max_workers=settings.graph_workers) as pool:
        return list(pool.map(_one, hits))


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Extract a knowledge graph from arXiv papers.")
    ap.add_argument("arxiv_ids", nargs="+")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    hits = [{"arxiv_id": a, "title": "", "abstract": ""} for a in args.arxiv_ids]
    for res in extract_graphs(hits, refresh=args.refresh):
        print(f"\n=== {res['arxiv_id']} [{res['source_type']}] {res['note']} ===")
        for e in res["entities"]:
            print(f"  ({e['type']}) {e['name']}")
        for r in res["relations"]:
            extra = f" = {r['value']}" + (f" on {r['dataset']}" if r["dataset"] else "") if r["value"] else ""
            print(f"  {r['source']} --{r['type']}--> {r['target']}{extra}")
