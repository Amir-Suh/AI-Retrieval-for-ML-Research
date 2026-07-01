"""Phase 5: graph-grounded question answering ("local search").

The user asks a natural-language question about the top-k papers; we answer it using the
cross-paper knowledge graph as the structured retrieval layer and the per-section
summaries as the grounding layer. One Gemini call, forced to cite arXiv ids and to stay
inside the provided context.

Both inputs are already cached from earlier stages:
- build_graph()      -> nodes/edges (Phase 5b)   -> concrete datasets/metrics/values
- summarize_papers() -> section summaries (Phase 4) -> explanation / context
so a question over an already-built graph is a single LLM call.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel

from config import settings
from src.llm import generate_structured
from src.synthesis.build_graph import build_graph
from src.synthesis.summarize import summarize_papers


class GraphAnswer(BaseModel):
    answer: str                 # grounded prose, with inline (arxiv_id) citations
    cited_papers: list[str]     # arxiv_ids actually relied on
    key_entities: list[str]     # main datasets/metrics/methods/models involved


_PROMPT = """You answer questions about a small set of machine-learning papers using ONLY
the GRAPH FACTS and PAPER SUMMARIES provided below. Do not use outside knowledge.

Rules:
- Ground every claim in the provided context; if the answer is not present, say so plainly.
- Prefer GRAPH FACTS for concrete datasets, metrics, values and cross-paper comparison;
  use PAPER SUMMARIES for explanation and context.
- Cite the arXiv id(s) you used inline, e.g. "(2603.02810)".
- Fill `cited_papers` with the arxiv_ids you relied on and `key_entities` with the main
  datasets/metrics/methods/models involved (use the names as written in the facts).

QUESTION:
{question}

=== GRAPH FACTS ===
{graph_block}

=== PAPER SUMMARIES ===
{summaries_block}
"""


def _serialize_graph(graph: dict) -> str:
    nodes = [n["data"] for n in graph.get("nodes", [])]
    paper_ids = {d["id"] for d in nodes if d["type"] == "paper"}
    id2label = {d["id"]: d["label"] for d in nodes}

    def lbl(nid: str) -> str:
        # Papers are cited by arxiv id; entities by their display label.
        return nid if nid in paper_ids else id2label.get(nid, nid)

    lines: list[str] = ["Papers:"]
    for d in nodes:
        if d["type"] == "paper":
            lines.append(f"- {d['id']}: {d.get('full') or d['label']}")

    by_type: dict[str, list[str]] = {}
    for d in nodes:
        if d["type"] != "paper":
            by_type.setdefault(d["type"], []).append(d["label"])
    for t in ("dataset", "metric", "method", "model", "task"):
        if by_type.get(t):
            lines.append(f"{t.capitalize()}s: " + ", ".join(sorted(set(by_type[t]))))

    lines.append("\nRelations:")
    for e in graph.get("edges", []):
        d = e["data"]
        extra = ""
        if d.get("value"):
            extra = f" = {d['value']}" + (f" on {d['dataset']}" if d.get("dataset") else "")
        lines.append(f"- {lbl(d['source'])} {d['type']} {lbl(d['target'])}{extra}")
    return "\n".join(lines)


def _serialize_summaries(summaries: list[dict]) -> str:
    blocks: list[str] = []
    for p in summaries:
        secs = p.get("section_summaries", [])
        if not secs:
            continue
        body = "\n".join(f"  [{s['section']}] {s['summary']}" for s in secs)
        blocks.append(f"{p['arxiv_id']} - {p.get('title') or ''}\n{body}")
    return "\n\n".join(blocks)


def answer_question(question: str, hits: list[dict], refresh: bool = False) -> dict:
    """Answer `question` grounded in the graph + summaries of `hits` (retrieval papers)."""
    graph = build_graph(hits, refresh=refresh)
    summaries = summarize_papers(hits, refresh=refresh)
    prompt = _PROMPT.format(
        question=question,
        graph_block=_serialize_graph(graph),
        summaries_block=_serialize_summaries(summaries),
    )
    try:
        res = generate_structured(prompt, GraphAnswer, model=settings.qa_model, temperature=0.2)
        return {
            "question": question,
            "answer": res.answer,
            "cited_papers": res.cited_papers,
            "key_entities": res.key_entities,
        }
    except Exception as exc:  # noqa: BLE001 - surface the error to the UI, don't crash
        return {"question": question, "answer": f"(error: {exc})",
                "cited_papers": [], "key_entities": []}


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Ask a question grounded in the paper graph.")
    ap.add_argument("question")
    ap.add_argument("arxiv_ids", nargs="+")
    args = ap.parse_args()

    hits = [{"arxiv_id": a, "title": "", "abstract": ""} for a in args.arxiv_ids]
    out = answer_question(args.question, hits)
    print(f"\nQ: {out['question']}\n")
    print(out["answer"])
    print("\ncited:", ", ".join(out["cited_papers"]) or "-")
    print("entities:", ", ".join(out["key_entities"]) or "-")
