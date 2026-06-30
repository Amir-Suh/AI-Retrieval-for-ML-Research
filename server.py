"""Dev web server for the retrieval pipeline.

Serves the vanilla JS/HTML/CSS UI in `web/` and exposes a JSON search endpoint that
returns every retrieval stage for visualization.

Run:
    .venv/Scripts/python.exe -m uvicorn server:app --reload --port 8000
Then open http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.retrieval.pipeline import search
from src.synthesis.summarize import summarize_papers

app = FastAPI(title="arXiv RAG — Retrieval Visualizer")

WEB_DIR = Path(__file__).resolve().parent / "web"


class SearchRequest(BaseModel):
    query: str
    rerank: bool = True


class PaperRef(BaseModel):
    arxiv_id: str
    title: str = ""
    abstract: str = ""


class SummarizeRequest(BaseModel):
    papers: list[PaperRef]
    refresh: bool = False


@app.post("/api/search")
def api_search(req: SearchRequest) -> dict:
    return search(req.query, use_rerank=req.rerank)


@app.post("/api/summarize")
def api_summarize(req: SummarizeRequest) -> dict:
    hits = [p.model_dump() for p in req.papers]
    return {"papers": summarize_papers(hits, refresh=req.refresh)}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# Mount the static UI at the root. Must be added last so /api/* routes win.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
