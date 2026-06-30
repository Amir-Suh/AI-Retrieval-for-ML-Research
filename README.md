# AI-Retrieval-for-ML-Research

An advanced **Retrieval-Augmented Generation (RAG) pipeline for machine-learning literature**.
Submit a natural-language query, retrieve the most relevant arXiv papers with hybrid search +
cross-encoder reranking, then summarize each paper section-by-section while preserving concrete
technical specifics (dataset sizes, metric values, loss functions) — surfaced in a visual UI.

---

## What it does

```
Query
  │
  ▼  Phase 1–2 · RETRIEVAL  (over ~50K indexed title+abstracts)
  ├─ Vector search   (Gemini embeddings, cosine)      ┐
  ├─ Keyword search  (BM25 sparse vectors)            ┘→ Qdrant native RRF fusion
  └─ Cross-encoder rerank (local) ─────────────────────→ TOP 5 papers
  │
  ▼  Phase 3 · LAZY FULL-TEXT  (only the 5 winners)
  ├─ Fetch LaTeX source (PDF fallback) from arXiv, cached
  └─ Split into canonical sections (intro / prior work / methods / experiments / results / conclusion)
  │
  ▼  Phase 4 · PER-SECTION SUMMARIES
  └─ Gemini summarizes each section, preserving specifics → rendered as blocks in the UI
```

A **vanilla JS/HTML/CSS web UI** visualizes every retrieval stage side-by-side (vector → keyword →
RRF → reranked, with rank-movement badges) and, on demand, the per-section summaries of the top 5.

---

## Tech stack

| Concern | Choice |
|---|---|
| Vector DB | **Qdrant** (self-hosted via Docker), native hybrid dense + sparse + RRF fusion |
| Dense embeddings | `gemini-embedding-001` @ 768-dim (MRL-truncated), L2-normalized |
| Sparse / keyword | FastEmbed `Qdrant/bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local; CPU-friendly) |
| Full-text parsing | `pylatexenc` (LaTeX→text) + PyMuPDF (PDF fallback) |
| Summarization | `gemini-2.5-flash` via native `google-genai` structured output |
| API / UI | FastAPI + Uvicorn serving a vanilla JS/HTML/CSS front-end |

---

## Setup

### Prerequisites
- **Docker Desktop** (for Qdrant)
- **Python 3.10+**
- A **Gemini API key** — https://aistudio.google.com/apikey
- A **Kaggle API token** (`~/.kaggle/kaggle.json`) — for the arXiv metadata snapshot

### Install
```bash
# 1. Python deps (use a project venv)
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt # macOS/Linux

# 2. Configure secrets
cp .env.example .env          # then set GEMINI_API_KEY in .env (plain UTF-8, no BOM)

# 3. Start Qdrant
docker compose up -d --wait
python scripts/check_qdrant.py
```

---

## Usage

### 1. Ingest (one-time)
```bash
# Download the Kaggle arXiv snapshot + filter to ML categories (cs.CV/LG/AI/CL/NE, stat.ML)
python -m src.ingest.download_metadata

# Embed + index a slice into Qdrant (50K most-recent papers; resumable + checkpointed)
python -m src.ingest.build_index --recent --limit 50000
```

### 2. Run the app
```bash
.venv/Scripts/python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
# open http://localhost:8000
```
Type a query, inspect the four retrieval stages, then click **"Summarize top-5 papers"**.

### CLI (no UI)
```bash
python -m src.retrieval.pipeline "self-supervised vision transformers for dense prediction"
python -m src.fulltext.parse 2602.20573          # fetch + section-split one paper
python -m src.synthesis.summarize 2602.20573     # per-section summaries
```

---

## Project structure

```
config.py                  # central settings (models, dims, categories, retrieval constants)
docker-compose.yml         # Qdrant service + persistent volume
server.py                  # FastAPI: /api/search, /api/summarize + static UI
scripts/check_qdrant.py    # connectivity smoke test
src/
  llm.py                   # shared Gemini client + structured generation (retry-aware)
  ingest/                  # Phase 1: download_metadata, embedder, sparse, build_index
  retrieval/               # Phase 2: hybrid_search, rerank, pipeline
  fulltext/                # Phase 3: fetch, parse, sections
  synthesis/               # Phase 4: summarize
web/                       # vanilla JS/HTML/CSS visualizer
```

---

## Implemented (Phases 0–4)

- **Phase 0 — Infra:** Dockerized Qdrant, typed config, connectivity check.
- **Phase 1 — Ingestion:** 581K ML papers filtered from 3.08M arXiv records; hybrid Qdrant
  collection (dense + BM25 sparse); resumable/checkpointed indexing with Gemini-RPM throttling.
- **Phase 2 — Retrieval:** dense + sparse + native RRF fusion + cross-encoder reranking, with a
  stage-by-stage visualizer UI.
- **Phase 3 — Lazy full-text:** LaTeX-source-first fetch (PDF fallback), section splitting into
  canonical buckets, on-disk caching.
- **Phase 4 — Per-section summaries:** concurrent Gemini summarization per section, preserving
  concrete specifics, surfaced as blocks in the UI.

---

## Future implementation

- **Phase 5 — Knowledge graph:** extract entities (datasets, metrics, methods, models) and
  relationships ("evaluated-on", "uses-method", "outperforms") across the top-k papers and render
  an interactive cross-paper graph in the UI.
- **Full-corpus index:** embed all ~581K papers (currently a 50K recent slice); ~3.5 hrs as a
  throttled background job.
- **Retrieval evaluation:** a hand-labeled golden set (query → relevant papers) with recall@5 / MRR
  to quantify the lift from hybrid search + reranking vs. a dense-only baseline.
- **Higher-quality reranking:** swap to `bge-reranker-v2-m3` when running on GPU (one-line config
  change; far slower on CPU).
- **Better PDF parsing:** optional GROBID service for cleaner section boundaries on PDF-only papers.
- **Category / date filtering:** expose Qdrant payload filters (category, year) in the UI.
- **Embedding A/B:** compare Gemini embeddings against SPECTER2 (citation-graph-trained) for
  paper-to-paper retrieval quality.

---

## Notes & gotchas

- Gemini counts **each text in a batched embedding request** as one request against the
  per-minute quota (~3000/min) — ingestion throttles below this.
- `gemini-3-*` model ids do **not** exist on the API (404); `2.5` is current.
- Save `.env` / `kaggle.json` as **plain UTF-8 (no BOM)** — a BOM breaks the Kaggle client.
- `data/` and `fulltext_cache/` are gitignored (large, regenerable).
