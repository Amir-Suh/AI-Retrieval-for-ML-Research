"""Phase 1, step 2: embed the filtered abstracts and index them into Qdrant.

Reads `data/arxiv_ml.jsonl`, embeds each paper with both the dense (Gemini) and sparse
(BM25) channels, and upserts into a hybrid Qdrant collection.

Resumable: progress is tracked as a processed-line count in `data/index_checkpoint.txt`.
Re-running continues where it left off. Point IDs are UUID5(arxiv_id), so re-processing a
record is idempotent.

Usage:
    python -m src.ingest.build_index                 # index all of data/arxiv_ml.jsonl
    python -m src.ingest.build_index --limit 50000   # only the first N (validation slice)
    python -m src.ingest.build_index --recreate      # drop + recreate the collection first
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from config import settings
from src.ingest.embedder import embed_texts
from src.ingest.sparse import embed_sparse_documents

INPUT_PATH = Path("data/arxiv_ml.jsonl")
CHECKPOINT_PATH = Path("data/index_checkpoint.txt")
ARXIV_NAMESPACE = uuid.UUID("6f1a7e2c-0000-4000-8000-000000000001")

UPSERT_BATCH = 128  # records per embed+upsert cycle


def get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    exists = client.collection_exists(settings.qdrant_collection)
    if exists and recreate:
        print(f"Dropping existing collection '{settings.qdrant_collection}'")
        client.delete_collection(settings.qdrant_collection)
        exists = False
    if not exists:
        print(f"Creating collection '{settings.qdrant_collection}' "
              f"(dense={settings.embedding_dim}d cosine + bm25 sparse)")
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.embedding_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )


def read_checkpoint(recreate: bool) -> int:
    if recreate:
        CHECKPOINT_PATH.unlink(missing_ok=True)
        return 0
    if CHECKPOINT_PATH.exists():
        return int(CHECKPOINT_PATH.read_text().strip() or "0")
    return 0


def write_checkpoint(n: int) -> None:
    CHECKPOINT_PATH.write_text(str(n))


def _embed_text(rec: dict) -> str:
    return f"{rec['title']}\n\n{rec['abstract']}"


def index_batch(client: QdrantClient, batch: list[dict]) -> None:
    texts = [_embed_text(r) for r in batch]
    dense_vecs = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    sparse_vecs = embed_sparse_documents(texts)

    points = []
    for rec, dense, (s_idx, s_val) in zip(batch, dense_vecs, sparse_vecs):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(ARXIV_NAMESPACE, rec["arxiv_id"])),
                vector={
                    "dense": dense,
                    "bm25": models.SparseVector(indices=s_idx, values=s_val),
                },
                payload={
                    "arxiv_id": rec["arxiv_id"],
                    "title": rec["title"],
                    "abstract": rec["abstract"],
                    "categories": rec["categories"],
                    "authors": rec["authors"],
                    "year": rec.get("year"),
                    "update_date": rec.get("update_date", ""),
                },
            )
        )
    client.upsert(collection_name=settings.qdrant_collection, points=points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="index at most N records (validation slice)")
    parser.add_argument("--recent", action="store_true",
                        help="with --limit, take the most recent N records "
                             "(last N lines) instead of the first N")
    parser.add_argument("--recreate", action="store_true",
                        help="drop the collection + reset checkpoint before indexing")
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise SystemExit(f"{INPUT_PATH} not found. Run download_metadata first.")

    total = sum(1 for _ in INPUT_PATH.open("r", encoding="utf-8"))

    # Window of line indices [window_start, target) to index. The snapshot is ordered
    # oldest-first, so --recent selects the tail (newest papers) for the slice.
    if args.recent and args.limit:
        window_start = max(0, total - args.limit)
        target = total
    else:
        window_start = 0
        target = min(total, args.limit) if args.limit else total

    client = get_client()
    ensure_collection(client, args.recreate)
    ckpt = read_checkpoint(args.recreate)
    start = max(window_start, ckpt)
    if start >= target:
        print(f"Window already fully indexed (start={start:,} >= target={target:,}). Nothing to do.")
        return 0
    print(f"Indexing records [{start:,} .. {target:,}) of {total:,} total")

    processed = start
    batch: list[dict] = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        # islice skips to `start` and stops at `target`, so tqdm counts only the window.
        window = islice(f, start, target)
        for line in tqdm(window, total=target - start, desc="indexing"):
            batch.append(json.loads(line))
            if len(batch) >= UPSERT_BATCH:
                index_batch(client, batch)
                processed += len(batch)
                write_checkpoint(processed)
                batch = []

    if batch:
        index_batch(client, batch)
        processed += len(batch)
        write_checkpoint(processed)

    count = client.count(settings.qdrant_collection, exact=True).count
    print(f"\nDone. Processed up to {processed:,}. Collection now holds {count:,} points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
