"""Restore the Qdrant collection from the newest snapshot in ./backups/.

Run this after a container/volume loss (e.g. a Docker reset) so you don't have to
re-embed. Assumes Qdrant is running and reachable; recreates the collection from the
snapshot's own config + data.

Usage:
    python -m scripts.restore_qdrant                 # newest snapshot (backups/LATEST.txt)
    python -m scripts.restore_qdrant <snapshot_file> # a specific file in ./backups/
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from config import settings

BACKUP_DIR = Path("backups")


def _resolve_snapshot(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else BACKUP_DIR / p.name
    latest = BACKUP_DIR / "LATEST.txt"
    if latest.exists():
        return BACKUP_DIR / latest.read_text(encoding="utf-8").strip()
    snaps = sorted(BACKUP_DIR.glob("*.snapshot"))
    if not snaps:
        raise SystemExit("No snapshot found in ./backups/ — run scripts.backup_qdrant first.")
    return snaps[-1]


def main() -> int:
    coll = settings.qdrant_collection
    path = _resolve_snapshot(sys.argv[1] if len(sys.argv) > 1 else None)
    if not path.exists():
        raise SystemExit(f"{path} not found.")

    url = f"{settings.qdrant_url}/collections/{coll}/snapshots/upload?priority=snapshot"
    print(f"Uploading {path.name} -> collection '{coll}' ...")
    with open(path, "rb") as f:
        r = requests.post(url, files={"snapshot": (path.name, f)}, timeout=1200)
    r.raise_for_status()

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    print(f"Restored. Collection '{coll}' now holds {client.count(coll, exact=True).count:,} points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
