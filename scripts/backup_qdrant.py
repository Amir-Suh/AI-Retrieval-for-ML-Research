"""Back up the Qdrant collection to a portable snapshot on the host.

Creates a Qdrant snapshot of the configured collection and downloads it to ./backups/,
so a container/volume loss no longer means re-embedding 50K papers — restore with
scripts/restore_qdrant.py.

Usage:
    python -m scripts.backup_qdrant
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from config import settings

BACKUP_DIR = Path("backups")


def main() -> int:
    coll = settings.qdrant_collection
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    if not client.collection_exists(coll):
        raise SystemExit(f"collection '{coll}' does not exist — nothing to back up.")

    n = client.count(coll, exact=True).count
    print(f"Creating snapshot of '{coll}' ({n:,} points)...")
    snap = client.create_snapshot(collection_name=coll, wait=True)
    name = snap.name

    BACKUP_DIR.mkdir(exist_ok=True)
    out = BACKUP_DIR / name
    url = f"{settings.qdrant_url}/collections/{coll}/snapshots/{name}"
    print(f"Downloading {name} -> {out} ...")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    # Free the in-container copy so the volume doesn't grow unbounded.
    try:
        client.delete_snapshot(collection_name=coll, snapshot_name=name)
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass

    (BACKUP_DIR / "LATEST.txt").write_text(name, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"Done. Saved {out} ({size_mb:.1f} MB). Restore with: python -m scripts.restore_qdrant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
