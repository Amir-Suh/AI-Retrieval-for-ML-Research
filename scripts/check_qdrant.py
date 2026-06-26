"""Phase 0 smoke test: confirm Qdrant is reachable from Python.

Run after `docker compose up -d --wait`:
    python scripts/check_qdrant.py
"""

import sys
from pathlib import Path

# Make the project root importable when run as `python scripts/check_qdrant.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from config import settings


def main() -> int:
    print(f"Connecting to Qdrant at {settings.qdrant_url} ...")
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collections = client.get_collections().collections
    print("OK - Qdrant is reachable.")
    print(f"Existing collections: {[c.name for c in collections] or '(none yet)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
