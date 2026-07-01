"""Phase 5b: merge per-paper graphs into one cross-paper knowledge graph.

Pipeline:
  1. extract_graphs()  -> one typed graph per paper (Phase 5a, cached)
  2. normalize entity names to stable ids (lowercase, strip punctuation)
  3. one global Gemini pass de-duplicates entities written differently across papers
     (e.g. "imagenet 1k" / "ilsvrc 2012" -> ImageNet)
  4. assemble a Cytoscape.js payload; cross-paper structure emerges from entities that
     end up referenced by >= 2 papers (`shared`)

The merged result is cached keyed on the *set* of paper ids (the merge depends on the
whole set, not any single paper). No networkx / no embeddings — overkill for ~5 papers.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import settings
from src.llm import generate_structured
from src.synthesis.extract_graph import extract_graphs
from src.synthesis.graph_schema import PAPER, CanonResult

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm(name: str) -> str:
    """Stable id for an entity name: lowercase, punctuation -> single spaces."""
    return _NON_ALNUM.sub(" ", name.lower()).strip()


def _short(text: str, n: int = 48) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


_CANON_PROMPT = """You are de-duplicating entities extracted from several machine-learning
papers. Some entries refer to the SAME real-world thing but are written differently, e.g.
"imagenet 1k" / "imagenet" / "ilsvrc 2012" are all ImageNet; "vit b 16" / "vision
transformer b 16" are the same model.

Group ONLY entries of the SAME type that are truly the same entity. Do NOT merge things
that are merely related or are variants (CIFAR-10 and CIFAR-100 are DIFFERENT; ResNet-50
and ResNet-101 are DIFFERENT). When unsure, keep them separate.

For each group of duplicates, set `canonical` to the clearest human-readable name and
`members` to the list of ids (exactly as given on the left below) in that group. Only
return groups that merge 2+ ids; ids you don't mention are kept as-is.

Entities (id : name : type):
{listing}
"""


def _canonicalize(ents: dict[str, dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (mapping: norm_id -> canonical_id, display: canonical_id -> name).

    Falls back to identity (every entity its own canonical) if there's nothing to merge
    or the LLM pass is unavailable/fails.
    """
    mapping = {nid: nid for nid in ents}
    display = {nid: info["display"] for nid, info in ents.items()}
    if len(ents) < 2 or not settings.gemini_api_key:
        return mapping, display

    listing = "\n".join(
        f"{nid} : {info['display']} : {info['type']}" for nid, info in ents.items()
    )
    try:
        res = generate_structured(
            _CANON_PROMPT.format(listing=listing), CanonResult, temperature=0.0
        )
    except Exception:  # noqa: BLE001 - degrade to normalization-only on any failure
        return mapping, display

    for grp in res.groups:
        members = [m for m in grp.members if m in ents]
        if len(members) < 2:
            continue
        canon = _norm(grp.canonical) or members[0]
        display[canon] = grp.canonical.strip() or ents[members[0]]["display"]
        for m in members:
            mapping[m] = canon
    return mapping, display


def build_graph(hits: list[dict], refresh: bool = False) -> dict:
    """Build the cross-paper Cytoscape graph for a set of retrieval hits."""
    ids = [h["arxiv_id"] for h in hits]
    key = hashlib.sha1("|".join(sorted(ids)).encode()).hexdigest()[:16]
    cache = Path("fulltext_cache") / "_graphs" / f"{key}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))

    graphs = extract_graphs(hits, refresh=refresh)
    paper_ids = {g["arxiv_id"] for g in graphs}
    # Prefer the title from the request; fall back to the cached parsed title (set on the
    # per-paper graph) so papers loaded by id alone still render a readable label.
    titles = {h["arxiv_id"]: h.get("title", "") for h in hits}
    for g in graphs:
        if not titles.get(g["arxiv_id"]):
            titles[g["arxiv_id"]] = g.get("title", "")

    # --- collect entities (norm_id -> {display, type, papers}) + raw relations ---
    ents: dict[str, dict] = {}

    def _touch(name: str, etype: str, pid: str | None) -> str:
        nid = _norm(name)
        if not nid:
            return ""
        slot = ents.setdefault(nid, {"display": name.strip(), "type": etype, "papers": set()})
        if pid:
            slot["papers"].add(pid)
        return nid

    rels: list[dict] = []  # {source, target, type, value, dataset}
    for g in graphs:
        pid = g["arxiv_id"]
        for e in g.get("entities", []):
            _touch(e["name"], e.get("type", "method"), pid)
        for r in g.get("relations", []):
            tgt = _touch(r["target"], "method", pid)  # type refined if also in entities
            if not tgt:
                continue
            if r.get("source") == PAPER:
                src = pid
                ents[tgt]["papers"].add(pid)
            else:
                src = _touch(r["source"], "method", pid)
                if not src:
                    continue
            rels.append({"source": src, "target": tgt, "type": r.get("type", ""),
                         "value": r.get("value", ""), "dataset": r.get("dataset", "")})

    # --- global entity resolution ---
    mapping, display = _canonicalize(ents)

    # --- fold entities onto their canonical id ---
    merged: dict[str, dict] = {}
    for nid, info in ents.items():
        cid = mapping.get(nid, nid)
        slot = merged.setdefault(cid, {"type": info["type"], "papers": set()})
        slot["papers"] |= info["papers"]
    node_ids = set(merged) | paper_ids

    # --- nodes ---
    nodes = []
    for pid in ids:
        nodes.append({"data": {
            "id": pid, "label": _short(titles.get(pid) or pid), "type": "paper",
            "full": titles.get(pid, ""), "shared": False, "papers": 1,
        }})
    for cid, info in merged.items():
        pc = len(info["papers"])
        nodes.append({"data": {
            "id": cid, "label": display.get(cid, cid), "type": info["type"],
            "shared": pc >= 2, "papers": pc,
        }})

    # --- edges (remap endpoints to canonical, drop self-loops, dedup) ---
    seen: set[tuple] = set()
    edges = []
    for r in rels:
        s = r["source"] if r["source"] in paper_ids else mapping.get(r["source"], r["source"])
        t = mapping.get(r["target"], r["target"])
        if s == t or s not in node_ids or t not in node_ids:
            continue
        ekey = (s, t, r["type"], r["value"], r["dataset"])
        if ekey in seen:
            continue
        seen.add(ekey)
        edges.append({"data": {
            "id": f"e{len(edges)}", "source": s, "target": t,
            "type": r["type"], "value": r["value"], "dataset": r["dataset"],
        }})

    shared = sum(1 for c, info in merged.items() if len(info["papers"]) >= 2)
    result = {
        "nodes": nodes,
        "edges": edges,
        "stats": {"papers": len(ids), "entities": len(merged),
                  "shared": shared, "edges": len(edges)},
        "notes": [{"arxiv_id": g["arxiv_id"], "note": g["note"]} for g in graphs],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Build a cross-paper knowledge graph.")
    ap.add_argument("arxiv_ids", nargs="+")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    hits = [{"arxiv_id": a, "title": "", "abstract": ""} for a in args.arxiv_ids]
    out = build_graph(hits, refresh=args.refresh)
    print(json.dumps(out["stats"], indent=2))
    shared_nodes = [n["data"] for n in out["nodes"] if n["data"].get("shared")]
    print(f"\nShared entities ({len(shared_nodes)}):")
    for n in sorted(shared_nodes, key=lambda d: -d["papers"]):
        print(f"  [{n['type']}] {n['label']}  — {n['papers']} papers")
