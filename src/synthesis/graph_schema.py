"""Pydantic schemas for the Phase 5 knowledge graph.

Two LLM-facing schemas:
- `PaperGraph` — per-paper extraction (entities + paper-local relations).
- `CanonResult` — the global de-duplication pass that merges entities written
  differently across papers (e.g. "ImageNet-1K" / "ILSVRC 2012" -> ImageNet).

Metric *values* are NOT nodes: a metric node is the metric NAME, and the reported
number rides on the `reports-metric` edge (`value` + the `dataset` it was measured on),
so two papers that report the same metric link to one shared node and their values are
comparable on the edges.
"""

from __future__ import annotations

from pydantic import BaseModel

# Allowed vocabularies (kept here so the prompt and any validation share one source).
ENTITY_TYPES = ("dataset", "metric", "method", "model", "task")
RELATION_TYPES = ("uses-method", "evaluated-on", "reports-metric", "based-on")

# Sentinel `source` meaning "the paper itself" (resolved to the arxiv_id at merge time).
PAPER = "__paper__"


class Entity(BaseModel):
    name: str   # as written in the paper, e.g. "ImageNet-1K"
    type: str   # one of ENTITY_TYPES


class Relation(BaseModel):
    source: str          # PAPER ("__paper__") or an entity name
    target: str          # an entity name
    type: str            # one of RELATION_TYPES
    value: str = ""      # for reports-metric: the number, e.g. "2.3%"
    dataset: str = ""    # for reports-metric: the dataset the value was measured on


class PaperGraph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


# --- global entity resolution ---

class CanonGroup(BaseModel):
    canonical: str        # clearest human-readable name for the group
    members: list[str]    # the input ids that refer to this same entity


class CanonResult(BaseModel):
    groups: list[CanonGroup]
