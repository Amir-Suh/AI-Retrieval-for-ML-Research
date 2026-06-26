"""Map messy real-world section headings to canonical buckets.

A paper might title a section "2. Experimental Setup", "Related Works", or
"Results and Discussion"; we collapse these into the fixed set the Phase 4 schema
expects so extraction always reads from a predictable place.
"""

from __future__ import annotations

import re

# Canonical buckets in document order. First matching keyword wins, and the order
# here resolves ambiguity (e.g. check "prior_work" before "methods").
CANONICAL_ORDER = ["introduction", "prior_work", "methods", "experiments", "results", "conclusion"]

_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("introduction", ("introduction", "intro")),
    ("prior_work", ("related work", "related works", "prior work", "previous work",
                    "background", "literature review")),
    ("methods", ("method", "methodology", "approach", "architecture", "proposed",
                 "framework", "model", "preliminaries", "formulation")),
    ("experiments", ("experimental setup", "experiment", "implementation detail",
                     "training detail", "dataset", "setup", "evaluation protocol")),
    ("results", ("result", "evaluation", "ablation", "analysis", "comparison",
                 "performance", "finding")),
    ("conclusion", ("conclusion", "concluding", "discussion", "future work",
                    "limitation", "summary")),
]

# Strip leading section numbering: an Arabic number ("2", "2.1", "2.") optionally
# followed by a separator, OR a roman numeral that MUST be followed by "." or ")"
# (so a word like "introduction"/"various" isn't mistaken for roman numerals i/v/x).
_LEADING_NUMBER = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?|[IVXLCDM]{1,6}[.)]|[ivxlcdm]{1,6}[.)])\s+"
)


def normalize_title(title: str) -> str:
    """Lowercase, drop leading section numbers/roman numerals and punctuation."""
    t = title.strip()
    t = _LEADING_NUMBER.sub("", t)
    return t.lower().strip(" .:-\t")


def canonical_bucket(title: str) -> str | None:
    """Return the canonical bucket for a raw section title, or None if unrecognized."""
    norm = normalize_title(title)
    if not norm:
        return None
    for bucket, keywords in _KEYWORDS:
        if any(kw in norm for kw in keywords):
            return bucket
    return None
