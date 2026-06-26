"""Parse a fetched paper into canonical sections.

LaTeX path (preferred): find the main .tex, inline \input/\include, strip comments,
split on \section/\subsection, then clean each chunk to readable text with pylatexenc.

PDF path (fallback): extract text with PyMuPDF and split on heuristically detected
section headings.

Both paths always populate `full_text` so downstream extraction degrades gracefully
when section detection is imperfect.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.fulltext.fetch import FetchResult, fetch_paper
from src.fulltext.sections import CANONICAL_ORDER, canonical_bucket


@dataclass
class ParsedPaper:
    arxiv_id: str
    title: str
    abstract: str
    source_type: str                       # "latex" | "pdf"
    sections: dict[str, str] = field(default_factory=dict)  # canonical -> text
    full_text: str = ""
    available: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def section_summary(self) -> dict[str, int]:
        """Char count per detected canonical section (for quick inspection)."""
        return {k: len(v) for k, v in self.sections.items()}


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

_COMMENT = re.compile(r"(?<!\\)%.*")
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_SECTION = re.compile(r"\\(section|subsection)\*?\s*\{")


def _strip_comments(tex: str) -> str:
    return "\n".join(_COMMENT.sub("", line) for line in tex.splitlines())


def _find_main_tex(tex_files: list[Path]) -> Path:
    """The main file is the one with \\begin{document}; fall back to the largest."""
    candidates = []
    for p in tex_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\\begin{document}" in txt:
            candidates.append((p, txt.count("\\section"), p.stat().st_size))
    if candidates:
        # prefer the one with the most \section commands, then largest
        candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
        return candidates[0][0]
    return max(tex_files, key=lambda p: p.stat().st_size)


def _inline_inputs(text: str, src_dir: Path, depth: int = 0) -> str:
    """Recursively replace \\input{f}/\\include{f} with the referenced file's contents."""
    if depth > 8:
        return text

    def repl(match: re.Match) -> str:
        name = match.group(1).strip()
        for cand in (name, name + ".tex", name + ".ltx"):
            f = src_dir / Path(cand).name
            if f.exists():
                try:
                    inner = _strip_comments(f.read_text(encoding="utf-8", errors="ignore"))
                    return _inline_inputs(inner, src_dir, depth + 1)
                except OSError:
                    return ""
        return ""

    return _INPUT.sub(repl, text)


def _braced_title(text: str, start: int) -> tuple[str, int]:
    """Read a balanced {...} group starting at `start` (the opening brace index)."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return "", start + 1


def _latex_to_text(latex: str) -> str:
    try:
        from pylatexenc.latex2text import LatexNodes2Text

        out = LatexNodes2Text().latex_to_text(latex)
    except Exception:  # noqa: BLE001 - pylatexenc can choke on exotic macros
        out = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", latex)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _parse_latex(fr: FetchResult, title: str, abstract: str) -> ParsedPaper:
    main = _find_main_tex(fr.tex_files)
    body = _strip_comments(main.read_text(encoding="utf-8", errors="ignore"))
    body = _inline_inputs(body, main.parent)

    # restrict to the document body if present
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", body, re.DOTALL)
    if m:
        body = m.group(1)

    # Walk \section/\subsection headers in order, slicing the text between them.
    headers: list[tuple[int, int, str]] = []  # (header_start, body_start, title)
    for m in _SECTION.finditer(body):
        brace = m.end() - 1
        sec_title, after = _braced_title(body, brace)
        headers.append((m.start(), after, sec_title))

    sections: dict[str, str] = {}
    if headers:
        for idx, (h_start, b_start, sec_title) in enumerate(headers):
            b_end = headers[idx + 1][0] if idx + 1 < len(headers) else len(body)
            bucket = canonical_bucket(sec_title)
            if not bucket:
                continue
            chunk = _latex_to_text(body[b_start:b_end])
            if chunk:
                sections[bucket] = (sections.get(bucket, "") + "\n\n" + chunk).strip()

    full_text = _latex_to_text(body)
    ordered = {k: sections[k] for k in CANONICAL_ORDER if k in sections}
    return ParsedPaper(
        arxiv_id=fr.arxiv_id, title=title, abstract=abstract, source_type="latex",
        sections=ordered, full_text=full_text, available=True,
        note=f"{fr.note}; {len(ordered)} sections from {main.name}",
    )


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

_PDF_HEADING = re.compile(
    r"^\s*(?:\d+\.?\d*\.?\s+)?"
    r"(introduction|related works?|prior work|background|method(?:s|ology)?|approach|"
    r"experiments?|experimental setup|implementation details|results?|evaluation|"
    r"ablation|discussion|conclusions?|future work|limitations?)\b.*$",
    re.IGNORECASE,
)


def _parse_pdf(fr: FetchResult, title: str, abstract: str) -> ParsedPaper:
    import fitz  # PyMuPDF

    doc = fitz.open(fr.pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # Heuristic: detect heading lines, slice text between them.
    lines = full_text.splitlines()
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if 0 < len(s) <= 60 and _PDF_HEADING.match(s):
            marks.append((i, s))

    sections: dict[str, str] = {}
    for j, (line_no, heading) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        bucket = canonical_bucket(heading)
        if not bucket:
            continue
        chunk = "\n".join(lines[line_no + 1 : end]).strip()
        if chunk:
            sections[bucket] = (sections.get(bucket, "") + "\n\n" + chunk).strip()

    ordered = {k: sections[k] for k in CANONICAL_ORDER if k in sections}
    return ParsedPaper(
        arxiv_id=fr.arxiv_id, title=title, abstract=abstract, source_type="pdf",
        sections=ordered, full_text=full_text.strip(), available=bool(full_text.strip()),
        note=f"{fr.note}; {len(ordered)} sections (pdf heuristic)",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_paper(arxiv_id: str, title: str = "", abstract: str = "",
                refresh: bool = False) -> ParsedPaper:
    """Fetch (cached) + parse a paper into canonical sections."""
    cache_dir = Path("fulltext_cache") / arxiv_id.replace("/", "_")
    parsed_json = cache_dir / "parsed.json"
    if parsed_json.exists() and not refresh:
        d = json.loads(parsed_json.read_text(encoding="utf-8"))
        return ParsedPaper(**d)

    fr = fetch_paper(arxiv_id, refresh=refresh)
    try:
        paper = _parse_latex(fr, title, abstract) if fr.source_type == "latex" \
            else _parse_pdf(fr, title, abstract)
    except Exception as exc:  # noqa: BLE001 - never let one bad paper crash the batch
        paper = ParsedPaper(arxiv_id, title, abstract, fr.source_type,
                            available=False, note=f"parse error: {exc}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed_json.write_text(json.dumps(paper.to_dict(), ensure_ascii=False), encoding="utf-8")
    return paper


def parse_papers(hits: list[dict], refresh: bool = False) -> list[ParsedPaper]:
    """Parse a list of retrieval hits ({arxiv_id, title, abstract, ...})."""
    return [
        parse_paper(h["arxiv_id"], h.get("title", ""), h.get("abstract", ""), refresh)
        for h in hits
    ]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fetch + parse arXiv papers into sections.")
    ap.add_argument("arxiv_ids", nargs="+")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    for aid in args.arxiv_ids:
        p = parse_paper(aid, refresh=args.refresh)
        print(f"\n=== {aid} [{p.source_type}] available={p.available} ===")
        print("note:", p.note)
        print("sections:", p.section_summary())
        for name, text in p.sections.items():
            first = " ".join(text.split())[:120]
            print(f"  [{name}] {first}...")
