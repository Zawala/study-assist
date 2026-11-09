#!/usr/bin/env python3
"""Index over the converted document bundle: diagrams, their captions, and body text.

Reads the two Markdown files the converters produce — `source-text.md`
(text only) and `diagrams/diagrams-index.md` (figures only) — and turns them into two
searchable collections:

    diagrams  one record per figure: label, caption, source document, page, the
              section it sat in, and the PNG on disk
    passages  the body text split at its headings, so a hit carries the heading
              trail it came from and can be quoted back with its neighbours

The Markdown is the whole data source. The bundle's PDFs under `data/pdfs/` are input to
the converters, not to this: nothing here opens one, and this runs fine on a
checkout that has none. What the figure records point at are the PNGs the diagram
index links, which live beside it under `diagrams/images/`. The "Source: DOC01.pdf"
lines are read for the document code and page count only — the filename itself is
never resolved to a file.

Search is BM25 over both collections. No embeddings, no network, no model download:
the corpus is half a million words of tightly-worded standards prose, where the terms
a question uses ("gap analysis", "risk assessment") are the terms the text
uses. Everything is built in memory at startup in about a second.

Run it directly to check the index or query it:

    python3 -m study_assist.knowledge_base --stats
    python3 -m study_assist.knowledge_base "risk assessment"
    python3 -m study_assist.knowledge_base --text "gap analysis"
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from study_assist.paths import GENERATED_DIR

# --- Layout of the bundle ---------------------------------------------------
TEXT_MD = "source-text.md"
DIAGRAMS_MD = "diagrams/diagrams-index.md"

# --- Parsing the two Markdown files -----------------------------------------
# "## Series Guide:  Example Topic" starts a document; the italic line
# under it names the PDF: "*Source: `DOC01.pdf` — 24 pages, 9 diagrams*".
DOC_HEADING_RE = re.compile(r"^## +(?P<title>\S.*?)\s*$")
SOURCE_RE = re.compile(
    r"^\*Source: `(?P<pdf>[^`]+)`(?: +[—-] +(?P<pages>\d+) pages)?"
    r"(?:, +(?P<diagrams>\d+) diagrams)?\*\s*$"
)
# "### Figure 2-1: The Documentation Set" in the diagram index. Artwork the
# extractor found without a caption is headed by its nearest heading instead, and
# marked: "### Function-Centric EA (page 51, uncaptioned)".
FIGURE_RE = re.compile(r"^### +(?P<heading>\S.*?)\s*$")
LABELLED_RE = re.compile(r"^(?P<label>(?:Figure|Table) +[0-9A-Za-z][-–.0-9A-Za-z]*)[:.] +(?P<caption>\S.*)$")
UNCAPTIONED_RE = re.compile(r"\s*\(page \d+, uncaptioned\)$")
# "*Page 25 — 2. The Documentation Set / 2.1. Structure of ...*"
PLACE_RE = re.compile(r"^\*Page +(?P<page>\d+)(?: +[—-] +(?P<section>.*?))?\*\s*$")
IMAGE_RE = re.compile(r"^!\[[^\]]*\]\((?P<src>[^)]+)\)\s*$")
# Any heading below the document level starts a new passage.
SUB_HEADING_RE = re.compile(r"^(?P<hashes>#{3,6}) +(?P<title>\S.*?)\s*$")
# Table-of-contents debris that survived conversion: "|7.4. Outputs. . . . . 77|".
TOC_JUNK_RE = re.compile(r"^\||\. ?\. ?\. ?\.")
# --- Passage sizing ---------------------------------------------------------
MAX_WORDS = 350  # a section longer than this is windowed
OVERLAP_WORDS = 50  # ... with this much carried into the next window
MIN_WORDS = 12  # shorter than this is a stray line, not a passage

# --- Scoring ----------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
CAPTION_WEIGHT = 3  # a diagram is mostly its caption, so count it three times
HEADING_WEIGHT = 2  # ... and a passage's heading trail twice

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in into is it its of on or that the their
    this to was were what when where which who will with within you your do does can could should
    there these those been being about across also any but if not may""".split()
)


def tokenize(text: str) -> list[str]:
    """Words, lowercased, stopped, and crudely singularized.

    The stemming is deliberately shallow — enough to tie "capabilities" to
    "capability" and "diagrams" to "diagram", not enough to conflate unrelated
    terms in a vocabulary this domain-specific.
    """
    out = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("sses"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        if word not in STOPWORDS:
            out.append(word)
    return out


@dataclass(frozen=True)
class Document:
    """One PDF of the bundle."""

    code: str  # "DOC01" — the PDF stem, and the diagram image directory
    title: str
    pdf: str
    pages: int | None
    diagrams: int

    def __str__(self) -> str:
        return f"{self.code}: {self.title}"


@dataclass(frozen=True)
class Diagram:
    """One figure: the PNG plus everything known about where it came from."""

    id: str  # the image stem, unique across the bundle
    label: str  # "Figure 2-1"
    caption: str
    doc_code: str
    doc_title: str
    page: int | None
    section: str
    image: Path

    @property
    def title(self) -> str:
        return f"{self.label}: {self.caption}" if self.label else f"{self.caption} (uncaptioned)"

    def cite(self) -> str:
        """Provenance as a document code, not a filename.

        The bundle's PDFs are the converters' input, never this module's: what it
        reads is the Markdown they produced. Citing `DOC01` rather than
        `DOC01.pdf` names the document without pointing at a file that need not be
        on disk at all.
        """
        where = f"page {self.page}" if self.page else "page unknown"
        return f"{self.doc_title} ({self.doc_code}), {where}"


@dataclass(frozen=True)
class Passage:
    """A run of body text under one heading, or one window of a long section."""

    id: str  # "DOC01#7"
    doc_code: str
    doc_title: str
    heading: str  # breadcrumb: "3. Overview / 3.2. Key Stages"
    text: str
    line: int  # 1-based line in the text Markdown, for tracing back

    def cite(self) -> str:
        return f"{self.doc_title} ({self.doc_code}) — {self.heading or 'front matter'}"


class BM25:
    """Okapi BM25 over a fixed list of token lists."""

    def __init__(self, corpus: list[list[str]]) -> None:
        self.lengths = [len(doc) for doc in corpus]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if corpus else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, doc in enumerate(corpus):
            counts: dict[str, int] = defaultdict(int)
            for term in doc:
                counts[term] += 1
            for term, freq in counts.items():
                self.postings[term].append((index, freq))
        total = len(corpus)
        self.idf = {
            term: math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for index, freq in posting:
                norm = 1 - BM25_B + BM25_B * self.lengths[index] / (self.avg_len or 1)
                scores[index] += idf * freq * (BM25_K1 + 1) / (freq + BM25_K1 * norm)
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit]


@dataclass
class KnowledgeBase:
    """The parsed bundle, with a BM25 index over each collection."""

    root: Path
    documents: dict[str, Document] = field(default_factory=dict)
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)  # code -> line range in TEXT_MD
    _counts: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    diagrams: list[Diagram] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)

    # --- loading ------------------------------------------------------------
    @classmethod
    def load(cls, root: Path) -> KnowledgeBase:
        root = root.resolve()
        text_md = root / TEXT_MD
        diagrams_md = root / DIAGRAMS_MD
        if not text_md.exists():
            raise FileNotFoundError(f"{text_md} not found — run pdf_to_markdown.py first")
        kb = cls(root=root)
        kb._read_passages(text_md)
        if diagrams_md.exists():
            kb._read_diagrams(diagrams_md)
        return kb

    def _read_passages(self, path: Path) -> None:
        """Split the body text into passages, one per heading, windowed if long."""
        doc: Document | None = None
        opened_at = 0
        trail: dict[int, str] = {}
        buffer: list[str] = []
        start = 1

        def flush() -> None:
            nonlocal buffer
            if doc is not None and buffer:
                heading = " / ".join(trail[level] for level in sorted(trail))
                self._add_passage(doc, heading, "\n".join(buffer), start)
            buffer = []

        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            heading = DOC_HEADING_RE.match(line)
            if heading and self._is_document_heading(lines, number):
                flush()
                if doc is not None:
                    self.spans[doc.code] = (opened_at, number - 1)
                doc = self._begin_document(lines, number, heading["title"])
                opened_at = number
                trail = {}
                start = number
                continue
            sub = SUB_HEADING_RE.match(line)
            if sub and not TOC_JUNK_RE.search(sub["title"]):
                flush()
                level = len(sub["hashes"])
                trail = {lvl: text for lvl, text in trail.items() if lvl < level}
                trail[level] = _clean(sub["title"])
                start = number
                continue
            if line.strip():
                buffer.append(line)
        flush()
        if doc is not None:
            self.spans[doc.code] = (opened_at, len(lines))

    @staticmethod
    def _is_document_heading(lines: list[str], number: int) -> bool:
        """A `##` heading is a document only if a `*Source: ...*` line follows it."""
        for peek in lines[number : number + 3]:
            if SOURCE_RE.match(peek):
                return True
        return False

    def _begin_document(self, lines: list[str], number: int, title: str) -> Document:
        source = next(SOURCE_RE.match(peek) for peek in lines[number : number + 3] if SOURCE_RE.match(peek))
        pdf = source["pdf"]
        code = pdf.removesuffix(".pdf")
        doc = Document(
            code=code,
            title=_clean(title),
            pdf=pdf,
            pages=int(source["pages"]) if source["pages"] else None,
            diagrams=int(source["diagrams"]) if source["diagrams"] else 0,
        )
        self.documents.setdefault(code, doc)
        return doc

    def _add_passage(self, doc: Document, heading: str, text: str, line: int) -> None:
        words = text.split()
        if len(words) < MIN_WORDS:
            return
        step = MAX_WORDS - OVERLAP_WORDS
        windows = [words[at : at + MAX_WORDS] for at in range(0, max(len(words) - OVERLAP_WORDS, 1), step)]
        for window in windows:
            self.passages.append(
                Passage(
                    id=f"{doc.code}#{self._counts[doc.code]}",
                    doc_code=doc.code,
                    doc_title=doc.title,
                    heading=heading,
                    text=" ".join(window),
                    line=line,
                )
            )
            self._counts[doc.code] += 1

    def _read_diagrams(self, path: Path) -> None:
        """Read the figure index: heading, place line, and image link per figure."""
        base = path.parent
        code = title = ""
        pending: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            heading = DOC_HEADING_RE.match(line)
            if heading:
                title = _clean(heading["title"])
                continue
            source = SOURCE_RE.match(line)
            if source:
                code = source["pdf"].removesuffix(".pdf")
                continue
            figure = FIGURE_RE.match(line)
            if figure:
                pending = _split_heading(figure["heading"])
                continue
            place = PLACE_RE.match(line)
            if place and pending:
                pending["page"] = place["page"]
                pending["section"] = _clean(place["section"] or "")
                continue
            image = IMAGE_RE.match(line)
            if image and pending:
                src = (base / image["src"]).resolve()
                self.diagrams.append(
                    Diagram(
                        id=src.stem,
                        label=pending["label"],
                        caption=pending["caption"],
                        doc_code=code,
                        doc_title=title,
                        page=int(pending["page"]) if pending.get("page") else None,
                        section=pending.get("section", ""),
                        image=src,
                    )
                )
                pending = {}

    # --- indexes ------------------------------------------------------------
    @cached_property
    def _diagram_index(self) -> BM25:
        return BM25(
            [
                tokenize(" ".join([d.caption] * CAPTION_WEIGHT + [d.label, d.section, d.doc_title]))
                for d in self.diagrams
            ]
        )

    @cached_property
    def _passage_index(self) -> BM25:
        return BM25(
            [tokenize(" ".join([p.heading] * HEADING_WEIGHT + [p.text])) for p in self.passages]
        )

    @cached_property
    def _by_id(self) -> dict[str, Diagram]:
        return {d.id: d for d in self.diagrams}

    @cached_property
    def _by_passage_id(self) -> dict[str, Passage]:
        return {p.id: p for p in self.passages}

    @cached_property
    def _passage_positions(self) -> dict[str, int]:
        return {p.id: index for index, p in enumerate(self.passages)}

    @cached_property
    def _text_lines(self) -> list[str]:
        return (self.root / TEXT_MD).read_text(encoding="utf-8").splitlines()

    # --- lookup -------------------------------------------------------------
    def search_diagrams(self, query: str, limit: int = 8, document: str | None = None) -> list[tuple[Diagram, float]]:
        pool = self._diagram_index.search(query, limit if not document else limit * 8)
        hits = [(self.diagrams[i], score) for i, score in pool]
        if document:
            hits = [hit for hit in hits if _matches_document(hit[0].doc_code, hit[0].doc_title, document)]
        return hits[:limit]

    def search_passages(self, query: str, limit: int = 6, document: str | None = None) -> list[tuple[Passage, float]]:
        pool = self._passage_index.search(query, limit if not document else limit * 12)
        hits = [(self.passages[i], score) for i, score in pool]
        if document:
            hits = [hit for hit in hits if _matches_document(hit[0].doc_code, hit[0].doc_title, document)]
        return hits[:limit]

    def diagram(self, diagram_id: str) -> Diagram | None:
        """Look a figure up by image id, or by `<code> Figure N-N` / plain caption."""
        exact = self._by_id.get(diagram_id)
        if exact:
            return exact
        wanted = tokenize(diagram_id)
        best = [d for d in self.diagrams if set(tokenize(f"{d.doc_code} {d.label} {d.caption}")) >= set(wanted)]
        if len(best) == 1:
            return best[0]
        hits = self.search_diagrams(diagram_id, limit=1)
        return hits[0][0] if hits and best else (best[0] if best else None)

    def passage(self, passage_id: str) -> Passage | None:
        return self._by_passage_id.get(passage_id)

    def around(self, passage: Passage, span: int) -> list[Passage]:
        """The passage plus up to `span` neighbours either side, same document only."""
        at = self._passage_positions[passage.id]
        window = self.passages[max(0, at - span) : at + span + 1]
        return [p for p in window if p.doc_code == passage.doc_code]

    def caption_context(self, diagram: Diagram, before: int = 12, after: int = 6) -> str:
        """The body text around where this figure sits in the prose.

        The figure index and the text conversion are separate passes over the same
        PDFs, so there is no shared anchor to join on — only the caption itself,
        which the converter emits italicized: `_Figure 2-1:  The Documentation
        Set_`. Failing that, fall back to the bare figure label and then to the
        section heading, each searched only inside this document's own line range so
        that a label as common as "Figure 2-1" cannot match the wrong guide.
        """
        first, last = self.spans.get(diagram.doc_code, (1, len(self._text_lines)))
        window = self._text_lines[first - 1 : last]
        candidates = [
            _squash(f"{diagram.label}: {diagram.caption}").lower() if diagram.label else "",
            _squash(diagram.caption).lower(),
            _squash(diagram.label).lower(),
            _squash(diagram.section.split(" / ")[0]).lower(),
        ]
        for needle in filter(None, candidates):
            for offset, line in enumerate(window):
                squashed = _squash(line).lower()
                if needle in squashed and (squashed.startswith(("_", "#")) or len(squashed) < 2 * len(needle)):
                    at = first - 1 + offset
                    quoted = self._text_lines[max(0, at - before) : at + after + 1]
                    return "\n".join(line for line in quoted if line.strip())
        return ""

    def warm(self) -> None:
        """Build every lazy index up front.

        The knowledge base is read-only once loaded and so is safe to share across
        concurrent requests — but only after this has run. The indexes are
        `cached_property`, and two requests racing to build one would each pay for
        it, `_text_lines` most expensively of all at 3.5MB of file.
        """
        for name in (
            "_diagram_index",
            "_passage_index",
            "_by_id",
            "_by_passage_id",
            "_passage_positions",
            "_text_lines",
        ):
            getattr(self, name)

    def stats(self) -> str:
        return (
            f"{len(self.documents)} documents, {len(self.diagrams)} diagrams, "
            f"{len(self.passages)} passages, {len(self._passage_index.postings)} terms"
        )


def _split_heading(heading: str) -> dict[str, str]:
    """A figure heading into label and caption; uncaptioned artwork gets no label."""
    labelled = LABELLED_RE.match(_clean(heading))
    if labelled:
        return {"label": labelled["label"], "caption": _clean(labelled["caption"])}
    return {"label": "", "caption": _clean(UNCAPTIONED_RE.sub("", heading))}


def _clean(text: str) -> str:
    """Strip Markdown emphasis and squeeze whitespace out of a heading or caption."""
    return _squash(re.sub(r"[*_`]+", "", text)).strip(" .")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _matches_document(code: str, title: str, wanted: str) -> bool:
    """Match a filter against a document code or title, leniently.

    A caller who has read a "Source: DOC01.pdf" line in the Markdown may well pass
    that whole filename, so the suffix is tolerated on input even though nothing
    emits it any more.
    """
    wanted = wanted.lower().removesuffix(".pdf")
    return wanted in code.lower() or wanted in title.lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*", help="search the diagram captions")
    parser.add_argument("-r", "--root", type=Path, default=GENERATED_DIR, help="bundle directory")
    parser.add_argument("-t", "--text", action="store_true", help="search body text instead of diagrams")
    parser.add_argument("-n", "--limit", type=int, default=8)
    parser.add_argument("--stats", action="store_true", help="just report what was indexed")
    args = parser.parse_args()

    kb = KnowledgeBase.load(args.root)
    if args.stats or not args.query:
        print(kb.stats())
        missing = [d.id for d in kb.diagrams if not d.image.exists()]
        if missing:
            print(f"warning: {len(missing)} diagrams have no image on disk, e.g. {missing[0]}", file=sys.stderr)
        return 0

    query = " ".join(args.query)
    if args.text:
        for passage, score in kb.search_passages(query, args.limit):
            print(f"[{score:5.1f}] {passage.id}  {passage.cite()}")
            print(f"        {passage.text[:160]}...\n")
    else:
        for diagram, score in kb.search_diagrams(query, args.limit):
            print(f"[{score:5.1f}] {diagram.title}")
            print(f"        {diagram.cite()}  id={diagram.id}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
