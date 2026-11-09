#!/usr/bin/env python3
"""Convert a bundle of PDFs into a single Markdown file.

Text only — images are never extracted or linked. Uses pymupdf4llm when
available (keeps headings, lists, and tables) and falls back to poppler's
`pdftotext -layout` otherwise.

Usage:
    python3 -m study_assist.pdf_to_markdown                  # data/pdfs/*.pdf -> data/generated/source-text.md
    python3 -m study_assist.pdf_to_markdown -o text.md -j 8  # pick output file / worker count
    python3 -m study_assist.pdf_to_markdown document.pdf     # convert specific files
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from study_assist.paths import PDF_DIR, TEXT_FILE

BUNDLE_TITLE = "Source Documents"

# --- Patterns for page furniture --------------------------------------------
# Standalone page numbers: arabic, roman numerals, or "3-12" style.
PAGE_NUM_RE = re.compile(r"^\s*(?:\d{1,4}(?:[-–]\d{1,4})?|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)
# A TOC row that pymupdf4llm renders as a one-cell table: |Heading. . . . 12|
TOC_ROW_RE = re.compile(r"^\|(.+?)[\s.]*?\s\.\s*(?:\.\s*)+(\S+)\|$")
# Lines never worth keeping, however often they appear.
WATERMARK_RE = re.compile(r"^(?:evaluation copy|©.*all rights reserved.*)$", re.IGNORECASE)
# Image embeds pymupdf4llm may emit; this build is text-only.
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
# The page watermark, which the extractor often welds onto adjacent text.
EVAL_COPY_RE = re.compile(r"\s*Evaluation Copy\s*")
# Split before a numbered section label, e.g. "3.3. Developing ...".
SECTION_SPLIT_RE = re.compile(r"(?=\b\d+(?:\.\d+)*\.\s+[A-Z])")


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_]+", "-", text)


def doc_title(path: Path) -> str:
    """Best-effort human title: PDF metadata, else filename."""
    stem = path.stem
    try:
        import pymupdf  # noqa: PLC0415

        with pymupdf.open(path) as doc:
            if meta_title := (doc.metadata or {}).get("title", "").strip():
                return meta_title
    except Exception:
        pass
    return stem


def sort_key(path: Path) -> str:
    """Order documents by their filename."""
    return path.stem.upper()


# --- Cleanup ----------------------------------------------------------------
def heading_text(line: str) -> str | None:
    """The plain text of a markdown heading, else None."""
    m = HEADING_RE.match(line)
    return m.group(2).replace("*", "").strip() if m else None


def is_running_head(text: str, headings: set[str]) -> bool:
    """True if the line is nothing but heading text — one or more headers run
    together by the extractor, e.g. "4. Partitioning 4.2. Applying ..."."""
    if text in headings:
        return True
    parts = [p.strip() for p in SECTION_SPLIT_RE.split(text) if p.strip()]
    return len(parts) > 1 and all(p in headings for p in parts)


def repeated_boilerplate(lines: list[str], min_hits: int = 5) -> set[str]:
    """Short lines recurring throughout a document are headers/footers."""
    counts = Counter(
        s
        for line in lines
        if (s := line.strip())
        and len(s) <= 90
        and not s.startswith(("#", "-", "*", "|", ">", "!["))
    )
    return {text for text, n in counts.items() if n >= min_hits}


def tidy(md: str) -> str:
    """Strip page furniture and normalize the structure the extractor emits."""
    md = IMAGE_RE.sub("", md)
    # The watermark lands anywhere, often welded to adjacent running heads.
    md = EVAL_COPY_RE.sub(" ", md)
    # `# **Heading**` -> `# Heading`; bold adds nothing on top of a heading.
    md = re.sub(r"^(#{1,6})\s*\*\*(.+?)\*\*\s*$", r"\1 \2", md, flags=re.MULTILINE)
    # Large-font bullets get mistaken for headings; put them back in a list.
    md = re.sub(r"^#{1,6}\s*[•·]\s*(.+)$", r"- \1", md, flags=re.MULTILINE)

    lines = md.split("\n")
    drop = repeated_boilerplate(lines)
    # A bare line repeating a heading's text is a running head, not content.
    headings = {h for line in lines if (h := heading_text(line))}

    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in drop or WATERMARK_RE.match(stripped) or PAGE_NUM_RE.match(stripped):
            continue
        if stripped and not stripped.startswith("#") and is_running_head(stripped, headings):
            continue
        # TOC table rows read better as a list, page number preserved.
        if m := TOC_ROW_RE.match(stripped):
            kept.append(f"- {m.group(1).strip().rstrip('.')} — {m.group(2)}")
            continue
        if re.fullmatch(r"\|-+\|", stripped):  # separator of those 1-cell tables
            continue
        kept.append(line)
    md = "\n".join(kept)

    # The extractor's page separators would become stray rules in one long file.
    md = re.sub(r"^-{3,}\s*$", "", md, flags=re.MULTILINE)
    # De-hyphenate words broken across lines.
    md = re.sub(r"(\w)-\n(\w)", r"\1\2", md)
    md = re.sub(r"[ \t]+$", "", md, flags=re.MULTILINE)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def demote(md: str, levels: int = 1) -> str:
    """Push every heading down so document headings nest under their section."""

    def shift(m: re.Match[str]) -> str:
        return f"{'#' * min(len(m.group(1)) + levels, 6)} {m.group(2)}"

    return HEADING_RE.sub(shift, md)


# --- Extraction -------------------------------------------------------------
def extract_pymupdf(path: Path) -> tuple[str, int]:
    import pymupdf  # noqa: PLC0415
    import pymupdf4llm  # noqa: PLC0415

    with pymupdf.open(path) as doc:
        pages = doc.page_count
    # write_images/embed_images left off: text-only output.
    return pymupdf4llm.to_markdown(str(path), show_progress=False), pages


def extract_pdftotext(path: Path) -> tuple[str, int]:
    proc = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.replace("\f", "\n\n"), proc.stdout.count("\f")


def convert_one(path: Path, engine: str) -> dict:
    """Worker: returns the finished section for one PDF."""
    raw, pages = (extract_pymupdf if engine == "pymupdf" else extract_pdftotext)(path)
    title = doc_title(path)
    body = demote(tidy(raw))
    return {
        "source": path.name,
        "title": title,
        "anchor": slugify(title),
        "pages": pages,
        "words": len(body.split()),
        "markdown": f"## {title}\n\n*Source: `{path.name}` — {pages} pages*\n\n{body}\n",
    }


def pick_engine(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import pymupdf4llm  # noqa: F401,PLC0415

        return "pymupdf"
    except ImportError:
        if shutil.which("pdftotext"):
            print(
                "pymupdf4llm not installed; falling back to pdftotext.\n"
                "  For much better Markdown: pip install pymupdf4llm",
                file=sys.stderr,
            )
            return "pdftotext"
    sys.exit("No converter available. Install pymupdf4llm or poppler-utils.")


def assemble(sections: list[dict], engine: str) -> str:
    total_pages = sum(s["pages"] for s in sections)
    out = [
        "---",
        f'title: "{BUNDLE_TITLE}"',
        f"converted: {time.strftime('%Y-%m-%d')}",
        f"converter: {engine} (text only, no images)",
        f"documents: {len(sections)}",
        f"pages: {total_pages}",
        "---",
        "",
        f"# {BUNDLE_TITLE}",
        "",
        f"{len(sections)} documents, {total_pages} pages, "
        f"{sum(s['words'] for s in sections):,} words. Converted from PDF; text only.",
        "",
        "## Contents",
        "",
    ]
    out += [f"{i}. [{s['title']}](#{s['anchor']}) — `{s['source']}`" for i, s in enumerate(sections, 1)]
    out.append("")
    for s in sections:
        out += ["---", "", s["markdown"], ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pdfs", nargs="*", type=Path, help="PDFs to convert (default: all in --input)")
    ap.add_argument("-i", "--input", type=Path, default=PDF_DIR, help="directory of PDFs")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output file (default: data/generated/source-text.md)",
    )
    ap.add_argument("-j", "--jobs", type=int, default=4, help="parallel workers (default: 4)")
    ap.add_argument("--engine", choices=["auto", "pymupdf", "pdftotext"], default="auto")
    args = ap.parse_args()

    pdfs = sorted(args.pdfs or args.input.glob("*.pdf"), key=sort_key)
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 1

    out_file = args.output or TEXT_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)
    engine = pick_engine(args.engine)
    print(f"Converting {len(pdfs)} PDF(s) with {engine} -> {out_file}\n")

    results: dict[Path, dict] = {}
    failures = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(convert_one, p, engine): p for p in pdfs}
        for done in concurrent.futures.as_completed(futures):
            pdf = futures[done]
            try:
                results[pdf] = done.result()
                print(f"  {pdf.name}: {results[pdf]['pages']} pages, {results[pdf]['words']:,} words")
            except Exception as exc:  # keep going; report at the end
                failures += 1
                print(f"  FAILED {pdf.name}: {exc}", file=sys.stderr)

    sections = [results[p] for p in pdfs if p in results]  # restore reading order
    if not sections:
        print("Nothing converted.", file=sys.stderr)
        return 1

    out_file.write_text(assemble(sections, engine), "utf-8")
    print(
        f"\nDone. {len(sections)} of {len(pdfs)} converted"
        f"{f', {failures} failed' if failures else ''}."
        f"\nWrote {out_file} ({out_file.stat().st_size / 1_048_576:.1f} MB)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
