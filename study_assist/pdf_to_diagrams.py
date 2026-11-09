#!/usr/bin/env python3
"""Extract only the diagrams (and their captions) from a bundle of PDFs.

Companion to pdf_to_markdown.py, which is text-only. This one is the inverse:
it pulls every figure out as a PNG, pairs it with its "Figure 3-1: ..." caption
and the running head of the page it sat on, and writes an index Markdown file
that links them in reading order. Body text is never emitted.

Figures in this bundle are embedded raster images; captions are ordinary text
blocks just below them. Artwork with no "Figure N-N" caption (full-page appendix
visuals, side-by-side models) is labelled from the nearest heading instead, or
skipped entirely with --captioned-only. Neighbouring images are merged into one
figure, and inline icons, repeated logos, and cover art are dropped.

Requires pymupdf (`pip install pymupdf`).

Usage:
    python3 -m study_assist.pdf_to_diagrams                  # data/pdfs/*.pdf -> data/generated/diagrams/
    python3 -m study_assist.pdf_to_diagrams -i pdfs -o out   # pick input dir / output dir
    python3 -m study_assist.pdf_to_diagrams --mode render --dpi 300  # rasterize the page region
    python3 -m study_assist.pdf_to_diagrams --captioned-only # only "Figure N-N" figures
    python3 -m study_assist.pdf_to_diagrams document.pdf     # just these PDFs
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import time
from collections import Counter
from pathlib import Path

from study_assist.paths import DIAGRAM_DIR, PDF_DIR

try:
    import pymupdf
except ImportError:  # pragma: no cover - environment problem, not logic
    sys.exit("pymupdf is required: pip install pymupdf")

BUNDLE_TITLE = "Diagrams"

# A caption line: "Figure 3-1: Overview" / "Figure 1-1. The Process". The
# separator is required, which is what tells a caption apart from a cross
# reference in running text ("Figure 1-3 illustrates how ...").
CAPTION_RE = re.compile(
    r"^(Figure|Table)\s+([0-9A-Za-z]+(?:[-–.][0-9]+)+)\s*[:.]\s*(\S.*?)\s*$", re.IGNORECASE
)
# Page furniture that must never be mistaken for a caption or a running head.
WATERMARK_RE = re.compile(r"evaluation copy|all rights reserved", re.IGNORECASE)
# Standalone page numbers: arabic, roman numerals, or "3-12" style.
PAGE_NUM_RE = re.compile(r"^\s*(?:\d{1,4}(?:[-–]\d{1,4})?|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)

# --- Tuning knobs -----------------------------------------------------------
MIN_W, MIN_H = 110, 75  # points; below this an image is an icon, not a figure
MAX_LOGO_PAGES = 2  # an image reused on more pages than this is a logo
CLUSTER_PAD = 14  # points; images this close belong to one figure
CAPTION_BELOW = 110  # points to look under a figure for its caption
CAPTION_ABOVE = 45  # ... and above, for the rarer caption-on-top layout
LABEL_GAP = 60  # ... and either way, for a plain heading standing in for one
LABEL_MAX = 80  # characters; longer than this it is a paragraph, not a label
LABEL_BELOW_BIAS = 8  # points; a heading over the artwork wins a near-tie
# A label reads as a heading: it starts big and does not trail off mid-sentence.
LABEL_OK_RE = re.compile(r"^[A-Z0-9(\u201c\u2018].*[^.,;:]$")


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_]+", "-", text)[:70].strip("-")


def doc_title(doc: pymupdf.Document, path: Path) -> str:
    """Best-effort human title: PDF metadata, else filename."""
    stem = path.stem
    if meta_title := (doc.metadata or {}).get("title", "").strip():
        return meta_title
    return stem


def sort_key(path: Path) -> str:
    """Order documents by their filename."""
    return path.stem.upper()


# --- Figure detection -------------------------------------------------------
def cluster(rects: list[pymupdf.Rect], pad: float = CLUSTER_PAD) -> list[pymupdf.Rect]:
    """Merge images that touch or nearly touch: one diagram, several XObjects."""
    boxes = sorted(rects, key=lambda r: (r.y0, r.x0))
    merged: list[pymupdf.Rect] = []
    for r in boxes:
        grown = pymupdf.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)
        for i, m in enumerate(merged):
            if m.intersects(grown):
                merged[i] = m | r
                break
        else:
            merged.append(pymupdf.Rect(r))
    # One pass rarely settles chains of three; repeat until it does.
    return merged if len(merged) == len(boxes) else cluster(merged, pad)


def logo_xrefs(doc: pymupdf.Document) -> set[int]:
    """Images that recur across pages are branding, not diagrams."""
    seen = Counter(im[0] for page in doc for im in page.get_images(full=True))
    return {xref for xref, n in seen.items() if n > MAX_LOGO_PAGES}


def figure_rects(page: pymupdf.Page, skip: set[int]) -> list[pymupdf.Rect]:
    """Bounding boxes of the diagrams on this page, largest artwork first."""
    rects: list[pymupdf.Rect] = []
    for im in page.get_images(full=True):
        if im[0] in skip:
            continue
        for r in page.get_image_rects(im[0]):
            if r.width >= MIN_W and r.height >= MIN_H and not r.is_empty:
                rects.append(r)
    return [r for r in cluster(rects) if r.width >= MIN_W and r.height >= MIN_H]


def caption_for(page: pymupdf.Page, box: pymupdf.Rect) -> tuple[str, str, str] | None:
    """The nearest caption block under (else over) a figure: (kind, number, text)."""
    best: tuple[float, tuple[str, str, str]] | None = None
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        first = text.strip().split("\n", 1)[0].strip()
        m = CAPTION_RE.match(first)
        if not m or WATERMARK_RE.search(first):
            continue
        if min(x1, box.x1) - max(x0, box.x0) <= 0:  # must sit under the artwork
            continue
        below, above = y0 - box.y1, box.y0 - y1
        gap = below if 0 <= below <= CAPTION_BELOW else (above if 0 <= above <= CAPTION_ABOVE else None)
        if gap is not None and (best is None or gap < best[0]):
            best = (gap, (m.group(1).title(), m.group(2), m.group(3)))
    return best[1] if best else None


def nearby_label(page: pymupdf.Page, box: pymupdf.Rect) -> str | None:
    """Stand-in for a missing caption: the short heading nearest the artwork."""
    best: tuple[float, str] | None = None
    body = pymupdf.Rect(0, 34, page.rect.width, page.rect.height - 60)
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        line = " ".join(text.split())
        if not line or len(line) > LABEL_MAX or WATERMARK_RE.search(line):
            continue
        if not LABEL_OK_RE.match(line):  # a sentence fragment, not a heading
            continue
        if not body.contains(pymupdf.Rect(x0, y0, x1, y1)) or PAGE_NUM_RE.match(line):
            continue
        if min(x1, box.x1) - max(x0, box.x0) <= 0:
            continue
        above, below = box.y0 - y1, y0 - box.y1
        gap = (
            above
            if 0 <= above <= LABEL_GAP
            else (below + LABEL_BELOW_BIAS if 0 <= below <= LABEL_GAP else None)
        )
        if gap is not None and (best is None or gap < best[0]):
            best = (gap, line.rstrip(":"))
    return best[1] if best else None


def running_head(page: pymupdf.Page) -> str:
    """The section trail printed at the top of the page, e.g. "2.5. Approach"."""
    header = pymupdf.Rect(0, 0, page.rect.width, 34)
    parts = [
        line.strip()
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks")
        if pymupdf.Rect(x0, y0, x1, y1).intersects(header)
        for line in text.split("\n")
        if line.strip() and not WATERMARK_RE.search(line)
    ]
    return " / ".join(dict.fromkeys(parts))


# --- Image writing ----------------------------------------------------------
def flatten(pix: pymupdf.Pixmap) -> pymupdf.Pixmap:
    """Composite a masked image onto white; transparency reads as black elsewhere."""
    if not pix.alpha:
        return pix
    scratch = pymupdf.open()
    page = scratch.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, pixmap=pix)
    return page.get_pixmap()


def write_embedded(doc: pymupdf.Document, page: pymupdf.Page, box: pymupdf.Rect, out: Path) -> bool:
    """Save the embedded image behind `box` verbatim — no page watermark on it."""
    for im in page.get_images(full=True):
        xref, smask = im[0], im[1]
        if not any(box.intersects(r) and r.get_area() > 0.5 * box.get_area() for r in page.get_image_rects(xref)):
            continue
        pix = pymupdf.Pixmap(doc, xref)
        if pix.colorspace is None or pix.colorspace.n > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if smask:
            pix = flatten(pymupdf.Pixmap(pix, pymupdf.Pixmap(doc, smask)))
        pix.save(out)
        return True
    return False


def write_render(page: pymupdf.Page, box: pymupdf.Rect, out: Path, dpi: int) -> bool:
    """Rasterize the page region — keeps vector overlays, keeps the watermark."""
    page.get_pixmap(clip=box & page.rect, dpi=dpi).save(out)
    return True


# --- Per-document work ------------------------------------------------------
def convert_one(path: Path, img_dir: Path, mode: str, dpi: int, captioned_only: bool) -> dict:
    """Worker: extract every diagram in one PDF, return its Markdown section."""
    figures: list[dict] = []
    with pymupdf.open(path) as doc:
        title = doc_title(doc, path)
        skip = logo_xrefs(doc)
        out_dir = img_dir / path.stem
        for pno, page in enumerate(doc):
            if pno == 0:  # front cover art, never a diagram
                continue
            for n, box in enumerate(figure_rects(page, skip), 1):
                cap = caption_for(page, box)
                if cap:
                    kind, number, text = cap
                    label, stem = f"{kind} {number}: {text}", slugify(f"{number}-{text}")
                elif captioned_only:
                    continue
                else:  # no "Figure N-N:" nearby — name it after its heading
                    near = nearby_label(page, box) or running_head(page) or "Untitled diagram"
                    label = f"{near} (page {pno + 1}, uncaptioned)"
                    stem = f"{slugify(near) or 'diagram'}-p{pno + 1:03d}-{n}"
                out = out_dir / f"{path.stem}-{stem}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                ok = (
                    write_render(page, box, out, dpi)
                    if mode == "render"
                    else write_embedded(doc, page, box, out) or write_render(page, box, out, dpi)
                )
                if ok:
                    figures.append(
                        {
                            "label": label,
                            "page": pno + 1,
                            "section": running_head(page),
                            "path": out,
                            "captioned": cap is not None,
                        }
                    )
        pages = doc.page_count

    body = [f"## {title}\n", f"*Source: `{path.name}` — {pages} pages, {len(figures)} diagrams*\n"]
    for fig in figures:
        rel = fig["path"].relative_to(img_dir.parent).as_posix()
        body.append(f"### {fig['label']}\n")
        trail = f" — {fig['section']}" if fig["section"] else ""
        body.append(f"*Page {fig['page']}{trail}*\n")
        alt = fig["label"].replace("[", "(").replace("]", ")")  # keep the link parseable
        body.append(f"![{alt}]({rel})\n")
    return {
        "source": path.name,
        "title": title,
        "anchor": slugify(title),
        "pages": pages,
        "figures": len(figures),
        "uncaptioned": sum(1 for f in figures if not f["captioned"]),
        "markdown": "\n".join(body),
    }


def assemble(sections: list[dict], mode: str, dpi: int) -> str:
    total = sum(s["figures"] for s in sections)
    out = [
        "---",
        f'title: "{BUNDLE_TITLE}"',
        f"converted: {time.strftime('%Y-%m-%d')}",
        f"extractor: pymupdf ({mode}{f', {dpi} dpi' if mode == 'render' else ''}; diagrams only, no body text)",
        f"documents: {len(sections)}",
        f"diagrams: {total}",
        "---",
        "",
        f"# {BUNDLE_TITLE}",
        "",
        f"{total} diagrams from {len(sections)} documents, with their captions and "
        "the section each appears in. Body text is in `../source-text.md`.",
        "",
        "## Contents",
        "",
    ]
    out += [
        f"{i}. [{s['title']}](#{s['anchor']}) — `{s['source']}`, {s['figures']} diagrams"
        for i, s in enumerate(sections, 1)
    ]
    out.append("")
    for s in sections:
        out += ["---", "", s["markdown"], ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pdfs", nargs="*", type=Path, help="PDFs to scan (default: all in --input)")
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=PDF_DIR,
        help="directory of PDFs",
    )
    ap.add_argument(
        "-o", "--output", type=Path, default=DIAGRAM_DIR, help="output directory"
    )
    ap.add_argument("-j", "--jobs", type=int, default=4, help="parallel workers (default: 4)")
    ap.add_argument(
        "--mode",
        choices=["image", "render"],
        default="image",
        help="image: save the embedded artwork (no watermark); render: rasterize the page region",
    )
    ap.add_argument("--dpi", type=int, default=200, help="resolution for --mode render (default: 200)")
    ap.add_argument(
        "--captioned-only",
        action="store_true",
        help="skip artwork that has no 'Figure N-N:' caption near it",
    )
    args = ap.parse_args()

    pdfs = sorted(args.pdfs or args.input.glob("*.pdf"), key=sort_key)
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 1

    img_dir = args.output / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output / "diagrams-index.md"
    print(f"Scanning {len(pdfs)} PDF(s) for diagrams ({args.mode}) -> {args.output}\n")

    results: dict[Path, dict] = {}
    failures = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(convert_one, p, img_dir, args.mode, args.dpi, args.captioned_only): p
            for p in pdfs
        }
        for done in concurrent.futures.as_completed(futures):
            pdf = futures[done]
            try:
                results[pdf] = r = done.result()
                extra = f" ({r['uncaptioned']} without a Figure caption)" if r["uncaptioned"] else ""
                print(f"  {pdf.name}: {r['figures']} diagrams{extra}")
            except Exception as exc:  # keep going; report at the end
                failures += 1
                print(f"  FAILED {pdf.name}: {exc}", file=sys.stderr)

    sections = [results[p] for p in pdfs if p in results]  # restore reading order
    if not sections:
        print("Nothing extracted.", file=sys.stderr)
        return 1

    out_file.write_text(assemble(sections, args.mode, args.dpi), "utf-8")
    total = sum(s["figures"] for s in sections)
    size = sum(f.stat().st_size for f in img_dir.rglob("*.png")) / 1_048_576
    print(
        f"\nDone. {total} diagrams from {len(sections)} of {len(pdfs)} PDFs"
        f"{f', {failures} failed' if failures else ''}."
        f"\nWrote {out_file} and {img_dir} ({size:.1f} MB of PNGs)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
