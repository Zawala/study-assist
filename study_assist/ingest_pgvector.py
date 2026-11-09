#!/usr/bin/env python3
"""Embed the document passages with Ollama and load them into pgvector.

Passages come from knowledge_base.py, so run pdf_to_markdown.py first if the Markdown is
missing. Every run replaces the contents of the `chunks` table. All embeddings are
computed before the table is touched, and the replacement happens in one transaction,
so a failed run leaves the existing data in place.

Usage:
    python3 -m study_assist.ingest_pgvector --dry-run    # show what would load; needs no services
    python3 -m study_assist.ingest_pgvector              # embed everything and replace the table
    python3 -m study_assist.ingest_pgvector --limit 50   # load a small sample while testing
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from study_assist.knowledge_base import KnowledgeBase, Passage
from study_assist.paths import GENERATED_DIR, REPO

MODEL = "bge-m3"
DIMENSIONS = 1024  # must match VECTOR(1024) in db/init/001_schema.sql
# Long passages can split the notice across windows, so match its start and its end.
COPYRIGHT_NOTICE = (
    "No part of this publication may be reproduced",
    "Any use of this publication for commercial purposes",
)
TOC_DOTS = ". . . ."
TOC_ENTRY_RE = re.compile(r" — (?:[ivxlc]+|\d{1,3})\b")  # "- Preface — ix"
INDEX_ENTRY_RE = re.compile(r"\b[a-z][\w -]*, \d{1,3}(?:, \d{1,3})*")  # "capability, 46"
MAX_TOC_ENTRIES = 4
MAX_INDEX_ENTRIES = 9


def is_front_or_back_matter(text: str) -> bool:
    """True for copyright, table-of-contents, or index text, which has no content to retrieve."""
    return (
        any(phrase in text for phrase in COPYRIGHT_NOTICE)
        or TOC_DOTS in text
        or len(TOC_ENTRY_RE.findall(text)) > MAX_TOC_ENTRIES
        or len(INDEX_ENTRY_RE.findall(text)) > MAX_INDEX_ENTRIES
    )


def load_passages() -> list[Passage]:
    passages = KnowledgeBase.load(GENERATED_DIR).passages
    return [p for p in passages if not is_front_or_back_matter(p.text)]


def embedding_text(passage: Passage) -> str:
    """Include the title and heading trail, which body text often leaves implicit."""
    return f"{passage.doc_title}\n{passage.heading}\n\n{passage.text}"


def embed(passages: list[Passage], batch_size: int) -> list[list[float]]:
    import ollama  # noqa: PLC0415

    vectors: list[list[float]] = []
    started = time.monotonic()
    for at in range(0, len(passages), batch_size):
        batch = [embedding_text(p) for p in passages[at : at + batch_size]]
        vectors.extend(ollama.embed(model=MODEL, input=batch).embeddings)
        print(f"  embedded {len(vectors)}/{len(passages)} ({time.monotonic() - started:.0f}s)")
    if len(vectors) != len(passages):
        raise RuntimeError(f"expected {len(passages)} embeddings, got {len(vectors)}")
    if wrong := {len(v) for v in vectors} - {DIMENSIONS}:
        raise RuntimeError(f"expected {DIMENSIONS} dimensions, got {sorted(wrong)}")
    return vectors


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = (
        line.split("=", 1)
        for line in path.read_text("utf-8").splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    return {key.strip(): value.strip() for key, value in pairs}


def connect():
    import psycopg  # noqa: PLC0415
    from pgvector.psycopg import register_vector  # noqa: PLC0415

    env = read_env_file(REPO / ".env") | os.environ
    if "POSTGRES_PASSWORD" not in env:
        raise SystemExit("POSTGRES_PASSWORD is not set; add it to .env")
    conn = psycopg.connect(
        host=env.get("POSTGRES_HOST", "localhost"),
        port=env.get("POSTGRES_PORT", "5433"),
        user=env.get("POSTGRES_USER", "postgres"),
        password=env["POSTGRES_PASSWORD"],
        dbname=env.get("POSTGRES_DB", "study"),
    )
    register_vector(conn)
    return conn


def replace_chunks(passages: list[Passage], vectors: list[list[float]]) -> None:
    import numpy as np  # noqa: PLC0415

    rows = [
        (f"{p.doc_title} ({p.doc_code})", p.heading or None, p.text, np.array(vector, dtype=np.float32))
        for p, vector in zip(passages, vectors, strict=True)
    ]
    with connect() as conn, conn.cursor() as cur:  # commits on success, rolls back on error
        cur.execute("TRUNCATE chunks RESTART IDENTITY")
        cur.executemany(
            "INSERT INTO chunks (source, section, content, embedding) VALUES (%s, %s, %s, %s)",
            rows,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true", help="report passages without embedding or writing")
    ap.add_argument("--limit", type=int, help="only load the first N passages")
    ap.add_argument("--batch-size", type=int, default=32, help="passages per embedding call (default: 32)")
    args = ap.parse_args()

    passages = load_passages()[: args.limit]
    documents = len({p.doc_code for p in passages})
    print(f"{len(passages)} passages from {documents} documents")

    if args.dry_run:
        sample = embedding_text(passages[0])[:300] if passages else "(none)"
        print(f"\nFirst text to embed:\n{sample}")
        return 0
    if not passages:
        print("No passages to load; leaving the table unchanged.", file=sys.stderr)
        return 1

    print(f"Embedding with {MODEL}...")
    vectors = embed(passages, args.batch_size)
    print("Replacing the chunks table...")
    replace_chunks(passages, vectors)
    print(f"Done. Loaded {len(passages)} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
