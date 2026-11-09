#!/usr/bin/env python3
"""Ask questions about the documents and get answers that cite their sources.

Each question is searched two ways: by meaning in pgvector and by keyword with BM25.
The results are merged, and a local Ollama model answers from those passages alone. If
the database is unavailable, the search falls back to BM25.

Usage:
    python3 -m study_assist.chat                          # start a chat
    python3 -m study_assist.chat "What is the ADM?"       # ask one question
    python3 -m study_assist.chat --model qwen3:8b         # use another Ollama model
    python3 -m study_assist.chat --show-context "..."     # also print the passages used
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

import httpx
import numpy as np
import ollama
import psycopg

from study_assist.ingest_pgvector import MODEL as EMBEDDING_MODEL
from study_assist.ingest_pgvector import connect, is_front_or_back_matter
from study_assist.knowledge_base import KnowledgeBase
from study_assist.paths import GENERATED_DIR

DEFAULT_MODEL = os.environ.get("STUDY_CHAT_MODEL", "llama3:8b")
PASSAGES_PER_ANSWER = 4  # each one adds about 25 seconds of waiting on CPU
CANDIDATES_PER_SEARCH = 20
RRF_K = 60  # the usual constant for reciprocal rank fusion
HISTORY_MESSAGES = 6  # three questions and answers
# Ollama's default context is small enough to silently cut off the passages.
CONTEXT_TOKENS = 8192
# Ollama's default thread count was 2-3x slower on a CPU with mixed core types.
THREADS = min(8, os.cpu_count() or 8)
KEEP_MODEL_LOADED = "30m"  # avoids a reload of about 11 seconds between questions

SYSTEM_PROMPT = """You are a study assistant for TOGAF and the related documents supplied to you.

Answer using only the numbered passages in the user's message.
- Cite each claim with its passage number, such as [2].
- If the passages don't contain the answer, say so instead of using outside knowledge.
- When a passage gives a list, include every item, even ones worded differently from the rest.
- Be direct and clear. Use a short list when the answer has several parts."""

# Words that usually refer back to an earlier question.
FOLLOW_UP_RE = re.compile(
    r"\b(it|its|they|them|their|this|that|these|those|he|she|one|ones)\b|^(and|also|what about|how about)\b",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\[(\d+)\]")
# BM25 drops "a" as a stopword and single letters carry little meaning, so search by name too.
ADM_PHASES = {
    "a": "Architecture Vision",
    "b": "Business Architecture",
    "c": "Information Systems Architectures",
    "d": "Technology Architecture",
    "e": "Opportunities and Solutions",
    "f": "Migration Planning",
    "g": "Implementation Governance",
    "h": "Architecture Change Management",
}
ADM_PHASE_RE = re.compile(r"\bphase ([a-h])\b", re.IGNORECASE)


@dataclass(frozen=True)
class Hit:
    source: str
    section: str | None
    text: str

    def cite(self) -> str:
        return f"{self.source} — {self.section or 'front matter'}"


def keyword_search(kb: KnowledgeBase, query: str) -> list[Hit]:
    hits = kb.search_passages(query, limit=CANDIDATES_PER_SEARCH)
    return [
        Hit(f"{p.doc_title} ({p.doc_code})", p.heading or None, p.text)
        for p, _ in hits
        if not is_front_or_back_matter(p.text)
    ]


def vector_search(query: str) -> list[Hit]:
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=query).embeddings[0]
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT source, section, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (np.array(embedding, dtype=np.float32), CANDIDATES_PER_SEARCH),
        )
        return [Hit(*row) for row in cur.fetchall()]


def merge(*rankings: list[Hit]) -> list[Hit]:
    """Combine ranked lists, favoring passages that rank well in either or both."""
    scores: dict[str, float] = defaultdict(float)
    by_text: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.text] += 1 / (RRF_K + rank)
            by_text.setdefault(hit.text, hit)
    ordered = sorted(scores, key=lambda text: -scores[text])
    return [by_text[text] for text in ordered]


def retrieve(kb: KnowledgeBase, query: str) -> list[Hit]:
    rankings = [keyword_search(kb, query)]
    try:
        rankings.append(vector_search(query))
    except psycopg.OperationalError:
        print("[database unavailable: using keyword search only]", file=sys.stderr)
    return merge(*rankings)[:PASSAGES_PER_ANSWER]


def search_query(question: str, previous_question: str | None) -> str:
    """Add the previous question to a follow-up, so "what are its outputs?" can find something."""
    query = question
    if previous_question and FOLLOW_UP_RE.search(question):
        query = f"{previous_question} {question}"
    names = {ADM_PHASES[letter.lower()] for letter in ADM_PHASE_RE.findall(query)}
    return " ".join([query, *sorted(names)])


def format_passages(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{n}] {hit.cite()}\n{hit.text}" for n, hit in enumerate(hits, start=1))


def stream_answer(model: str, messages: list[dict[str, str]]) -> str:
    """Print the answer as it arrives and return what was printed, even if interrupted."""
    parts: list[str] = []
    try:
        for chunk in ollama.chat(
            model=model,
            messages=messages,
            stream=True,
            keep_alive=KEEP_MODEL_LOADED,
            options={"num_ctx": CONTEXT_TOKENS, "num_thread": THREADS, "temperature": 0.2},
        ):
            piece = chunk.message.content or ""
            parts.append(piece)
            print(piece, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[stopped]", end="")
    print()
    return "".join(parts)


def print_sources(answer: str, hits: list[Hit]) -> None:
    cited = sorted({int(n) for n in CITATION_RE.findall(answer) if 1 <= int(n) <= len(hits)})
    if not cited:
        print("\nSources: none cited")
        return
    print("\nSources:")
    for n in cited:
        print(f"  [{n}] {hits[n - 1].cite()}")


def ask(
    kb: KnowledgeBase,
    model: str,
    question: str,
    history: list[dict[str, str]],
    show_context: bool = False,
) -> None:
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    hits = retrieve(kb, search_query(question, previous))
    if not hits:
        print("No matching passages found. Try different wording.")
        return
    if show_context:
        print(f"--- passages ---\n{format_passages(hits)}\n--- answer ---")
    print(f"[reading {len(hits)} passages; this can take a minute or two on CPU]", file=sys.stderr)
    prompt = f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"
    answer = stream_answer(
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-HISTORY_MESSAGES:],
            {"role": "user", "content": prompt},
        ],
    )
    print_sources(answer, hits)
    # Earlier passages are left out because passage numbers restart with each question.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": CITATION_RE.sub("", answer)})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("question", nargs="*", help="ask this one question and exit")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama chat model (default: {DEFAULT_MODEL})")
    ap.add_argument("--show-context", action="store_true", help="print the passages given to the model")
    args = ap.parse_args()

    try:
        kb = KnowledgeBase.load(GENERATED_DIR)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    history: list[dict[str, str]] = []
    try:
        if args.question:
            ask(kb, args.model, " ".join(args.question), history, args.show_context)
            return 0

        print(f"Chatting with {args.model}. Press Ctrl+C to stop an answer. Type 'exit' to quit.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {"exit", "quit"}:
                return 0
            if question:
                print()
                ask(kb, args.model, question, history, args.show_context)
    except (ConnectionError, httpx.TransportError):
        print("Lost the connection to Ollama. Check that it's running, then try again.", file=sys.stderr)
    except ollama.ResponseError as error:
        print(f"Ollama error: {error.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
