# study-assist

A local tool for studying your own PDF documents. It converts the PDFs to Markdown, extracts their diagrams, and loads the text into pgvector for RAG using local Ollama models.

## Source documents are not included

This repository contains no study materials. Add your own copies of the PDFs to `data/pdfs/`. Your use of them is subject to their license terms.

Git ignores everything in `data/` except the empty folders, so your documents and everything generated from them stay on your machine.

## Repository layout

```text
.
├── study_assist/            Python code
│   ├── pdf_to_markdown.py   Converts PDFs to a single Markdown file
│   ├── pdf_to_diagrams.py   Extracts diagrams and their captions
│   ├── knowledge_base.py    Splits the text into passages; BM25 search
│   ├── ingest_pgvector.py   Embeds passages and loads them into pgvector
│   └── paths.py             Default data locations
├── data/                    Your local files, ignored by git
│   ├── pdfs/                PDFs to convert
│   ├── decks/               Study decks, which the scripts don't read
│   └── generated/           Converted text and diagrams
├── db/init/                 Schema, applied when the database is first created
├── compose.yaml             pgvector database
├── .env.example             Database settings template
└── requirements.txt         Python dependencies
```

## Setup

You need Python 3.10 or later, Docker with Compose, and [Ollama](https://ollama.com). Run all commands from the repository root.

### 1. Install the dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pymupdf4llm
ollama pull bge-m3
```

### 2. Add your PDFs to `data/pdfs/`, then convert them

```bash
python3 -m study_assist.pdf_to_markdown
python3 -m study_assist.pdf_to_diagrams
```

### 3. Start the database

```bash
cp .env.example .env    # then set POSTGRES_PASSWORD
docker compose up -d
```

The database listens on `localhost:5433`. To use a different port, change `POSTGRES_PORT` in `.env`.

### 4. Load the passages

```bash
python3 -m study_assist.ingest_pgvector --dry-run    # check what would load
python3 -m study_assist.ingest_pgvector
```

Each run replaces the contents of the `chunks` table.

## Keyword search

`knowledge_base.py` provides BM25 search over the generated Markdown, with no database or models required:

```bash
python3 -m study_assist.knowledge_base --stats
python3 -m study_assist.knowledge_base "risk assessment"
python3 -m study_assist.knowledge_base --text "gap analysis"
```
