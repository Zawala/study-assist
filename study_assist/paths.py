"""Where the project keeps its local data."""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PDF_DIR = REPO / "data" / "pdfs"
GENERATED_DIR = REPO / "data" / "generated"
TEXT_FILE = GENERATED_DIR / "source-text.md"
DIAGRAM_DIR = GENERATED_DIR / "diagrams"
