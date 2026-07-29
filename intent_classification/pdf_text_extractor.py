from __future__ import annotations

from pathlib import Path


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text from a text-based PDF.

    Scanned/image-only PDFs should go through OCR before intent classification.
    """

    path = Path(pdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF text extraction requires the 'pypdf' package.") from exc

    reader = PdfReader(str(path))
    page_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page_text).strip()
    if not text:
        raise ValueError(
            "No text could be extracted from this PDF. If it is scanned, run OCR first."
        )
    return text
