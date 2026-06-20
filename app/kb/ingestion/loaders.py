"""Load supported document types to plain text."""
from __future__ import annotations

from pathlib import Path


class UnsupportedFileType(ValueError):
    pass


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _load_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_text,
    ".md": _load_text,
    ".markdown": _load_text,
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS)


def load_document(path: str | Path) -> str:
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileType(
            f"Unsupported file type {path.suffix!r}; supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return loader(path)
