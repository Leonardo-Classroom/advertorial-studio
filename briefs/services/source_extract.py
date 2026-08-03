"""Dispatch a source file to its format-specific extractor.

One entry point per format, added as each format's parser lands: this phase
only wires up `.pptx`, `.docx`/`.pdf` are declared in `BriefSourceFile.FORMATS`
but have no branch here yet. Callers never talk to `ppt_extract` (or, later,
a `docx_extract`/`pdf_extract`) directly — this module is the only thing that
knows which parser a format uses, so adding one means adding a branch here,
not touching every call site that currently assumes `.pptx`.
"""
from __future__ import annotations

from pathlib import Path

EXTENSION_FORMATS = {
    ".pptx": "pptx",
}


def format_for(filename: str) -> str:
    """Guess a `BriefSourceFile.format` value from a filename's extension.

    Raises for anything not yet supported, so an unsupported file is rejected
    at upload time rather than accepted and failed later in the background.
    """
    ext = Path(filename).suffix.lower()
    fmt = EXTENSION_FORMATS.get(ext)
    if not fmt:
        raise ValueError(f"不支援的檔案格式：{ext or '（無副檔名）'}")
    return fmt


def extract_text(path: str, format: str) -> tuple[str, int]:
    """(text, page/slide count) for one file, dispatched by format."""
    if format == "pptx":
        from briefs.services import ppt_extract

        return ppt_extract.extract_text(path)
    raise ValueError(f"格式 {format} 尚未支援文字抽取")


def extract_images(path: str, format: str) -> list[dict]:
    """Picture dicts (see `ppt_extract.extract_images` for the shape), by format."""
    if format == "pptx":
        from briefs.services import ppt_extract

        return ppt_extract.extract_images(path)
    raise ValueError(f"格式 {format} 尚未支援圖片抽取")
