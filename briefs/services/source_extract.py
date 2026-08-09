"""Dispatch a source file to its format-specific extractor.

One entry point per format. Callers never talk to `ppt_extract`,
`docx_extract` or `pdf_extract` directly — this module is the only thing that
knows which parser a format uses, so adding one means adding a branch here,
not touching every call site that would otherwise assume `.pptx`.
"""
from __future__ import annotations

from pathlib import Path

EXTENSION_FORMATS = {
    ".pptx": "pptx",
    ".docx": "docx",
    ".pdf": "pdf",
}

# Formats that are zip containers underneath, and so can be decompression
# bombs. A pdf is not one of these: it has its own compressed streams, but
# nothing here unpacks them wholesale the way picture extraction unpacks a
# deck's media parts.
ARCHIVE_FORMATS = {"pptx", "docx"}


def _human(size: int) -> str:
    return f"{size / 1024 / 1024:.0f}MB" if size < 1024 ** 3 else f"{size / 1024 ** 3:.1f}GB"


# `SiteSettings` field -> `.env` setting used when the row cannot be read.
# Values are megabytes except the ratio, which is a plain multiple.
_LIMIT_SOURCES = {
    "upload_max_file_mb": ("UPLOAD_MAX_FILE_BYTES", 200 * 1024 * 1024),
    "upload_max_batch_mb": ("UPLOAD_MAX_BATCH_BYTES", 600 * 1024 * 1024),
    "upload_max_unpacked_mb": ("UPLOAD_MAX_UNPACKED_BYTES", 2 * 1024 ** 3),
    "upload_max_compression_ratio": ("UPLOAD_MAX_COMPRESSION_RATIO", 200),
}


def limits() -> dict:
    """Upload ceilings in bytes, from 高級設定 with `.env` as the fallback.

    Staff can change these without a redeploy, which is the point — the right
    ceiling depends on what people actually upload here, and that is not
    knowable from the code. Lazy import so `briefs` carries no load-time
    dependency on `studio`, the same shape `core.llm._backend()` uses.
    """
    row = None
    try:
        from studio.models import SiteSettings

        row = SiteSettings.load()
    except Exception:  # noqa: BLE001 - a missing row must not block uploads
        pass

    from django.conf import settings

    out = {}
    for field, (env_name, env_default) in _LIMIT_SOURCES.items():
        value = getattr(row, field, None) if row is not None else None
        if value and value > 0:
            out[field] = value if field.endswith("ratio") else value * 1024 * 1024
        else:
            out[field] = getattr(settings, env_name, env_default)
    return out


def check_upload(upload_file, fmt: str) -> None:
    """Reject a file that is too large, or that would unpack to too much.

    Raises `ValueError` with a message meant for the person uploading; callers
    turn that into a form error (see `briefs.services.upload.create_brief`).

    The archive check reads only the zip's central directory — the sizes each
    member *declares* — so it costs no decompression. That is enough to stop
    the ordinary bomb, whose whole trick is declaring gigabytes; it is not a
    proof, because those numbers are written by whoever made the file. The
    per-file byte cap above is the backstop that does not depend on them.
    """
    caps = limits()

    limit = caps["upload_max_file_mb"]
    size = getattr(upload_file, "size", None) or 0
    if size > limit:
        raise ValueError(
            f"「{upload_file.name}」有 {_human(size)}，超過單檔上限 {_human(limit)}。")

    if fmt not in ARCHIVE_FORMATS:
        return

    import zipfile

    max_unpacked = caps["upload_max_unpacked_mb"]
    max_ratio = caps["upload_max_compression_ratio"]
    try:
        upload_file.seek(0)
        with zipfile.ZipFile(upload_file) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, OSError, ValueError):
        # Not a readable zip. Not this function's problem to report: the
        # parser already fails safely on a renamed text file (任務二 §六),
        # and saying so here would turn one clear error into two.
        return
    finally:
        try:
            upload_file.seek(0)
        except (OSError, ValueError):
            pass

    unpacked = sum(e.file_size for e in entries)
    packed = sum(e.compress_size for e in entries) or 1
    if unpacked > max_unpacked:
        raise ValueError(
            f"「{upload_file.name}」解開後有 {_human(unpacked)}，超過上限 "
            f"{_human(max_unpacked)}，可能是壓縮炸彈。")
    if unpacked / packed > max_ratio:
        raise ValueError(
            f"「{upload_file.name}」的壓縮比為 {unpacked / packed:.0f}:1，"
            f"高於上限 {max_ratio}:1，可能是壓縮炸彈。")


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
    """(text, unit count) for one file, dispatched by format.

    The second number counts whatever that format's reviewable unit is —
    slides for a deck, heading-led blocks for a document. See
    `BriefSourceFile.unit_label` for how it is named on screen.
    """
    if format == "pptx":
        from briefs.services import ppt_extract

        return ppt_extract.extract_text(path)
    if format == "docx":
        from briefs.services import docx_extract

        return docx_extract.extract_text(path)
    if format == "pdf":
        from briefs.services import pdf_extract

        return pdf_extract.extract_text(path)
    raise ValueError(f"格式 {format} 尚未支援文字抽取")


def extract_images(path: str, format: str) -> list[dict]:
    """Picture dicts (see `ppt_extract.extract_images` for the shape), by format."""
    if format == "pptx":
        from briefs.services import ppt_extract

        return ppt_extract.extract_images(path)
    if format == "docx":
        from briefs.services import docx_extract

        return docx_extract.extract_images(path)
    if format == "pdf":
        from briefs.services import pdf_extract

        return pdf_extract.extract_images(path)
    raise ValueError(f"格式 {format} 尚未支援圖片抽取")
