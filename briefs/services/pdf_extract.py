"""Turn an uploaded .pdf into the same shape `ppt_extract` yields for a deck.

Of the three formats this is the closest fit to the deck model: a PDF has
pages, pages have coordinates, so `slide_index` is the page number and
`nearby_text` goes back to being the spatial guess it is for .pptx — nearest
text to the picture, measured.

Pictures are taken by **rendering the page and cropping**, not by decoding the
embedded image objects. That trades some fidelity (the crop is only as sharp
as the render) for a great deal of reliability: a PDF's embedded images come in
a dozen filters and colour spaces, several of which need a decoder this project
would otherwise have no reason to carry, and the material here is photographs
being judged for whether they belong in an article — a judgement 150 DPI serves
perfectly well. It also means the crop shows the picture *as the reader sees
it*, masks and clipping included.

`pdfplumber` was chosen over PyMuPDF deliberately: PyMuPDF is AGPL-3.0, which
is a problem for procurement here. pdfplumber (MIT) over pdfminer.six (MIT),
rendering through pypdfium2 (BSD-3/Apache-2.0), keeps the licence chain clean.

Scanned PDFs are refused rather than OCR'd — see `SCANNED_WORDS_PER_PAGE`.
"""
from __future__ import annotations

import io
from pathlib import Path

# What the page is rendered at before cropping. 150 DPI puts a full-width A4
# photograph at roughly 1200px on its long edge — ample for the vision model,
# and a quarter of the memory of 300.
RENDER_DPI = 150

# PDF coordinates are in points (72 per inch). Anything smaller than this in
# either direction is a rule, a bullet or an icon; skipping it here saves the
# crop, and `images.filter_candidates` would drop it moments later anyway.
MIN_POINTS = 30

# A text-layer PDF runs to hundreds of words a page. A scan has none at all, so
# this floor only has to separate "has a text layer" from "does not" — it is
# not trying to judge how wordy a document is.
SCANNED_WORDS_PER_PAGE = 5

NEARBY_CHARS = 300


def _gap(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    """Shortest distance between two rectangles; 0 when they overlap.

    Same figure/caption geometry `ppt_extract._gap` does for slides. Kept as
    its own copy rather than imported because the two modules disagree about
    what a rectangle *is* — .pptx measures in EMU from the shape tree, this
    measures in points from a rendered page — and a shared helper would invite
    someone to "fix" one caller's units into the other's.
    """
    dx = max(0, a[0] - b[2], b[0] - a[2])
    dy = max(0, a[1] - b[3], b[1] - a[3])
    return (dx * dx + dy * dy) ** 0.5


def _page_lines(page) -> list[tuple[tuple[float, float, float, float], str]]:
    """(box, text) for each text line on the page, in reading order."""
    out = []
    for line in page.extract_text_lines():
        text = (line.get("text") or "").strip()
        if text:
            out.append(((line["x0"], line["top"], line["x1"], line["bottom"]), text))
    return out


def extract_text(path: str | Path) -> tuple[str, int]:
    """Return (all text, page count) from a .pdf.

    Raises when the file has no text layer, which is what a scan is. There is
    nothing useful to do with one here: the facts are extracted by reading, and
    OCR is a different piece of engineering that this phase deliberately does
    not take on. Saying so plainly beats handing back an empty document and
    letting the user wonder why their brief came out blank.
    """
    import pdfplumber

    chunks: list[str] = []
    total_words = 0
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            total_words += len(page.extract_words())
            if text:
                chunks.append(f"--- 第 {i} 頁 ---\n{text}")
            # pdfplumber keeps every page's parsed objects alive on the page
            # once touched; on a long proposal that is the difference between
            # a few MB and a few hundred.
            page.flush_cache()

    if page_count and total_words / page_count < SCANNED_WORDS_PER_PAGE:
        raise ValueError(
            "這份 PDF 幾乎沒有文字層，看起來是掃描檔或純圖片檔，目前不支援。"
            "請改用原始的可選取文字版本，或改上傳 .pptx／.docx。")

    return "\n\n".join(chunks), page_count


def _crop_box(image: dict, page, scale: float) -> tuple[int, int, int, int] | None:
    """Pixel crop box for one placed image, clamped to the rendered page.

    A PDF may place a picture partly off-page, or record its box inverted;
    either would make PIL raise or hand back an empty crop.
    """
    x0, x1 = sorted((float(image["x0"]), float(image["x1"])))
    top, bottom = sorted((float(image["top"]), float(image["bottom"])))
    if (x1 - x0) < MIN_POINTS or (bottom - top) < MIN_POINTS:
        return None

    x0 = max(0.0, min(x0, float(page.width)))
    x1 = max(0.0, min(x1, float(page.width)))
    top = max(0.0, min(top, float(page.height)))
    bottom = max(0.0, min(bottom, float(page.height)))
    box = (round(x0 * scale), round(top * scale), round(x1 * scale), round(bottom * scale))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return box


def extract_images(path: str | Path, max_nearby_chars: int = NEARBY_CHARS) -> list[dict]:
    """Collect every placed picture, with the context needed to judge it.

    Returns the same dict shape `ppt_extract.extract_images` does, so the
    filter/classify/store pipeline in `briefs/services/images.py` needs no
    branch of its own.

    A page is only rendered when it actually holds a picture — rendering every
    page of a long PDF to find out it was all text is the one genuinely
    expensive mistake available here.
    """
    import pdfplumber

    out: list[dict] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            if not page.images:
                page.flush_cache()          # see extract_text
                continue

            lines = _page_lines(page)
            # The topmost line stands in for a slide's title: on a paged
            # document it is nearly always the heading of what follows.
            heading = min(lines, key=lambda item: item[0][1])[1] if lines else ""

            try:
                rendered = page.to_image(resolution=RENDER_DPI).original
            except Exception:  # noqa: BLE001 - one unrenderable page must not
                continue       # cost the pictures on every other page
            scale = rendered.size[0] / float(page.width)

            for image in page.images:
                box = _crop_box(image, page, scale)
                if box is None:
                    continue
                try:
                    crop = rendered.crop(box)
                    buffer = io.BytesIO()
                    crop.convert("RGB").save(buffer, format="PNG")
                except Exception:  # noqa: BLE001 - see above, per picture
                    continue

                spot = (float(image["x0"]), float(image["top"]),
                        float(image["x1"]), float(image["bottom"]))
                nearby = ""
                if lines:
                    nearby = min(lines, key=lambda item: _gap(spot, item[0]))[1]

                out.append({
                    "slide_index": i,
                    "left": image["x0"], "top": image["top"],
                    "width": float(image["x1"]) - float(image["x0"]),
                    "height": float(image["bottom"]) - float(image["top"]),
                    "blob": buffer.getvalue(),
                    "ext": "png",
                    "slide_heading": heading[:200],
                    "nearby_text": nearby[:max_nearby_chars],
                })

            del rendered
            page.flush_cache()              # see extract_text

    return out
