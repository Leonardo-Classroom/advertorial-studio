"""Turn an uploaded .docx into the same shape `ppt_extract` yields for a deck.

A Word file has no slides, so the two things the picture pipeline leans on have
to be re-derived from what a document does have:

  `slide_index` becomes a **block** number — the heading-led section a picture
  sits in. Headings are what a reader would point at ("the photo in 產品介紹"),
  and a document with no headings is simply one block rather than a hundred
  paragraph-sized ones nobody could review.

  `nearby_text` becomes the paragraphs either side of the picture instead of
  the shapes nearest it on a canvas. .docx has no coordinates to measure, but
  it has something better: pictures sit *in* the text, in reading order, so
  the neighbouring paragraphs are a stronger hint than a spatial guess ever was.

Both are labelled as blocks, not slides, everywhere they surface — see
`BriefSourceFile.unit_label`.

Word records images as relationships and refers to them from the body by id,
in two dialects: modern DrawingML (`a:blip`) and legacy VML (`v:imagedata`),
the latter being what older Word versions and pasted content still produce.
Both are read here, because a deck that loses half its pictures to a namespace
is indistinguishable from one that had none.
"""
from __future__ import annotations

from pathlib import Path

# python-docx does not carry the VML namespace in its nsmap, so `qn()` cannot
# resolve it and the URI is spelled out here.
VML_NS = "urn:schemas-microsoft-com:vml"
VML_IMAGEDATA = f"{{{VML_NS}}}imagedata"

# How much text either side of a picture is worth keeping as a hint.
NEARBY_CHARS = 300


def _is_heading(paragraph) -> bool:
    name = (paragraph.style.name or "") if paragraph.style is not None else ""
    return name.startswith("Heading") or name in ("Title", "Subtitle")


def _iter_block_items(parent):
    """Yield the paragraphs and tables of `parent` in document order.

    python-docx exposes `.paragraphs` and `.tables` as separate lists, which
    loses the interleaving — and the interleaving is the whole point here,
    since a picture's neighbours are whatever actually precedes and follows it.
    """
    from docx.document import Document as _Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph

    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:  # pragma: no cover - defensive
        raise ValueError(f"不支援的容器：{type(parent)!r}")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _table_lines(table) -> list[str]:
    """One line per row, cells joined — same shape `ppt_extract` uses."""
    lines = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return lines


def extract_text(path: str | Path) -> tuple[str, int]:
    """Return (all text, block count) from a .docx.

    The block count stands in for a deck's slide count. It is deliberately not
    a page count: pagination is decided by Word's layout engine at render time
    and python-docx cannot know it without one.
    """
    import docx

    document = docx.Document(str(path))
    blocks: list[tuple[str, list[str]]] = [("", [])]

    for item in _iter_block_items(document):
        if hasattr(item, "rows"):                       # a table
            blocks[-1][1].extend(_table_lines(item))
            continue
        text = item.text.strip()
        if not text:
            continue
        if _is_heading(item):
            # The heading goes in the block's label, not its body — repeating it
            # would spend the model's context saying the same thing twice.
            blocks.append((text, []))
        else:
            blocks[-1][1].append(text)

    # A leading empty block only exists when the document opens with a heading.
    if not blocks[0][1]:
        blocks.pop(0)

    chunks = []
    for i, (heading, lines) in enumerate(blocks, 1):
        label = f"--- 第 {i} 段：{heading} ---" if heading else f"--- 第 {i} 段 ---"
        chunks.append(label + "\n" + "\n".join(lines))

    return "\n\n".join(chunks), len(blocks)


def _relationship_ids(paragraph) -> list[str]:
    """Every image relationship id referenced by this paragraph, in order."""
    from docx.oxml.ns import qn

    ids: list[str] = []
    for blip in paragraph._p.iter(qn("a:blip")):
        rid = blip.get(qn("r:embed"))
        if rid:
            ids.append(rid)
    for imagedata in paragraph._p.iter(VML_IMAGEDATA):
        rid = imagedata.get(qn("r:id"))
        if rid:
            ids.append(rid)
    return ids


def _blob(part, rid: str) -> tuple[bytes, str] | None:
    """(bytes, extension) for one relationship, or None if it is not an image."""
    try:
        image_part = part.related_parts[rid]
    except KeyError:
        return None
    blob = getattr(image_part, "blob", None)
    if blob is None:
        return None
    ext = ""
    image = getattr(image_part, "image", None)
    if image is not None:
        ext = (getattr(image, "ext", "") or "").lower()
    if not ext:
        ext = Path(str(getattr(image_part, "partname", ""))).suffix.lstrip(".").lower()
    return blob, ext


def extract_images(path: str | Path, max_nearby_chars: int = NEARBY_CHARS) -> list[dict]:
    """Collect every embedded picture, with the context needed to judge it.

    Returns the same dict shape `ppt_extract.extract_images` does, so the
    filter/classify/store pipeline in `briefs/services/images.py` needs no
    branch of its own. `left`/`top` come back None — a Word picture has no
    canvas position, and the sort in `images.slide_order` already treats them
    as optional, falling back to document order within a block.
    """
    import docx

    document = docx.Document(str(path))
    part = document.part

    # One pass to lay the document out flat, so a picture can see both the
    # paragraph before it and the one after — the latter is not yet known
    # while walking forward.
    items: list[dict] = []
    block_index = 0
    heading = ""
    seen_content = False

    for item in _iter_block_items(document):
        if hasattr(item, "rows"):                       # a table
            for line in _table_lines(item):
                items.append({"kind": "text", "text": line,
                              "block": max(block_index, 1), "heading": heading})
                seen_content = True
            continue

        if _is_heading(item) and item.text.strip():
            block_index += 1
            heading = item.text.strip()
            items.append({"kind": "text", "text": heading,
                          "block": block_index, "heading": heading})
            seen_content = True
            continue

        if not seen_content:
            block_index = 1                             # content before any heading
            seen_content = True
        current = max(block_index, 1)

        text = item.text.strip()
        if text:
            items.append({"kind": "text", "text": text,
                          "block": current, "heading": heading})
        for rid in _relationship_ids(item):
            items.append({"kind": "image", "rid": rid,
                          "block": current, "heading": heading})

    def nearby_for(position: int) -> str:
        """Nearest text before, then after — whichever exists."""
        before = next((items[j]["text"] for j in range(position - 1, -1, -1)
                       if items[j]["kind"] == "text"), "")
        after = next((items[j]["text"] for j in range(position + 1, len(items))
                      if items[j]["kind"] == "text"), "")
        joined = "\n".join(t for t in (before, after) if t)
        return joined[:max_nearby_chars]

    out: list[dict] = []
    for i, item in enumerate(items):
        if item["kind"] != "image":
            continue
        found = _blob(part, item["rid"])
        if found is None:
            continue                                    # linked, or not an image
        blob, ext = found
        out.append({
            "slide_index": item["block"],
            "left": None, "top": None, "width": None, "height": None,
            "blob": blob,
            "ext": ext,
            "slide_heading": item["heading"][:200],
            "nearby_text": nearby_for(i),
        })

    return out
