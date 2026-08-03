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

# python-docx carries neither the VML nor the markup-compatibility namespace in
# its nsmap, so `qn()` cannot resolve them and the URIs are spelled out here.
VML_NS = "urn:schemas-microsoft-com:vml"
VML_IMAGEDATA = f"{{{VML_NS}}}imagedata"

MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
MC_ALTERNATE = f"{{{MC_NS}}}AlternateContent"
MC_CHOICE = f"{{{MC_NS}}}Choice"
MC_FALLBACK = f"{{{MC_NS}}}Fallback"

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


def _annotated(document) -> list[tuple[int, str, str, str]]:
    """The document as a flat (block, heading, kind, value) list.

    Both `extract_text` and `extract_images` read this rather than counting
    blocks for themselves. When they each did their own counting they drifted
    apart the moment one of them learned to look inside tables, and a picture
    filed under 第 1 段 while the text called the same place 第 2 段 is the
    kind of disagreement nobody notices until the review page is confusing.
    """
    events = list(_walk(document))
    first_heading = next((i for i, (kind, _) in enumerate(events)
                          if kind == "heading"), len(events))
    # Content before the first heading is a block in its own right, so the
    # first heading becomes block 2. A document that opens with a heading has
    # no such preamble and starts at 1.
    block = 1 if any(kind != "heading" for kind, _ in events[:first_heading]) else 0

    heading = ""
    out: list[tuple[int, str, str, str]] = []
    for kind, value in events:
        if kind == "heading":
            block += 1
            heading = value
        out.append((max(block, 1), heading, kind, value))
    return out


def extract_text(path: str | Path) -> tuple[str, int]:
    """Return (all text, block count) from a .docx.

    The block count stands in for a deck's slide count. It is deliberately not
    a page count: pagination is decided by Word's layout engine at render time
    and python-docx cannot know it without one.
    """
    import docx

    document = docx.Document(str(path))
    blocks: dict[int, tuple[str, list[str]]] = {}

    for block, heading, kind, value in _annotated(document):
        if kind == "image":
            continue
        entry = blocks.setdefault(block, (heading, []))
        # A heading goes in its block's label, not its body — repeating it
        # would spend the model's context saying the same thing twice.
        if kind == "text":
            entry[1].append(value)

    chunks = []
    for i, key in enumerate(sorted(blocks), 1):
        heading, lines = blocks[key]
        label = f"--- 第 {key} 段：{heading} ---" if heading else f"--- 第 {key} 段 ---"
        chunks.append(label + "\n" + "\n".join(lines))

    return "\n\n".join(chunks), len(blocks)


def _paragraph_contents(paragraph) -> list[tuple[str, str]]:
    """('text'|'image', value) for what is anchored inside one paragraph.

    Covers the two things that hide below paragraph level and that
    `Paragraph.text` cannot see:

      **Text boxes.** Their words live in a `w:txbxContent` nested inside a
      run, so a callout holding the price and the on-sale date — exactly the
      consumer-facing facts this system exists to capture — was being dropped
      without trace.

      **Floating pictures.** Word writes these twice, a modern `mc:Choice` and
      a legacy `mc:Fallback` describing the same object, so only one branch is
      read. Reading both took every floating picture twice over.
    """
    from docx.oxml.ns import qn

    blip_tag, embed_attr = qn("a:blip"), qn("r:embed")
    rid_attr, para_tag, text_tag = qn("r:id"), qn("w:p"), qn("w:t")
    txbx_tag = qn("w:txbxContent")

    out: list[tuple[str, str]] = []

    def scan(node, want_text: bool) -> None:
        tag = node.tag
        if tag == MC_ALTERNATE:
            branch = node.find(MC_CHOICE)
            if branch is None:
                branch = node.find(MC_FALLBACK)
            if branch is not None:
                for child in branch:
                    scan(child, want_text)
            return
        if tag == blip_tag:
            rid = node.get(embed_attr)
            if rid:
                out.append(("image", rid))
            return
        if tag == VML_IMAGEDATA:
            rid = node.get(rid_attr)
            if rid:
                out.append(("image", rid))
            return
        if tag == txbx_tag:
            if want_text:
                for para in node.iter(para_tag):
                    text = "".join(t.text or "" for t in para.iter(text_tag)).strip()
                    if text:
                        out.append(("text", text))
            # Keep descending for pictures inside the box, but not for its
            # words again — the `iter` above already took every nested one.
            for child in node:
                scan(child, want_text=False)
            return
        for child in node:
            scan(child, want_text)

    scan(paragraph._p, want_text=True)
    return out


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


def _walk(container):
    """Yield ('heading'|'text'|'image', value) over `container`, in document order."""
    for item in _iter_block_items(container):
        if hasattr(item, "rows"):                       # a table
            yield from _walk_table(item)
            continue

        text = item.text.strip()
        if text and _is_heading(item):
            yield ("heading", text)
            continue
        if text:
            yield ("text", text)
        yield from _paragraph_contents(item)


def _walk_table(table):
    """A table's rows as joined lines, plus any pictures inside its cells.

    The text keeps its row shape — `售價 | NT$4,280` reads as one fact, which
    is how `ppt_extract` renders deck tables too — but the cells are also
    descended into for pictures, because a proposal's product shots are as
    likely to sit in a spec table as in the body, and a picture that never
    surfaces is indistinguishable from a document that had none.
    """
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:
            # A merged cell is returned once per grid position it spans;
            # taking it each time would repeat its text and its pictures.
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(cell)

        texts = [c.text.strip() for c in cells]
        if any(texts):
            yield ("text", " | ".join(texts))
        for cell in cells:
            for kind, value in _walk(cell):
                if kind == "image":
                    yield ("image", value)


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
    # Flat and materialised, so a picture can see both the text before it and
    # the text after — the latter is not yet known while walking forward.
    items = _annotated(document)

    def nearby_for(position: int) -> str:
        """Nearest text before, then after — whichever exists."""
        before = next((items[j][3] for j in range(position - 1, -1, -1)
                       if items[j][2] != "image"), "")
        after = next((items[j][3] for j in range(position + 1, len(items))
                      if items[j][2] != "image"), "")
        joined = "\n".join(t for t in (before, after) if t)
        return joined[:max_nearby_chars]

    out: list[dict] = []
    for i, (block, heading, kind, value) in enumerate(items):
        if kind != "image":
            continue
        found = _blob(part, value)
        if found is None:
            continue                                    # linked, or not an image
        blob, ext = found
        out.append({
            "slide_index": block,
            "left": None, "top": None, "width": None, "height": None,
            "blob": blob,
            "ext": ext,
            "slide_heading": heading[:200],
            "nearby_text": nearby_for(i),
        })

    return out
