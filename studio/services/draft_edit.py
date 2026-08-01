"""Take a draft apart into editable pieces, and put it back together.

The article view shows a draft as the page it will become. Editing it as one
big blob of markdown would undo that: the reason to reread your own copy in
page form is that a paragraph reads differently there, and dropping the person
into a textarea full of `## 內文` puts them back in the delivery document.

So the draft is split at its own boundaries — sections at the `##` headings the
format spec asks for, and the body further into blocks at blank lines — and each
piece becomes one field. Rebuilding is the exact inverse: the original heading
text is carried through untouched, so a draft that is opened and saved without
changes comes back byte for byte, and 純文字 does not silently rewrite itself
into a tidier version of what the model actually wrote.

No markdown reaches the boxes. The person checking a draft writes copy, not
markup: a bullet list has to arrive as one box per bullet, and a 小標 as a box
with the words in it, because `- ` and `### ` in a text field are three
characters someone will eventually delete by accident and never notice. The
markers are stripped on the way out and put back on the way in.

Image tokens are the exception. `[[img:3]]` is a reference into the approved
picture roster, not prose, and letting someone hand-edit the number is a way to
publish a picture nobody approved. Those blocks travel through the form as
hidden fields and are shown as the picture they stand for.
"""
from __future__ import annotations

import re

from django.utils.html import escape
from django.utils.safestring import mark_safe

from studio.templatetags.mdformat import IMG_TOKEN, SECTION

# Like the renderer's own section split, but keeping the `#` marks. The renderer
# only needs to know where a heading is; rebuilding needs to write it back at
# the level it was written, or the 小標 inside 內文 come back as `##` and the
# draft grows top-level sections it never had.
HEADING = re.compile(r"^(#{1,3})[ \t]+(.+?)\s*$", re.M)

# One block per paragraph: markdown's own separator, and close enough to how
# the article view lays them out that the fields line up with what is on screen.
BLOCK_SPLIT = re.compile(r"\n\s*\n")

# A block that is nothing but a picture reference.
ONLY_IMAGE = re.compile(r"^\s*\[\[img:(\d+)\]\]\s*$", re.I)

# One bullet or numbered item: indent, marker, text.
LIST_LINE = re.compile(r"^(\s*)([-*+]|\d+[.)])[ \t]+(.*)$")

# Which sections are worth editing by hand. Everything else in the draft is
# carried through unchanged rather than dropped.
LABELS = {
    "title": "文章標題",
    "body": "內文",
    "tags": "Hashtag",
    "fb": "FB 貼文文案",
    "todo": "待確認",
}


def _kind(heading: str) -> str:
    from studio.templatetags.mdformat import _classify

    return _classify(heading)


def _as_list(block: str) -> list[tuple[str, str, str]] | None:
    """(indent, marker, text) per line when the block is a list, else None."""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    items = [LIST_LINE.match(ln) for ln in lines]
    if not lines or not all(items):
        return None
    return [(m.group(1), m.group(2), m.group(3).strip()) for m in items]


def _sections(text: str) -> list[tuple[str, str, str]]:
    """(marker, heading, body) triples. Content before the first heading has
    an empty marker and heading, so a draft that ignores the format is still
    editable rather than silently dropped."""
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("", "", text)]

    out: list[tuple[str, str, str]] = []
    lead = text[: matches[0].start()].strip()
    if lead:
        out.append(("", "", lead))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), m.group(2), text[m.end():end].strip()))
    return out


# Inline bold. The only markup that survives into the boxes, and it survives as
# bold text rather than as asterisks — `**1000＋T500：俐落都會感**` on screen is
# four characters of noise plus a formatting instruction nobody asked to read.
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


def to_html(text: str):
    """One block as editable rich text: escaped, with `**` turned into bold."""
    return mark_safe(  # noqa: S308 - escaped first, only our own tags added
        BOLD.sub(r"<strong>\1</strong>", escape(text)).replace("\n", "<br>"))


def _clean(value: str) -> str:
    """Normalise one submitted box.

    Picture references are stripped rather than honoured: the roster is what
    approval means, and a token typed by hand bypasses it.
    """
    return IMG_TOKEN.sub("", (value or "").replace("\r\n", "\n")).strip()


def to_fields(text: str, brief=None) -> list[dict]:
    """Split a draft into form fields, in the order they appear.

    Each section becomes `{heading, kind, label, blocks}` where every block is
    `{name, value, image}` — `image` set when the block is a picture reference,
    which the template renders instead of offering a box to type in.
    """
    if not text or not text.strip():
        return []

    images = {}
    if brief is not None:
        images = {i: image for i, image in enumerate(brief.usable_images(), 1)}

    sections: list[dict] = []
    for index, (marker, heading, body) in enumerate(_sections(str(text))):
        kind = _kind(heading)
        blocks: list[dict] = []
        for j, raw in enumerate(BLOCK_SPLIT.split(body.strip())):
            block = raw.strip()
            if not block:
                continue
            match = ONLY_IMAGE.match(block)
            listed = None if match else _as_list(block)
            blocks.append({
                "name": f"s{index}b{j}",
                "value": block,
                "html": to_html(block),
                "image": images.get(int(match.group(1))) if match else None,
                # A bullet per box, with the dash left behind in the markup.
                "items": [{"name": f"s{index}b{j}i{k}", "value": text, "html": to_html(text)}
                          for k, (_, _, text) in enumerate(listed)] if listed else None,
            })
        sections.append({
            "index": index,
            "marker": marker,
            "heading": heading,
            "kind": kind,
            "label": LABELS.get(kind) or heading or "內容",
            # `##` names a delivery section — FB 貼文文案, 內文 — and renaming it
            # would break the format the whole pipeline reads. A `###` is a 小標
            # inside the article: that is copy, and copy is editable.
            "heading_name": f"s{index}h" if marker == "###" else "",
            # The title is one line; everything else can run long.
            "single_line": kind == "title",
            "blocks": blocks,
        })
    return sections


def from_fields(text: str, posted) -> str:
    """Rebuild the draft from what came back, keeping everything else as it was.

    Driven by the original text rather than by the POST data: a field that is
    missing keeps its old content instead of vanishing, so a truncated or
    tampered-with submission cannot quietly delete half a draft.
    """
    out: list[str] = []
    for index, (marker, heading, body) in enumerate(_sections(str(text or ""))):
        blocks: list[str] = []
        for j, raw in enumerate(BLOCK_SPLIT.split(body.strip())):
            block = raw.strip()
            if not block:
                continue
            if ONLY_IMAGE.match(block):
                blocks.append(block)          # not editable
                continue

            listed = _as_list(block)
            if listed is not None:
                lines = []
                for k, (indent, bullet, text) in enumerate(listed):
                    item = posted.get(f"s{index}b{j}i{k}")
                    item = text if item is None else _clean(item)
                    if item:                  # an emptied box removes that bullet
                        lines.append(f"{indent}{bullet} {item}")
                blocks.append("\n".join(lines) if lines else block)
                continue

            edited = posted.get(f"s{index}b{j}")
            blocks.append(block if edited is None else (_clean(edited) or block))

        joined = "\n\n".join(blocks).strip()
        if heading:
            edited_heading = posted.get(f"s{index}h") if marker == "###" else None
            if edited_heading is not None:
                heading = _clean(edited_heading).replace("\n", " ") or heading
            head = f"{marker} {heading}"
            out.append(f"{head}\n\n{joined}" if joined else head)
        elif joined:
            out.append(joined)
    return "\n\n".join(out).strip() + "\n"


def has_sections(text: str) -> bool:
    """Whether the draft uses the expected heading structure at all."""
    return bool(SECTION.search(str(text or "")))
