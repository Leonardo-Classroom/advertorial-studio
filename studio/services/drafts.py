"""Draft versions: creating them, and showing what changed between two.

The copy gets the same treatment as the facts. Every arrival is a version —
the one the pipeline produced, the ones the user typed, the ones they asked the
model to rewrite — and none of them replaces another. A version records what it
was built from, so "改一下第二版" is a thing the interface can actually do
rather than a thing the user has to reconstruct by hand.

The comparison is by paragraph rather than by word. Copy is judged in blocks:
what matters is that the second paragraph now says something different, not
that three characters inside it moved. A word-level diff of rewritten prose
lights up almost everything and tells you nothing.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

BLOCK_SPLIT = re.compile(r"\n\s*\n")
HEADING_LINE = re.compile(r"^#{1,3}[ \t]+\S")


def _units(block: str) -> list[str]:
    """Split a block so a heading is never glued to the text under it.

    The model writes `## 文章標題` with a blank line after it; the edit form
    rebuilds it the same way. But some drafts arrive with the heading on the
    line directly above its copy, and then a version that has been through the
    editor differs from one that has not by a line break — which the comparison
    would report as the heading having been added, next to the same heading
    marked as deleted. Splitting both sides the same way removes the phantom.
    """
    units, buffer = [], []
    for line in block.splitlines():
        if HEADING_LINE.match(line.strip()):
            if buffer:
                units.append("\n".join(buffer).strip())
                buffer = []
            units.append(line.strip())
        else:
            buffer.append(line)
    if buffer:
        units.append("\n".join(buffer).strip())
    return [u for u in units if u]


def first_text(run) -> str:
    """What the pipeline handed over, folding its own rewrites into one draft.

    The rewrite budget makes the generator revise its own work before handing
    anything back. That is the generator finishing the job, not a second draft:
    showing both invites the user to choose between two texts they never asked
    to compare.
    """
    last_auto = (run.revisions.filter(source="auto", accepted=True)
                 .order_by("-round").first())
    return last_auto.output if last_auto else run.output


def versions(run) -> list:
    """Newest first, matching how the dropdowns are read."""
    return list(run.draft_versions.all())


def latest(run):
    return run.draft_versions.first()


def add_version(run, text: str, source: str = "manual", user_input: str = "",
                parent=None):
    """Record a new version. A save that changes nothing is not a version."""
    from studio.models import DraftVersion

    text = (text or "").strip()
    last = run.draft_versions.first()
    if last is not None and last.text.strip() == text:
        return last
    return DraftVersion.objects.create(
        run=run,
        version=(last.version + 1) if last else 1,
        text=text,
        source=source,
        user_input=user_input,
        parent=parent or last,
    )


def ensure_first_version(run):
    """Give a finished run its v1, if it has not got one.

    Called when a run completes, and again from the pages that display drafts so
    that runs finished before versioning still show something.
    """
    if run.draft_versions.exists():
        return run.draft_versions.first()
    text = first_text(run)
    if not text.strip():
        return None
    return add_version(run, text, source="generate")


# How alike two paragraphs must be before the comparison shows the change
# inside one block rather than as a deletion beside an addition. Tuned by what
# reads better: appending a sentence, fixing a name, changing a number all land
# well above this; a paragraph rewritten from scratch lands below it, and there
# the honest display is two paragraphs side by side.
SIMILAR_ENOUGH = 0.5


# A diff unit: a whole `**bold**`/`*italic*` span where the source actually
# pairs it up, or one character. Diffing raw characters can split a pair so
# only one side lands in a segment — rendering that segment alone then shows
# a stray asterisk with no partner, or (worse) pairs it with an unrelated
# marker elsewhere and mangles the tags. Keeping an intact span as a single
# unit means the matcher can only mark it whole: a segment either carries the
# complete `**text**` or none of it, never half.
ATOM = re.compile(r"\*\*.+?\*\*|\*.+?\*|.", re.S)


def _atoms(text: str) -> list[str]:
    return ATOM.findall(text)


def _inline(old: str, new: str) -> list[dict]:
    """Segments within one edited paragraph, diffed by atom (see ATOM).

    Character-level rather than word-level: Chinese has no spaces to split
    on, and a word tokeniser would be a dependency and a guess. Runs of
    unchanged text stay unmarked, so a paragraph that gained three characters
    shows three characters, not itself twice. The cost of keeping emphasis
    spans whole is coarser granularity for an edit made *inside* one — the
    whole span swaps rather than showing just the changed character — which
    reads fine since these spans are short phrases, not paragraphs.
    """
    old_atoms, new_atoms = _atoms(old), _atoms(new)
    matcher = SequenceMatcher(a=old_atoms, b=new_atoms, autojunk=False)
    segments: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append({"text": "".join(old_atoms[i1:i2]), "mark": "same"})
            continue
        if i2 > i1:
            segments.append({"text": "".join(old_atoms[i1:i2]), "mark": "del"})
        if j2 > j1:
            segments.append({"text": "".join(new_atoms[j1:j2]), "mark": "add"})
    return segments


def segments_html(segments: list[dict]):
    """One edited block as HTML, with only the changed runs marked.

    Each segment is rendered through the same `**`/`*` → bold/italic renderer
    the edit boxes use, so a diff reads with the same formatting as the
    draft itself rather than showing the raw markdown symbols. This is safe
    because segments are built from whole emphasis spans or single plain
    characters (see ATOM) — a segment never contains half of a `**pair**`,
    so there is nothing for the renderer to mis-pair.
    """
    from django.utils.safestring import mark_safe

    from studio.services.draft_edit import to_html

    out = []
    for segment in segments:
        body = to_html(segment["text"])
        if segment["mark"] == "add":
            out.append(f"<ins>{body}</ins>")
        elif segment["mark"] == "del":
            out.append(f"<del>{body}</del>")
        else:
            out.append(str(body))
    return mark_safe("".join(out))  # noqa: S308 - to_html escapes internally


def _edited_block(old: str, new: str) -> dict:
    """One entry showing `new` with what changed inside it marked."""
    heading = bool(HEADING_LINE.match(new.strip()))
    segments = _inline(old, new)
    if heading:
        # Strip the marks from both sides so the diff is over the words, not
        # over the `##` that is identical anyway.
        bare_old = old.lstrip("# ").strip()
        bare_new = new.lstrip("# ").strip()
        segments = _inline(bare_old, bare_new)
    return {"mark": "edit", "heading": heading, "segments": segments,
            "html": segments_html(segments), "text": new}


IMG_UNIT = re.compile(r"^\s*\[\[img:(\d+)(?:\|(.*?))?\]\]\s*$", re.I)
# A unit made only of picture tokens. The model writes two on consecutive
# lines and markdown keeps them in one block, so without splitting these the
# pair stays a single unit, matches nothing, and renders as raw `[[img:2]]`
# text — the exact thing dropping tokens used to prevent.
IMG_ONLY_BLOCK = re.compile(r"^\s*(?:\[\[img:\d+(?:\|.*?)?\]\]\s*)+$", re.I)


def _as_image_entry(text: str, images: dict) -> dict | None:
    """A diff unit that is exactly one picture token, resolved, or None."""
    match = IMG_UNIT.match(text or "")
    if not match:
        return None
    image = images.get(int(match.group(1)))
    if image is None:
        return None
    carried = match.group(2)
    return {"image": image,
            "caption": carried if carried is not None else image.display_caption()}


def mark_paragraphs(old: str, new: str, brief=None) -> list[dict]:
    """Paragraph-level marks — `same`, `add`, `del` — for the comparison view.

    Picture tokens used to be stripped here, on the grounds that the comparison
    is about what the copy says and a raw `[[img:2]]` on a reading screen looks
    like breakage. That stopped being right once captions moved *into* the
    token: a caption edit is a copy edit, and stripping the token hid the one
    change the reader most needs to check. Tokens are now kept as units of
    their own and handed to the template as the picture plus its caption, so a
    changed caption shows up marked like any other edited line.

    Without a `brief` there is nothing to resolve a number against, so the old
    behaviour stands and tokens are dropped rather than shown raw.
    """
    from studio.templatetags.mdformat import IMG_TOKEN

    images = {}
    if brief is not None:
        images = {i: image for i, image in enumerate(brief.usable_images(), 1)}

    def blocks(text: str) -> list[str]:
        text = text or ""
        if not images:
            text = IMG_TOKEN.sub("", text)
        out: list[str] = []
        for chunk in BLOCK_SPLIT.split(text.strip()):
            for unit in _units(chunk):
                if images and IMG_ONLY_BLOCK.match(unit):
                    # One token per unit, so each picture is diffed on its own
                    # and a caption change on the second of two does not mark
                    # both as edited.
                    out.extend(m.group(0) for m in IMG_TOKEN.finditer(unit))
                else:
                    out.append(unit)
        return out

    def entry(text: str, mark: str) -> dict:
        """One unit, as a picture if it is one and as prose otherwise."""
        picture = _as_image_entry(text, images)
        if picture:
            return {"text": text, "mark": mark, **picture}
        return {"text": text, "mark": mark}

    before, after = blocks(old), blocks(new)
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    out: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(entry(t, "same") for t in before[i1:i2])
            continue

        gone, added = before[i1:i2], after[j1:j2]
        # Pair them off in order. A paragraph that was merely edited is shown
        # once, with the edit inside it; one that was replaced outright is shown
        # as the two paragraphs it really is.
        for left, right in zip(gone, added):
            # Same picture, different caption: one entry showing the picture
            # once with the caption's own diff inside it. Falling through to
            # the prose path would diff the token syntax and print
            # "[[img:1|…]]" with half of it in red.
            was, now = _as_image_entry(left, images), _as_image_entry(right, images)
            if was and now and was["image"].pk == now["image"].pk:
                segments = _inline(was["caption"], now["caption"])
                out.append({"text": right, "mark": "edit", "image": now["image"],
                            "caption": now["caption"], "was_caption": was["caption"],
                            "caption_html": segments_html(segments)})
                continue
            if SequenceMatcher(a=left, b=right, autojunk=False).ratio() >= SIMILAR_ENOUGH:
                out.append(_edited_block(left, right))
            else:
                out.append(entry(left, "del"))
                out.append(entry(right, "add"))
        out.extend(entry(t, "del") for t in gone[len(added):])
        out.extend(entry(t, "add") for t in added[len(gone):])
    return out
