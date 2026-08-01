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


def _inline(old: str, new: str) -> list[dict]:
    """Character-level segments within one edited paragraph.

    Characters, not words: Chinese has no spaces to split on, and a word
    tokeniser would be a dependency and a guess. Runs of unchanged text stay
    unmarked, so a paragraph that gained three characters shows three
    characters, not itself twice.
    """
    matcher = SequenceMatcher(a=old, b=new, autojunk=False)
    segments: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append({"text": old[i1:i2], "mark": "same"})
            continue
        if i2 > i1:
            segments.append({"text": old[i1:i2], "mark": "del"})
        if j2 > j1:
            segments.append({"text": new[j1:j2], "mark": "add"})
    return segments


def segments_html(segments: list[dict]):
    """One edited block as HTML, with only the changed runs marked."""
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
    return mark_safe("".join(out))  # noqa: S308 -每段都經過 escape


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


def mark_paragraphs(old: str, new: str) -> list[dict]:
    """Paragraph-level marks — `same`, `add`, `del` — for the comparison view.

    Picture tokens are dropped here rather than rendered. The comparison is
    about what the copy says; `[[img:2]]` is a position marker, and showing it
    raw in a screen meant for reading is the sort of thing that makes people
    think the draft is broken.
    """
    from studio.templatetags.mdformat import IMG_TOKEN

    def blocks(text: str) -> list[str]:
        text = IMG_TOKEN.sub("", text or "")
        out: list[str] = []
        for chunk in BLOCK_SPLIT.split(text.strip()):
            out.extend(_units(chunk))
        return out

    before, after = blocks(old), blocks(new)
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    out: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend({"text": t, "mark": "same"} for t in before[i1:i2])
            continue

        gone, added = before[i1:i2], after[j1:j2]
        # Pair them off in order. A paragraph that was merely edited is shown
        # once, with the edit inside it; one that was replaced outright is shown
        # as the two paragraphs it really is.
        for left, right in zip(gone, added):
            if SequenceMatcher(a=left, b=right, autojunk=False).ratio() >= SIMILAR_ENOUGH:
                out.append(_edited_block(left, right))
            else:
                out.append({"text": left, "mark": "del"})
                out.append({"text": right, "mark": "add"})
        out.extend({"text": t, "mark": "del"} for t in gone[len(added):])
        out.extend({"text": t, "mark": "add"} for t in added[len(gone):])
    return out
