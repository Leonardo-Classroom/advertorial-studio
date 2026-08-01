"""Present extracted facts as prose a non-programmer can check.

The portal used to show the fact dict as raw JSON in an editable textarea. That
is the right tool for whoever debugs the extractor and the wrong one for the
person who knows whether the KOL's name is spelled correctly — they were being
asked to proofread a data structure.

So the same dict is rendered here as labelled sections, and only the parts that
reach a draft are shown. `internal_only` is dropped outright: budgets, KPIs and
media plans are never written into copy, so asking someone to review them wastes
the attention that should go on the product name. `uncertain` is dropped too —
it is rendered separately, as the questions it is.

Unknown keys are not silently discarded. If the extractor starts emitting a
field this list has never heard of, it appears under its own raw name rather
than disappearing, because a fact the user cannot see is a fact they cannot
correct.
"""
from __future__ import annotations

from difflib import SequenceMatcher

# Ordered the way someone reads a brief: who it is for, what is being sold,
# what may be said about it, then the constraints.
SECTIONS: list[tuple[str, str]] = [
    ("brand", "品牌"),
    ("campaign", "檔期／活動"),
    ("product", "產品"),
    ("product_details", "產品特徵"),
    ("slogan", "主標語"),
    ("key_message", "想傳達的重點"),
    ("kol", "合作人選"),
    ("kol_content", "KOL 演繹內容"),
    ("activation", "實體活動／體驗"),
    ("consumer_info", "消費者資訊（售價、開賣日、通路）"),
    ("target_audience", "目標受眾"),
    ("deliverables", "交付項目"),
    ("publish_schedule", "刊登時間"),
    ("mandatory_terms", "必須出現的字詞"),
    ("forbidden_terms", "不能出現的字詞"),
]

HIDDEN_KEYS = {"internal_only", "uncertain"}

LABELS = dict(SECTIONS)


def as_lines(value) -> list[str]:
    """Flatten one fact value into display lines, dropping the empties."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(as_lines(item))
        return out
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            for line in as_lines(item):
                out.append(f"{LABELS.get(key, key)}：{line}")
        return out
    return [str(value)]


def to_sections(facts: dict) -> list[dict]:
    """Labelled, non-empty sections in reading order, internals removed."""
    facts = facts or {}
    out: list[dict] = []

    for key, label in SECTIONS:
        lines = as_lines(facts.get(key))
        if lines:
            out.append({"key": key, "label": label, "lines": lines})

    known = {key for key, _ in SECTIONS} | HIDDEN_KEYS
    for key, value in facts.items():
        if key in known:
            continue
        lines = as_lines(value)
        if lines:
            out.append({"key": key, "label": key, "lines": lines})

    return out


def uncertain_items(facts: dict) -> list[str]:
    raw = (facts or {}).get("uncertain") or []
    return [str(u).strip() for u in raw if str(u).strip()]


def _mark_lines(old_lines: list[str], new_lines: list[str]) -> list[dict]:
    """Line-level marks for one field, the way a diff viewer shows them.

    Matched by whole line rather than by character. These values are names,
    product codes and single sentences: highlighting three changed characters
    inside 「合作人選：陳○○」 hides the thing the reader is checking, which is
    whether the name is now right.
    """
    matcher = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    out: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend({"text": t, "mark": "same"} for t in old_lines[i1:i2])
            continue
        out.extend({"text": t, "mark": "del"} for t in old_lines[i1:i2])
        out.extend({"text": t, "mark": "add"} for t in new_lines[j1:j2])
    return out


def to_sections_marked(old: dict, new: dict) -> list[dict]:
    """`to_sections` for the newer version, with what changed marked up.

    Unchanged sections stay in place and unmarked. A view that showed only the
    changes would answer "what moved" while losing "what does this brief say",
    and the reason to look at a diff here is to judge one against the other.
    """
    old, new = old or {}, new or {}
    known = {key for key, _ in SECTIONS} | HIDDEN_KEYS
    ordered = [key for key, _ in SECTIONS]
    ordered += [k for k in list(new) + list(old) if k not in known]

    out: list[dict] = []
    seen: set[str] = set()
    for key in ordered:
        if key in seen or key in HIDDEN_KEYS:
            continue
        seen.add(key)
        lines = _mark_lines(as_lines(old.get(key)), as_lines(new.get(key)))
        if lines:
            out.append({
                "key": key,
                "label": LABELS.get(key, key),
                "lines": lines,
                "changed": any(line["mark"] != "same" for line in lines),
            })
    return out


def uncertain_marked(old: dict, new: dict) -> list[dict]:
    """The open questions, with the ones this version answered struck through."""
    return _mark_lines(uncertain_items(old), uncertain_items(new))


def diff(old: dict, new: dict) -> list[dict]:
    """What a merge changed, field by field, for the confirmation screen.

    Every top-level key is compared, `internal_only` included. The user is not
    asked to care about the internal block, but a merge that quietly rewrote it
    would still be a merge that did more than it was told, and this screen only
    means something if it shows everything that moved.
    """
    old, new = old or {}, new or {}
    rows: list[dict] = []

    ordered = [key for key, _ in SECTIONS]
    ordered += [k for k in list(old) + list(new) if k not in ordered]

    seen: set[str] = set()
    for key in ordered:
        if key in seen:
            continue
        seen.add(key)
        before, after = old.get(key), new.get(key)
        if before == after:
            continue
        label = LABELS.get(key, key)
        if key == "uncertain":
            label = "待確認事項"
        elif key == "internal_only":
            label = "內部資訊（不會寫進稿子）"
        rows.append({
            "key": key,
            "label": label,
            "before": as_lines(before),
            "after": as_lines(after),
        })
    return rows
