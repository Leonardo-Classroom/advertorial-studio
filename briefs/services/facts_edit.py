"""Take extracted facts apart into editable fields, and put them back.

The natural-language correction box beside this asks a model to fold a sentence
into the fact dict. That is the right tool when you can describe the change
("開賣日是 9/15") and the wrong one when you just want to fix a typo in a KOL's
name: it spends a model call, takes seconds, and can quietly reword a field you
never mentioned. Typing over the value directly cannot do any of that.

Shapes are preserved, not normalised. In real briefs every visible field is
either a string or a list of strings, and which one it is varies per brief —
`target_audience` arrives both ways. A field that came in as a string is
written back as a string, so hand-editing a brief never changes the shape the
extractor produced and the prompt builder reads.

`internal_only` and `uncertain` are carried through untouched. They are hidden
from this screen on purpose (see `facts_text`), and a round-trip that dropped
whatever it did not display would delete them the first time anyone saved.
"""
from __future__ import annotations

from briefs.services.facts_text import HIDDEN_KEYS, LABELS, SECTIONS


def _kind(value) -> str:
    return "list" if isinstance(value, list) else "text"


def to_fields(facts: dict) -> list[dict]:
    """Editable fields in the same order the read-only view shows them.

    Empty fields are included, unlike the read-only view which drops them: a
    brief with no 主標語 is exactly the case where someone needs a box to type
    one into. Unknown keys appear under their raw name for the same reason the
    read-only view shows them — a fact nobody can see is a fact nobody can fix.
    """
    facts = facts or {}
    ordered = [key for key, _ in SECTIONS]
    ordered += [k for k in facts if k not in set(ordered) | HIDDEN_KEYS]

    out: list[dict] = []
    for key in ordered:
        value = facts.get(key)
        kind = _kind(value)
        field = {
            "key": key,
            "label": LABELS.get(key, key),
            "kind": kind,
            "name": f"f_{key}",
        }
        if kind == "list":
            items = [str(v).strip() for v in value if str(v).strip()]
            field["items"] = [{"name": f"f_{key}_{i}", "value": v}
                              for i, v in enumerate(items)]
            # One spare box, so adding a line needs no separate "add" control.
            field["blank"] = f"f_{key}_{len(items)}"
        else:
            field["value"] = "" if value is None else str(value)
        out.append(field)
    return out


def from_fields(facts: dict, posted) -> dict:
    """Rebuild the fact dict from what came back.

    Driven by the original dict rather than by the POST data, the same way the
    draft editor is: a field missing from the submission keeps its old value
    instead of being blanked, so a truncated or tampered-with form cannot
    quietly empty half a brief.
    """
    facts = facts or {}
    out = dict(facts)

    for field in to_fields(facts):
        key, name = field["key"], field["name"]
        if field["kind"] == "list":
            # Collected by prefix and sorted, not walked from 0 until a number
            # is missing. Removing a line takes its input out of the form
            # entirely, so the numbering arrives with holes in it — and the
            # walking version stopped at the first hole, silently dropping
            # every line after the one that was deleted.
            #
            # `f_kol_` is also a prefix of `f_kol_content_0`; the digit test is
            # what keeps one field from swallowing another's boxes.
            prefix = f"f_{key}_"
            found = []
            for name in posted:
                if not name.startswith(prefix):
                    continue
                index = name[len(prefix):]
                if index.isdigit():
                    found.append((int(index), posted.get(name)))
            if not found:                  # field absent from the submission
                continue
            values = []
            for _, item in sorted(found):
                item = " ".join(str(item).split())
                if item:                   # a blank box is how a line is cleared
                    values.append(item)
            out[key] = values
        else:
            value = posted.get(name)
            if value is None:
                continue
            out[key] = " ".join(str(value).split())

    # A key that only ever held an empty value adds nothing but noise to the
    # diff between versions; drop it rather than storing "" forever.
    for key in [k for k, v in out.items() if k not in HIDDEN_KEYS and v in ("", [])]:
        if not facts.get(key):
            del out[key]
    return out
