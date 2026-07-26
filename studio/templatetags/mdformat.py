"""Render generated drafts.

Two filters:

`markdown` renders the text as-is — used wherever the raw structure matters.

`article` re-lays the draft out as the page it is meant to become. The model
emits a delivery document (## FB貼文文案 / ## 文章標題 / ## 內文 / ## Hashtag /
## 待確認), which is the right shape for handing to an editor but the wrong
shape for judging. Reading "## 文章標題" above the headline tells you nothing
about whether the headline works; seeing it set as a headline does. So the
article parts are promoted — title to an h1, body unlabelled, hashtags to a tag
row — and the two things that are not the article (the Facebook copy and the
open questions) follow it, still labelled, because they are separate
deliverables rather than page furniture.

Raw HTML stays disabled in the renderer: the text is model output, and piping
model-generated markup straight into a page is how a prompt injection becomes
stored XSS. Nothing in an advertorial needs raw HTML.
"""
from __future__ import annotations

import re
from functools import lru_cache

from django import template
from django.utils.html import escape
from django.utils.safestring import mark_safe

register = template.Library()

# The whitespace after the hashes is required, not optional. Markdown demands
# it, and without it a hashtag line — "#NewBalance #RunYourWay ..." — parses as
# a heading, which swallowed the tag row and invented a phantom section.
SECTION = re.compile(r"^#{1,3}[ \t]+(.+?)\s*$", re.M)

# Matched loosely: the model reproduces the spec's headings faithfully but not
# always its punctuation or spacing.
TITLE_KEYS = ("文章標題", "標題")
BODY_KEYS = ("內文", "正文", "文章")
TAG_KEYS = ("hashtag", "標籤")
FB_KEYS = ("fb貼文", "fb 貼文", "facebook", "貼文文案")
TODO_KEYS = ("待確認", "待補", "待提供")


@lru_cache(maxsize=1)
def _renderer():
    from markdown_it import MarkdownIt

    # `html=False` escapes any raw HTML in the source rather than passing it on.
    return MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": True})


def _render(text: str) -> str:
    return _renderer().render(text.strip()) if text.strip() else ""


@register.filter(name="markdown")
def markdown(text: str) -> str:
    if not text:
        return ""
    return mark_safe(_render(str(text)))  # noqa: S308 - html disabled above


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings into (heading, body) pairs.

    Content before the first heading is kept under an empty heading so a draft
    that ignores the format is still displayed rather than silently dropped.
    """
    matches = list(SECTION.finditer(text))
    if not matches:
        return [("", text)]

    out: list[tuple[str, str]] = []
    lead = text[: matches[0].start()].strip()
    if lead:
        out.append(("", lead))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), text[m.end():end].strip()))
    return out


def _classify(heading: str) -> str:
    h = heading.lower().replace(" ", "")
    for keys, name in ((TITLE_KEYS, "title"), (FB_KEYS, "fb"), (TAG_KEYS, "tags"),
                       (TODO_KEYS, "todo"), (BODY_KEYS, "body")):
        if any(k.lower().replace(" ", "") in h for k in keys):
            return name
    return "body" if not heading else "other"


@register.filter(name="article")
def article(text: str) -> str:
    """Lay a draft out as the published page it is meant to become."""
    if not text:
        return ""

    parts: dict[str, list[str]] = {}
    for heading, body in _split_sections(str(text)):
        kind = _classify(heading)
        if kind == "other":
            # An unexpected section is still article content; keep its heading.
            parts.setdefault("body", []).append(f"### {heading}\n\n{body}")
        else:
            parts.setdefault(kind, []).append(body)

    def joined(key: str) -> str:
        return "\n\n".join(parts.get(key, [])).strip()

    html: list[str] = ['<article class="post">']

    title = joined("title")
    if title:
        # A title arriving as a list item or with stray marks should still read
        # as one line of prose.
        clean = re.sub(r"^[\s*\-–—#>]+", "", title.splitlines()[0]).strip()
        html.append(f'<h1 class="post-title">{escape(clean)}</h1>')

    body = joined("body")
    if body:
        html.append(f'<div class="post-body">{_render(body)}</div>')

    tags = joined("tags")
    if tags:
        found = re.findall(r"#[^\s#、,，]+", tags)
        if found:
            html.append('<p class="post-tags">'
                        + "".join(f'<span class="tag">{escape(t)}</span>' for t in found)
                        + "</p>")
    html.append("</article>")

    for key, label in (("fb", "FB 貼文文案"), ("todo", "待確認")):
        content = joined(key)
        if content:
            html.append(f'<section class="aside"><h4>{label}</h4>{_render(content)}</section>')

    return mark_safe("".join(html))  # noqa: S308 - html disabled in the renderer
