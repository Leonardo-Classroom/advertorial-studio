"""Render generated drafts as Markdown.

Drafts come back in Markdown (`## FB貼文文案`, `### 小標`, bullet lists), so
showing them as preformatted text makes the operator read markup instead of
copy. Rendering them lets the structure be judged the way a reader would see it.

Raw HTML stays disabled in the renderer: the text is model output, and although
it is our own pipeline producing it, passing model-generated markup straight
into the page is exactly the habit that turns a prompt injection into stored
XSS. Nothing in an advertorial needs raw HTML anyway.
"""
from __future__ import annotations

from functools import lru_cache

from django import template
from django.utils.safestring import mark_safe

register = template.Library()


@lru_cache(maxsize=1)
def _renderer():
    from markdown_it import MarkdownIt

    # `html=False` escapes any raw HTML in the source rather than passing it on.
    return MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": True})


@register.filter(name="markdown")
def markdown(text: str) -> str:
    if not text:
        return ""
    return mark_safe(_renderer().render(str(text)))  # noqa: S308 - html disabled above
