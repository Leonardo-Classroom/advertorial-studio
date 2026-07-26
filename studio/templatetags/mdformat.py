"""Render generated drafts.

Two filters:

`markdown` renders the text as-is — used wherever the raw structure matters.

`article` re-lays the draft out as the page it is meant to become. The model
emits a delivery document (## FB貼文文案 / ## 文章標題 / ## 內文 / ## Hashtag /
## 待確認), which is the right shape for handing to an editor but the wrong
shape for judging. Reading "## 文章標題" above the headline tells you nothing
about whether the headline works; seeing it set as a headline does. So the
article parts are promoted — title to an h1, body unlabelled, hashtags to a tag
row — with the Facebook copy following it, still labelled, since it is a
separate deliverable.

`article_notes` returns the 待確認 block on its own. It is not page content at
all — it is a note to whoever briefs the client — so the template puts it in its
own card rather than tucking it under the article, where it sat oddly.

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

# Sections the format spec asks for exactly once. Only the body may accumulate:
# it is genuinely many chunks (prose plus every 小標), while a second title or a
# second Facebook block is the model repeating itself, not extra content.
SINGLE_SLOTS = frozenset({"title", "fb", "tags", "todo"})

# Where the draft asked for one of the deck's own pictures. Deliberately not
# markdown image syntax: `![](…)` would let model output name an arbitrary URL,
# and the whole point is that only approved, locally-stored deck images can
# appear. A number indexes the approved roster and nothing else can.
IMG_TOKEN = re.compile(r"\[\[img:(\d+)\]\]", re.I)
# The same token after rendering, when it ended up alone in its own paragraph —
# the normal case, since the spec asks for it on its own line. Matched so the
# <figure> replaces the <p> instead of nesting inside it.
IMG_PARAGRAPH = re.compile(r"<p>\s*\[\[img:(\d+)\]\]\s*</p>", re.I)


@lru_cache(maxsize=1)
def _renderer():
    from markdown_it import MarkdownIt

    # `html=False` escapes any raw HTML in the source rather than passing it on.
    return MarkdownIt("commonmark", {"html": False, "linkify": False, "breaks": True})


# The reference advertorials flag missing material with a short starred line
# placed where it applies — "＊再請品牌提供需導連網址", "＊補品牌活動空景照" —
# rather than a list of open questions at the end. Those lines are notes to the
# editor, so they are marked as such instead of reading as body copy.
NOTE_LINE = re.compile(r"<p>([＊※*][^<]*)</p>")


def _render(text: str) -> str:
    if not text.strip():
        return ""
    html = _renderer().render(text.strip())
    return NOTE_LINE.sub(r'<p class="note">\1</p>', html)


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


def _figure(image) -> str:
    caption = image.display_caption()
    cap_html = f'<figcaption>{escape(caption)}</figcaption>' if caption else ""
    return (f'<figure class="post-figure">'
            f'<img src="{escape(image.file.url)}" alt="{escape(caption)}" loading="lazy">'
            f'{cap_html}</figure>')


def _place_images(html: str, brief) -> str:
    """Swap image tokens for the pictures they refer to.

    Runs on rendered HTML rather than on the markdown source so the surrounding
    paragraph structure is already settled and a figure can replace its own
    paragraph cleanly.

    A token with no matching approved picture is dropped, not shown: the number
    came from a language model, and leaving `[[img:9]]` visible in a draft
    someone is about to hand to a client is worse than quietly omitting an image
    that was never approved.
    """
    if brief is None:
        return IMG_TOKEN.sub("", html)

    images = list(brief.usable_images())
    if not images:
        return IMG_TOKEN.sub("", html)

    by_number = {i: image for i, image in enumerate(images, 1)}

    def swap(match):
        image = by_number.get(int(match.group(1)))
        return _figure(image) if image else ""

    return IMG_TOKEN.sub(swap, IMG_PARAGRAPH.sub(swap, html))


@register.filter(name="article")
def article(text: str, brief=None) -> str:
    """Lay a draft out as the published page it is meant to become.

    Pass the brief — `{{ run.output|article:run.brief }}` — to have the deck's
    approved pictures placed where the draft asked for them. Without it the
    tokens are stripped, so an older call site degrades to the text-only layout
    it already produced rather than leaking markup at the reader.
    """
    if not text:
        return ""

    parts: dict[str, list[str]] = {}
    for heading, body in _split_sections(str(text)):
        kind = _classify(heading)
        if kind == "other":
            # An unexpected section is still article content; keep its heading.
            parts.setdefault("body", []).append(f"### {heading}\n\n{body}")
        elif kind in SINGLE_SLOTS and kind in parts:
            # The deck asks for one of each; occasionally the model writes one
            # twice — a run was observed repeating the whole Facebook block as a
            # ### sub-heading inside 內文, which then rendered as two FB copies
            # stacked in the same aside. The first is the one in its specified
            # position, so later repeats are dropped. Nothing is lost that the
            # reader needs: 純文字 still shows the draft exactly as written.
            continue
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
        html.append(f'<div class="post-body">{_place_images(_render(body), brief)}</div>')

    tags = joined("tags")
    if tags:
        found = re.findall(r"#[^\s#、,，]+", tags)
        if found:
            html.append('<p class="post-tags">'
                        + "".join(f'<span class="tag">{escape(t)}</span>' for t in found)
                        + "</p>")
    html.append("</article>")

    content = joined("fb")
    if content:
        # Tokens do not belong outside the body, but a stray one must not
        # reach the reader as literal markup either.
        html.append('<section class="aside"><h4>FB 貼文文案</h4>'
                    f'{IMG_TOKEN.sub("", _render(content))}</section>')

    return mark_safe("".join(html))  # noqa: S308 - html disabled in the renderer


EMPTY_NOTES = {"無", "無。", "（無）", "(無)", "none", "n/a", "-"}


@register.filter(name="article_notes")
def article_notes(text: str) -> str:
    """Just the 待確認 block, so the template can place it outside the article.

    It is not page content — it is a note to whoever briefs the client — so
    tucking it under the copy read oddly. Returns empty when the model says
    there is nothing outstanding, since a card announcing "無" is noise.
    """
    if not text:
        return ""
    body = "\n\n".join(
        b for heading, b in _split_sections(str(text))
        if _classify(heading) == "todo" and b.strip()
    ).strip()
    if not body or body.strip().lower() in EMPTY_NOTES:
        return ""
    return mark_safe(IMG_TOKEN.sub("", _render(body)))  # noqa: S308
