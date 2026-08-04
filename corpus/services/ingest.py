"""Parse and clean crawled articles into style-corpus records.

The crawler writes one .txt per article:

    標題：...
    記者：...
    新聞網：...
    日期：YYYY-MM-DD
    網址：...
    ----------------------------------------
    <body>

Cleaning matters more than usual here: these articles are used as *style*
exemplars, so site furniture (related-article links, image credits, embed
residue) would teach the model habits that belong to the CMS rather than to
the writer. The boilerplate rules below were derived by sampling the actual
corpus rather than guessed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

SEPARATOR = re.compile(r"^-{10,}$")

_HEADER_KEYS = {
    "標題": "title",
    "記者": "author",
    "新聞網": "outlet",
    "日期": "published_on",
    "網址": "url",
}

# Lines matching any of these are site furniture, not authored prose.
_DROP_LINE_PATTERNS = [
    re.compile(r"^[（(]?延伸閱讀[：:]"),     # COOL-STYLE related-article link,
                                             # sometimes wrapped in parentheses
    re.compile(r"^繼續閱讀$"),               # GQ related-article widget header
    re.compile(r"^image via\b", re.I),       # COOL-STYLE image credit
    re.compile(r"^photo\s*(by|from|credit)", re.I),
    # COOL-STYLE's own credit line, on a third of its articles. It escaped the
    # repeated-line scan because the source differs every time — only the
    # prefix repeats, which is why detection now looks at prefixes too.
    re.compile(r"^source\s*[/／]", re.I),
    re.compile(r"^在 Instagram 查看這則貼文$"),
    re.compile(r"分享的貼文$"),              # "X（@handle）分享的貼文"
    re.compile(r"^By\s+\S", re.I),           # byline of a linked article
    re.compile(r"Getty Images"),             # wire-service caption
    # Widened from `.{0,20}(粉絲團|IG|LINE|頻道|會員)`, which missed
    # COOL-STYLE's own sign-off — "追蹤 @cool_magazine_taiwan Instagram 帳號，
    # 觀看更多有趣的潮流、時事知識" — because the handle eats more than 20
    # characters and 帳號 was not among the endings. It closes 18% of that
    # outlet's articles, and duly turned up as its top "distinctive" phrase.
    re.compile(r"^(追蹤|訂閱|加入).{0,40}(粉絲團|IG|Instagram|帳號|LINE|頻道|會員)", re.I),
    # Widget captions the crawler kept as text: GQ's gallery and video buttons
    # (10% of its articles each), 中時新聞網's video-subscription line (6%).
    re.compile(r"^(VIEW GALLERY|WATCH|SHOP NOW|READ MORE)$", re.I),
    re.compile(r"^訂閱影音[：:]?$"),
    re.compile(r"^※?免責聲明[：:]"),         # 聯合新聞網, on financial pieces
    re.compile(r"^Powered by\s*$", re.I),    # 聯合新聞網 embedded-video credit,
    re.compile(r"^GliaStudios$"),            # which is split across two lines
]

# A "繼續閱讀" widget is header + linked title + byline; drop the title line too.
_CONTINUE_READING = re.compile(r"^繼續閱讀$")

# Everything from these lines to the end of the file is furniture. News sites
# append a paywall pitch and then a list of unrelated headlines, and it is the
# headlines that do the damage: they read as prose, so nothing downstream can
# tell they were never part of the article.
_TRUNCATE_PATTERNS = [
    re.compile(r"^這則內容有觸動你嗎[？?]$"),        # 中時新聞網 (96% of articles)
    re.compile(r"^將工商時報加入Google偏好來源$"),   # 工商時報 (100%)
    re.compile(r"^推薦閱讀$"),                       # 工商時報 link list (30%)
    re.compile(r"^你今年最好的選擇$"),               # 聯合新聞網 subscription pitch
    # 聯合新聞網's own section rails, each heading a run of unrelated headlines
    # that continues to the end of the file: 房地產推薦新聞 closes 12% of its
    # articles, 全球熱話題 another 6%. Their headlines are what put 敢買、偷走
    # and 身分證外洩 into that outlet's "distinctive vocabulary".
    re.compile(r"^.{0,6}推薦新聞$"),
    re.compile(r"^.{0,6}熱話題$"),
]

# A line that is nothing but hashtags is 中時新聞網's end-of-article metadata,
# sitting between the related-link list and a subscription advert. It is only
# treated as the end when it is *near* the end: a quarter of that outlet's
# articles carry one and every single one falls in the last three lines, but
# COOL-STYLE occasionally writes tags mid-body, and truncating there would
# throw away the rest of a real article.
_HASHTAG_ONLY = re.compile(r"^#[^\s#]+(\s+#[^\s#]+)*$")
_HASHTAG_TAIL_SLACK = 2      # non-blank lines that may follow and still count as the tail

# These head a block of related-article links that sits *inside* the body —
# 聯合新聞網 puts it above the first paragraph — so the marker and the block
# after it go, and the article resumes at the next blank line.
_DROP_BLOCK_PATTERNS = [
    re.compile(r"^【編輯推薦】$"),
]


@dataclass
class ParsedArticle:
    title: str
    author: str
    outlet: str
    url: str
    published_on: date | None
    body: str
    source_path: str
    dropped_lines: list[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            (self.title + "\n" + self.body).encode("utf-8")
        ).hexdigest()

    @property
    def char_count(self) -> int:
        return len(self.body)


def _parse_date(raw: str) -> date | None:
    raw = raw.strip()
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def clean_body(raw: str, outlet: str = "") -> tuple[str, list[str]]:
    """Strip site furniture. Returns (clean_text, dropped_lines).

    `outlet` lets a line that is nothing but the publication's own name go —
    a byline fragment left behind by the crawler. It is passed rather than
    pattern-matched per outlet so a new source needs no new rule.
    """
    kept: list[str] = []
    dropped: list[str] = []
    lines = raw.splitlines()
    self_name = outlet.strip()
    skip_next_nonblank = 0
    in_block = False
    block_had_content = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        truncate = stripped and any(p.match(stripped) for p in _TRUNCATE_PATTERNS)
        if not truncate and stripped and _HASHTAG_ONLY.match(stripped):
            following = sum(1 for x in lines[i + 1:] if x.strip())
            truncate = following <= _HASHTAG_TAIL_SLACK
        if truncate:
            dropped.extend(s for s in (x.strip() for x in lines[i:]) if s)
            break

        if in_block:
            # The marker is followed by a blank, then the links, then the blank
            # that ends the block — so only a blank *after* content closes it.
            if not stripped:
                if block_had_content:
                    in_block = False
                    kept.append("")
                continue
            dropped.append(stripped)
            block_had_content = True
            continue

        if not stripped:
            kept.append("")
            continue

        if skip_next_nonblank:
            skip_next_nonblank -= 1
            dropped.append(stripped)
            continue

        if self_name and stripped == self_name:
            dropped.append(stripped)
            continue

        if any(p.match(stripped) for p in _DROP_BLOCK_PATTERNS):
            dropped.append(stripped)
            in_block = True
            block_had_content = False
            continue

        if _CONTINUE_READING.match(stripped):
            # Header line itself plus the linked headline that follows.
            dropped.append(stripped)
            skip_next_nonblank = 1
            continue

        if any(p.search(stripped) for p in _DROP_LINE_PATTERNS):
            dropped.append(stripped)
            continue

        kept.append(stripped)

    # Collapse the blank runs left behind by removed lines.
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, dropped


def parse_file(path: Path) -> ParsedArticle | None:
    """Parse one crawled .txt. Returns None if the file is unusable."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    lines = raw.splitlines()
    header: dict[str, str] = {}
    body_start = 0
    for i, line in enumerate(lines):
        if SEPARATOR.match(line.strip()):
            body_start = i + 1
            break
        m = re.match(r"^(標題|記者|新聞網|日期|網址)[：:]\s*(.*)$", line)
        if m:
            header[_HEADER_KEYS[m.group(1)]] = m.group(2).strip()
    else:
        # No separator found — not in the expected format.
        return None

    body, dropped = clean_body("\n".join(lines[body_start:]),
                               outlet=header.get("outlet", ""))
    title = header.get("title", "").strip()
    if not title or not body:
        return None

    return ParsedArticle(
        title=title,
        author=header.get("author", "").strip() or "（未署名）",
        outlet=header.get("outlet", "").strip() or path.parts[-3],
        url=header.get("url", "").strip(),
        published_on=_parse_date(header.get("published_on", "")),
        body=body,
        source_path=str(path),
        dropped_lines=dropped,
    )


def iter_article_files(root: Path, outlet: str | None = None):
    """Yield article .txt paths under `root`, optionally limited to one outlet."""
    outlets = [root / outlet] if outlet else [p for p in root.iterdir() if p.is_dir()]
    for outlet_dir in outlets:
        if not outlet_dir.is_dir() or outlet_dir.name.startswith("_"):
            continue
        for author_dir in sorted(outlet_dir.iterdir()):
            if not author_dir.is_dir():
                continue
            for txt in sorted(author_dir.glob("*.txt")):
                yield txt
