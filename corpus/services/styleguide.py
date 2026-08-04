"""Style-guide induction — component D of the plan.

Rather than dumping raw articles into every generation prompt, we distil the
recurring habits of an outlet (or author) once, into an editable document.
Two halves:

  * measured statistics from `stats.py` — countable, not guessable;
  * an LLM reading of a stratified sample, for the qualitative habits
    (title formulas, opening moves, closing conventions) that no metric
    captures.

The measured half is handed to the model as evidence so it describes what the
corpus actually does instead of reciting generic "fashion magazine" tropes.
"""
from __future__ import annotations

import json
import re
import random

from corpus.models import Article, Author, Outlet, StyleGuide
from corpus.services import stats
from core import llm

INSTRUCTIONS = """你是一位資深的中文文體分析師，專長是拆解媒體的「寫作套路」。
你的任務不是稱讚或評論內容，而是精準描述這個媒體「怎麼寫」，讓另一位寫手能照著寫出幾可亂真的稿子。
全程使用繁體中文。只輸出 JSON，不要有任何其他文字或 ``` 標記。"""

TASK_TEMPLATE = """請分析以下來自「{scope}」的 {n} 篇文章，歸納出可操作的寫作風格指南。

【已量測到的客觀統計數據（請以此為準，不要與之矛盾）】
{measured}

【高辨識度用詞（相對於另一家媒體明顯偏好的詞彙）】
{terms}

【文章樣本】
{samples}

請輸出以下 JSON 結構：
{{
  "title_formulas": ["標題公式，用具體句型描述，例如：驚嘆句＋品牌＋聯名資訊＋問句收尾", ...],
  "opening_moves": ["開場慣用手法，2-4 種", ...],
  "structure": "全文結構的典型安排（幾段、有無小標、小標怎麼下）",
  "diction_and_tone": ["用詞與語氣特徵，越具體越好", ...],
  "signature_phrases": ["這個媒體特別愛用的詞彙或句式", ...],
  "punctuation_habits": "標點與 emoji 的使用習慣（請對照上面的統計數據）",
  "closing_conventions": ["結尾慣例，例如購買資訊、CTA、發問互動", ...],
  "avoid": ["模仿時要避免的寫法，也就是這個媒體不會出現的東西", ...],
  "one_paragraph_summary": "用一段話總結這個媒體的聲音"
}}"""


COLUMN_PREFIX = re.compile(r"^([^｜|]{1,12})[｜|]")
# Share of the sample reserved for the outlet's recurring column formats.
COLUMN_QUOTA = 0.15


def _uniform_sample(qs, n: int, seed: int = 20260726) -> list[Article]:
    """Plain random draw — kept as the control arm for sampling experiments."""
    ids = list(qs.order_by().values_list("id", flat=True))
    if not ids:
        return []
    rng = random.Random(seed)
    picked = rng.sample(ids, min(n, len(ids)))
    return list(Article.objects.filter(id__in=picked).select_related("author", "outlet"))


def _stratified_sample(qs, n: int, seed: int = 20260726) -> list[Article]:
    """Draw across year and length, with a floor for recurring column formats.

    ⚠ NOT the default, despite being the more sophisticated option — it
    measured *worse*. The reasoning behind it was sound as far as it went:
    uniform sampling covers rare-but-distinctive patterns badly, and on
    COOL-STYLE a 24-article uniform draw catches only 0.9 of the two main
    column formats (台灣販售預告｜, COOL 開箱｜) where this catches 2.0.

    But better corpus coverage did not produce better drafts. In a 2×2 over
    sampling method and sample size (3 drafts per cell), stratified sampling
    added a full extra off-target style dimension versus uniform (4.67 vs 3.67
    deviations), consistently at both sample sizes. The likely cause is the
    quota itself: reserving 15% for formats that are 6% of the corpus pulls the
    guide toward an announcement-style register, and the drafts followed —
    exclamation marks fell 7× below target while question marks ran 4.4× over.

    Kept because the coverage argument may still hold for an outlet whose
    formats are a larger share of output, and because the negative result is
    worth being able to reproduce.
    """
    rows = list(qs.order_by().values_list("id", "title", "char_count", "published_on"))
    if not rows:
        return []
    if n >= len(rows):
        return list(Article.objects.filter(id__in=[r[0] for r in rows])
                    .select_related("author", "outlet"))

    rng = random.Random(seed)
    columns = [r for r in rows if COLUMN_PREFIX.match(r[1] or "")]
    plain = [r for r in rows if not COLUMN_PREFIX.match(r[1] or "")]

    picked: list[int] = []
    want_columns = min(int(n * COLUMN_QUOTA), len(columns))
    if want_columns:
        # Spread the quota over distinct formats rather than 3 of the same one.
        by_format: dict[str, list] = {}
        for r in columns:
            by_format.setdefault(COLUMN_PREFIX.match(r[1]).group(1).strip(), []).append(r)
        formats = sorted(by_format, key=lambda k: -len(by_format[k]))
        i = 0
        while len(picked) < want_columns and formats:
            fmt = formats[i % len(formats)]
            bucket = by_format[fmt]
            if bucket:
                picked.append(bucket.pop(rng.randrange(len(bucket)))[0])
            else:
                formats.remove(fmt)
                continue
            i += 1

    # The rest: stratify by publication year, then by length tertile within it,
    # so neither a single season nor one article length dominates the guide.
    remaining = n - len(picked)
    by_year: dict[int, list] = {}
    for r in plain:
        by_year.setdefault(r[3].year if r[3] else 0, []).append(r)

    years = sorted(by_year)
    per_year = max(1, remaining // max(len(years), 1))
    for year in years:
        bucket = sorted(by_year[year], key=lambda r: r[2])
        if not bucket:
            continue
        third = max(len(bucket) // 3, 1)
        bands = [bucket[:third], bucket[third:2 * third], bucket[2 * third:]]
        for b_i in range(per_year):
            band = bands[b_i % 3]
            if band:
                picked.append(band.pop(rng.randrange(len(band)))[0])
            if len(picked) >= n:
                break
        if len(picked) >= n:
            break

    # Top up if year buckets ran dry.
    if len(picked) < n:
        leftovers = [r[0] for r in plain if r[0] not in set(picked)]
        rng.shuffle(leftovers)
        picked.extend(leftovers[:n - len(picked)])

    return list(Article.objects.filter(id__in=picked).select_related("author", "outlet"))


def induce(
    outlet: Outlet,
    author: Author | None = None,
    sample_size: int = 24,
    sampling: str = "uniform",
    excerpt_chars: int = 1200,
    stats_sample: int = 400,
) -> StyleGuide:
    """Build (and persist) a style guide for an outlet or a single author."""
    qs = Article.objects.filter(outlet=outlet, char_count__gte=400)
    if author:
        qs = qs.filter(author=author)

    if not qs.exists():
        raise RuntimeError(f"{outlet.name} 底下沒有足夠的文章可供分析。")

    draw = _stratified_sample if sampling == "stratified" else _uniform_sample
    sample = draw(qs, sample_size)
    # Statistics stay on a uniform draw: they are meant to describe the corpus
    # as it is, and a column quota would bias the measured averages.
    measured = stats.profile(_uniform_sample(qs, stats_sample))

    # Contrast against the other target outlet so "distinctive" means
    # distinctive *between the two styles we actually have to tell apart*.
    contrast_qs = Article.objects.filter(
        outlet__is_target=True, char_count__gte=400
    ).exclude(outlet=outlet)
    target_bodies = [a.body for a in _uniform_sample(qs, 200)]
    contrast_bodies = [a.body for a in _uniform_sample(contrast_qs, 200)]
    terms = (stats.distinctive_terms(target_bodies, contrast_bodies, exclude=outlet.name)
             if contrast_bodies else [])

    samples_text = "\n\n".join(
        f"--- 第 {i + 1} 篇（{a.published_on}，作者 {a.author.name}）---\n"
        f"標題：{a.title}\n內文：{a.body[:excerpt_chars]}"
        for i, a in enumerate(sample)
    )
    scope = f"{outlet.name} / {author.name}" if author else f"{outlet.name}（全站）"

    data = llm.complete_json(
        instructions=INSTRUCTIONS,
        user_input=TASK_TEMPLATE.format(
            scope=scope,
            n=len(sample),
            measured=json.dumps(measured, ensure_ascii=False, indent=2),
            terms="、".join(f"{t}({r})" for t, r in terms[:25]) or "（無對照資料）",
            samples=samples_text,
        ),
        timeout=300,
    )

    sections = {"measured": measured, "distinctive_terms": terms[:30], "induced": data,
                "sampling": sampling, "sample_size": len(sample),
                "truncated": bool(data.get("_truncated")),
                "unparsed": "_raw" in data}
    return StyleGuide.objects.create(
        outlet=outlet,
        author=author,
        content=render_markdown(scope, data, measured, terms),
        sections=sections,
        sample_size=len(sample),
        model=llm.current_model(),
    )


def render_markdown(scope: str, data: dict, measured: dict, terms: list) -> str:
    """Human-readable, hand-editable form — this is what goes into the prompt."""
    if "_raw" in data:
        return (f"# {scope} 風格指南\n\n"
                "> ⚠ 模型未回傳可解析的結構化輸出，以下為原始文字。"
                "常見原因是取樣篇數過多、分析被輸出長度上限截斷——"
                "請降低 --sample-size 後重新產生。\n\n"
                f"{data['_raw']}")

    warning = ""
    if data.get("_truncated"):
        warning = ("> ⚠ 模型的輸出被長度上限截斷，本指南是從殘缺 JSON 修復而來，"
                   "末尾可能缺少一到兩個欄位。若要完整版，請降低 --sample-size。\n\n")

    def bullets(key):
        vals = data.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        return "\n".join(f"- {v}" for v in vals) or "-（未歸納出）"

    body = measured.get("body", {})
    punct = measured.get("punctuation_per_100_chars", {})
    return warning + f"""# {scope} 風格指南

## 一句話總結
{data.get('one_paragraph_summary', '（無）')}

## 標題公式
{bullets('title_formulas')}

## 開場手法
{bullets('opening_moves')}

## 全文結構
{data.get('structure', '（無）')}

## 用詞與語氣
{bullets('diction_and_tone')}

## 招牌詞彙／句式
{bullets('signature_phrases')}

## 標點與 emoji
{data.get('punctuation_habits', '（無）')}

## 結尾慣例
{bullets('closing_conventions')}

## 模仿時要避免
{bullets('avoid')}

## 客觀統計（量測值，非模型推測）
- 平均全文字數：{body.get('平均全文字數', '-')}；平均段落數：{body.get('平均段落數', '-')}；段落平均字數：{body.get('段落平均字數', '-')}
- 句子平均字數：{body.get('句子平均字數', '-')}（中位數 {body.get('句子字數中位數', '-')}）
- 每 100 字標點：！{punct.get('！', '-')}　？{punct.get('？', '-')}　～{punct.get('～', '-')}　「」{punct.get('「」', '-')}
- 每 1000 字 emoji：{measured.get('emoji_per_1000_chars', '-')}
- 英文字母佔比：{measured.get('latin_ratio', '-')}

## 高辨識度用詞
{('、'.join(f'{t}' for t, _ in terms[:25])) or '（無對照資料）'}
"""
