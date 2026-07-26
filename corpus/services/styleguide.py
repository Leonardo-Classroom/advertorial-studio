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


def _stratified_sample(qs, n: int) -> list[Article]:
    """Spread the sample across time and length instead of taking the newest N.

    A style guide built only from the most recent articles would encode a
    season's fads as if they were house style.
    """
    ids = list(qs.values_list("id", flat=True))
    if not ids:
        return []
    rng = random.Random(20260726)
    picked = rng.sample(ids, min(n, len(ids)))
    return list(Article.objects.filter(id__in=picked).select_related("author", "outlet"))


def induce(
    outlet: Outlet,
    author: Author | None = None,
    sample_size: int = 24,
    excerpt_chars: int = 1200,
    stats_sample: int = 400,
) -> StyleGuide:
    """Build (and persist) a style guide for an outlet or a single author."""
    qs = Article.objects.filter(outlet=outlet, char_count__gte=400)
    if author:
        qs = qs.filter(author=author)

    if not qs.exists():
        raise RuntimeError(f"{outlet.name} 底下沒有足夠的文章可供分析。")

    sample = _stratified_sample(qs, sample_size)
    measured = stats.profile(_stratified_sample(qs, stats_sample))

    # Contrast against the other target outlet so "distinctive" means
    # distinctive *between the two styles we actually have to tell apart*.
    contrast_qs = Article.objects.filter(
        outlet__is_target=True, char_count__gte=400
    ).exclude(outlet=outlet)
    target_bodies = [a.body for a in _stratified_sample(qs, 200)]
    contrast_bodies = [a.body for a in _stratified_sample(contrast_qs, 200)]
    terms = stats.distinctive_terms(target_bodies, contrast_bodies) if contrast_bodies else []

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

    sections = {"measured": measured, "distinctive_terms": terms[:30], "induced": data}
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
        return f"# {scope} 風格指南\n\n（模型未回傳結構化 JSON，以下為原始輸出）\n\n{data['_raw']}"

    def bullets(key):
        vals = data.get(key) or []
        if isinstance(vals, str):
            vals = [vals]
        return "\n".join(f"- {v}" for v in vals) or "-（未歸納出）"

    body = measured.get("body", {})
    punct = measured.get("punctuation_per_100_chars", {})
    return f"""# {scope} 風格指南

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
