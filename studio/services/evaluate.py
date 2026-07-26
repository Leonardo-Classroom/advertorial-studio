"""Draft evaluation — a metric ensemble, not a single score.

arXiv:2508.06374 tested BLEU/ROUGE, style embeddings and LLM-as-judge against
human judgement and found no single metric reliable; ensembles beat every
individual metric. So four independent signals are computed and stored
side by side, and none of them is collapsed into an overall number:

  style_similarity   — is this close to how the outlet normally writes?
  max_overlap        — did it lift text from the exemplars? (plagiarism guard)
  fact_coverage      — did the sanctioned facts survive, and only those?
  judge_scores       — the qualitative dimensions no metric captures.

Plus a deviation report against the measured style profile, which turns "feels
off" into "exclamation marks are 3x the outlet's rate".
"""
from __future__ import annotations

import re

import numpy as np

from corpus.models import Article
from corpus.services import index, stats
from core import llm
from studio.models import Evaluation, GenerationRun, Revision

SHINGLE = 12  # characters; long enough that a match means copying, not coincidence

JUDGE_INSTRUCTIONS = """你是嚴格的文體評審，負責判斷一篇廣編稿「像不像」目標媒體的風格。
你只評風格與執行品質，不評品牌決策。
評分要嚴格：3 分代表「勉強及格、看得出是外人寫的」，5 分要真的難以分辨。
全程使用繁體中文。只輸出 JSON，不要有其他文字或 ``` 標記。"""

JUDGE_TASK = """請依據下方風格指南，為這篇廣編稿評分。

【風格指南】
{guide}

【待評稿件】
{draft}

輸出 JSON：
{{
  "title_fit": 1-5,
  "tone_fit": 1-5,
  "rhythm_fit": 1-5,
  "diction_fit": 1-5,
  "advertorial_completeness": 1-5,
  "reads_as_human": 1-5,
  "weakest_dimension": "最弱的一項，並說明為什麼",
  "concrete_fixes": ["具體可執行的修改建議，逐條"],
  "comment": "整體評語，2-4 句"
}}"""


def _shingles(text: str, n: int = SHINGLE) -> set[str]:
    clean = re.sub(r"\s+", "", text)
    return {clean[i:i + n] for i in range(len(clean) - n + 1)} if len(clean) >= n else set()


def overlap_against_exemplars(draft: str, exemplar_ids: list[int]) -> tuple[float, str]:
    """Highest proportion of the draft's shingles found in any single exemplar."""
    draft_sh = _shingles(draft)
    if not draft_sh or not exemplar_ids:
        return 0.0, ""
    worst, source = 0.0, ""
    for art in Article.objects.filter(id__in=exemplar_ids):
        shared = draft_sh & _shingles(art.body + art.title)
        ratio = len(shared) / len(draft_sh)
        if ratio > worst:
            worst, source = ratio, art.title
    return round(worst, 4), source


def check_facts(draft: str, facts: dict) -> dict:
    """Which sanctioned facts made it into the draft, and which went missing.

    Only checks the fields that must survive verbatim (brand, product codes,
    KOL names, mandatory terms) — prose fields like campaign_context are meant
    to be reworded, so absence there is not an error.
    """
    checkable: list[str] = []
    for key in ("brand", "slogan"):
        value = facts.get(key)
        if isinstance(value, str) and value.strip():
            checkable.append(value.strip())
    for key in ("product", "kol", "mandatory_terms"):
        value = facts.get(key)
        if isinstance(value, list):
            checkable.extend(str(v).strip() for v in value if str(v).strip())

    normalised = re.sub(r"\s+", "", draft).lower()
    present = [t for t in checkable if re.sub(r"\s+", "", t).lower() in normalised]
    missing = [t for t in checkable if t not in present]

    forbidden = [
        t for t in (facts.get("forbidden_terms") or [])
        if str(t).strip() and re.sub(r"\s+", "", str(t)).lower() in normalised
    ]

    return {
        "checked": len(checkable),
        "present": present,
        "missing": missing,
        "forbidden_hits": forbidden,
        "coverage": round(len(present) / len(checkable), 3) if checkable else None,
    }


def style_deviation(draft: str, guide_sections: dict) -> dict:
    """Compare the draft's countable habits against the outlet's measured profile."""
    measured = (guide_sections or {}).get("measured") or {}
    if not measured:
        return {}

    class _Fake:
        def __init__(self, body):
            self.title = ""
            self.body = body

    draft_profile = stats.profile([_Fake(draft)])
    out = {}

    for label, path in [
        ("！每百字", ("punctuation_per_100_chars", "！")),
        ("？每百字", ("punctuation_per_100_chars", "？")),
        ("～每百字", ("punctuation_per_100_chars", "～")),
        ("「」每百字", ("punctuation_per_100_chars", "「」")),
    ]:
        target = measured.get(path[0], {}).get(path[1])
        actual = draft_profile.get(path[0], {}).get(path[1])
        if target is None or actual is None:
            continue
        out[label] = {"目標": target, "本稿": actual,
                      "偏離": _ratio_label(actual, target)}

    t_emoji = measured.get("emoji_per_1000_chars")
    a_emoji = draft_profile.get("emoji_per_1000_chars")
    if t_emoji is not None and a_emoji is not None:
        out["emoji每千字"] = {"目標": t_emoji, "本稿": a_emoji,
                              "偏離": _ratio_label(a_emoji, t_emoji)}

    t_sent = (measured.get("body") or {}).get("句子平均字數")
    a_sent = (draft_profile.get("body") or {}).get("句子平均字數")
    if t_sent and a_sent:
        out["句子平均字數"] = {"目標": t_sent, "本稿": a_sent,
                               "偏離": _ratio_label(a_sent, t_sent)}
    return out


def _ratio_label(actual: float, target: float) -> str:
    if not target:
        return "—"
    ratio = actual / target
    if ratio > 1.5:
        return f"偏多 {ratio:.1f}x"
    if ratio < 0.67:
        return f"偏少 {ratio:.1f}x"
    return "接近"


def evaluate(run: GenerationRun, revision: Revision | None = None,
             run_judge: bool = True) -> Evaluation:
    draft = revision.output if revision else run.output
    ev = Evaluation(run=run, revision=revision)

    exemplar_ids = [e.get("id") for e in (run.exemplars or []) if e.get("id")]
    ev.max_overlap, ev.overlap_source = overlap_against_exemplars(draft, exemplar_ids)
    ev.fact_coverage = check_facts(draft, run.brief.facts or {})

    # Style-embedding distance, when an index exists for this outlet.
    try:
        if index.is_built():
            from core import embeddings

            outlet_ids = list(
                Article.objects.filter(outlet_id=run.outlet_id)
                .exclude(vector_row__isnull=True)
                .values_list("id", flat=True)[:4000]
            )
            centre = index.centroid(outlet_ids)
            draft_vec = embeddings.embed_texts([draft[:2000]], input_type="document",
                                               concurrency=1)[0]
            if centre is not None:
                ev.style_similarity = round(float(np.dot(draft_vec, centre)), 4)
            if exemplar_ids:
                ex_centre = index.centroid(exemplar_ids)
                if ex_centre is not None:
                    ev.exemplar_similarity = round(float(np.dot(draft_vec, ex_centre)), 4)
    except Exception as exc:  # noqa: BLE001 - a metric failing must not lose the draft
        ev.judge_comment = f"（風格向量計算失敗：{exc}）\n"

    if run_judge and run.style_guide:
        try:
            scores = llm.complete_json(
                instructions=JUDGE_INSTRUCTIONS,
                user_input=JUDGE_TASK.format(
                    guide=run.style_guide.content[:6000], draft=draft[:8000]
                ),
                timeout=300,
            )
            ev.judge_scores = scores
            ev.judge_comment += str(scores.get("comment", ""))
        except Exception as exc:  # noqa: BLE001
            ev.judge_comment += f"（LLM 評審失敗：{exc}）"

    if run.style_guide:
        ev.judge_scores = {**(ev.judge_scores or {}),
                           "deviation": style_deviation(draft, run.style_guide.sections)}

    ev.save()
    return ev
