"""Measured style statistics.

The LLM-induced style guide describes a voice in prose; these numbers pin down
the parts of "voice" that are countable — punctuation density, sentence
rhythm, title shape. Two reasons to compute them rather than let the model
guess: the guide becomes checkable, and the same profile is reused by the
evaluator to score a draft against how the outlet actually writes.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter

EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿]"
)
SENTENCE_END = re.compile(r"[。！？!?]+")
CJK = re.compile(r"[一-鿿]")


def _per_100(count: int, total: int) -> float:
    return round(count / total * 100, 3) if total else 0.0


def profile(articles) -> dict:
    """Compute an aggregate style profile over an iterable of Articles."""
    titles, bodies = [], []
    for a in articles:
        titles.append(a.title)
        bodies.append(a.body)
    if not bodies:
        return {}

    all_body = "\n".join(bodies)
    total_chars = len(all_body)

    sent_lengths, para_counts, para_lengths = [], [], []
    for body in bodies:
        paras = [p for p in body.split("\n") if p.strip()]
        para_counts.append(len(paras))
        para_lengths.extend(len(p) for p in paras)
        for sent in SENTENCE_END.split(body):
            s = sent.strip()
            if s:
                sent_lengths.append(len(s))

    def mean(xs):
        return round(statistics.mean(xs), 1) if xs else 0.0

    def median(xs):
        return round(statistics.median(xs), 1) if xs else 0.0

    title_marks = Counter()
    for t in titles:
        for mark, pattern in [
            ("驚嘆號！", "！"), ("問號？", "？"), ("引號「」", "「"),
            ("分隔｜", "｜"), ("冒號：", "："), ("英文品牌名", None),
        ]:
            if pattern and pattern in t:
                title_marks[mark] += 1
        if re.search(r"[A-Za-z]{3,}", t):
            title_marks["英文品牌名"] += 1
        if re.search(r"\d", t):
            title_marks["含數字"] += 1

    return {
        "sample_size": len(bodies),
        "title": {
            "平均字數": mean([len(t) for t in titles]),
            "特徵出現率": {k: f"{v / len(titles):.0%}" for k, v in title_marks.most_common()},
        },
        "body": {
            "平均全文字數": mean([len(b) for b in bodies]),
            "平均段落數": mean(para_counts),
            "段落平均字數": mean(para_lengths),
            "段落字數中位數": median(para_lengths),
            "句子平均字數": mean(sent_lengths),
            "句子字數中位數": median(sent_lengths),
        },
        "punctuation_per_100_chars": {
            "！": _per_100(all_body.count("！") + all_body.count("!"), total_chars),
            "？": _per_100(all_body.count("？") + all_body.count("?"), total_chars),
            "～": _per_100(all_body.count("～") + all_body.count("~"), total_chars),
            "，": _per_100(all_body.count("，"), total_chars),
            "、": _per_100(all_body.count("、"), total_chars),
            "「」": _per_100(all_body.count("「"), total_chars),
            "（）": _per_100(all_body.count("（") + all_body.count("("), total_chars),
        },
        "emoji_per_1000_chars": round(len(EMOJI.findall(all_body)) / max(total_chars, 1) * 1000, 3),
        "latin_ratio": round(len(re.findall(r"[A-Za-z]", all_body)) / max(total_chars, 1), 4),
        "cjk_ratio": round(len(CJK.findall(all_body)) / max(total_chars, 1), 4),
    }


def distinctive_terms(target_bodies: list[str], contrast_bodies: list[str],
                      top_n: int = 30, min_len: int = 2, max_len: int = 4) -> list[tuple[str, float]]:
    """Terms over-represented in `target` relative to `contrast`.

    Character n-grams rather than a word segmenter: fashion/street jargon
    ("聯乘", "上腳", "鞋頭") is exactly the vocabulary a general-purpose
    Chinese tokeniser tends to split wrongly, and n-grams sidestep that
    without adding a dependency.
    """
    def ngrams(bodies):
        c = Counter()
        for b in bodies:
            for chunk in re.split(r"[^一-鿿]+", b):
                for n in range(min_len, max_len + 1):
                    for i in range(len(chunk) - n + 1):
                        c[chunk[i:i + n]] += 1
        return c

    tgt, ctr = ngrams(target_bodies), ngrams(contrast_bodies)
    tgt_total, ctr_total = max(sum(tgt.values()), 1), max(sum(ctr.values()), 1)

    scored = []
    for term, count in tgt.items():
        if count < 20:
            continue
        p_t = count / tgt_total
        p_c = (ctr.get(term, 0) + 1) / ctr_total
        scored.append((term, round(p_t / p_c, 2)))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]
