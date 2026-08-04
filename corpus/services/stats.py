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
                      top_n: int = 30, min_len: int = 2, max_len: int = 4,
                      min_doc_ratio: float = 0.05,
                      exclude: str = "") -> list[tuple[str, float]]:
    """Terms over-represented in `target` relative to `contrast`.

    Character n-grams rather than a word segmenter: fashion/street jargon
    ("聯乘", "上腳", "鞋頭") is exactly the vocabulary a general-purpose
    Chinese tokeniser tends to split wrongly, and n-grams sidestep that
    without adding a dependency.

    Three filters keep the result about *style* rather than about whatever the
    sample happened to be reporting. Before them, 中時新聞網's list came back as
    「李四、四川、李四川、長照、食藥、藥署、食藥署、聞網、新聞網…」 — a
    politician, a health agency and the outlet's own name, each shredded into
    every overlapping fragment of itself:

    **Document frequency.** A habit of writing shows up across many articles;
    a name shows up many times inside one. Requiring a term in `min_doc_ratio`
    of the sample separates the two, and is the filter that does most of the
    work — no name survives it unless the outlet really does keep saying it.

    The threshold is a measured compromise, not a principled constant. Raising
    it to 0.15 does clear a news sample completely, but by then it has also
    taken 威士忌 and 樂團 off GQ's list and left function words behind: the two
    genres disagree about what a frequent term looks like. At 0.05 people's
    names are reliably gone from both, while a news outlet still surfaces the
    subjects it happens to cover that season (颱風, 花蓮). That residue is
    inherent — a typhoon really is discussed across a tenth of the sample, and
    no frequency test can tell that from a habit of phrasing. It is tolerable
    because this list is a supporting section: the style a draft actually
    imitates comes from the induced analysis, which reads the articles rather
    than counting them.

    **Maximal n-grams.** 「李四」 and 「四川」 are not two findings about a
    corpus that says 「李四川」. A shorter gram that nearly always appears
    inside a longer one is dropped, so one term takes one slot instead of six.

    **The outlet's own name.** Bylines and self-references make 中時新聞網 look
    highly distinctive of 中時新聞網. True, and useless.
    """
    def count(bodies):
        total, docs = Counter(), Counter()
        for b in bodies:
            seen = set()
            for chunk in re.split(r"[^一-鿿]+", b):
                for n in range(min_len, max_len + 1):
                    for i in range(len(chunk) - n + 1):
                        gram = chunk[i:i + n]
                        total[gram] += 1
                        seen.add(gram)
            for gram in seen:
                docs[gram] += 1
        return total, docs

    tgt, tgt_docs = count(target_bodies)
    ctr, _ = count(contrast_bodies)
    tgt_total, ctr_total = max(sum(tgt.values()), 1), max(sum(ctr.values()), 1)
    min_docs = max(2, int(len(target_bodies) * min_doc_ratio))
    banned = {exclude[i:i + n]
              for n in range(min_len, max_len + 1)
              for i in range(len(exclude) - n + 1)} if exclude else set()

    scored = []
    for term, n in tgt.items():
        if n < 20 or tgt_docs[term] < min_docs or term in banned:
            continue
        p_t = n / tgt_total
        p_c = (ctr.get(term, 0) + 1) / ctr_total
        scored.append((term, round(p_t / p_c, 2)))
    scored.sort(key=lambda x: -x[1])

    # Longest first, so 「李四川」 claims the slot and 「李四」 is recognised as
    # part of it rather than the other way round.
    kept: list[tuple[str, float]] = []
    for term, score in sorted(scored, key=lambda x: (-len(x[0]), -x[1])):
        if any(term in longer and tgt[term] <= tgt[longer] * 1.3
               for longer, _ in kept):
            continue
        kept.append((term, score))

    kept.sort(key=lambda x: -x[1])
    return kept[:top_n]
