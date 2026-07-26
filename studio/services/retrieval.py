"""Exemplar retrieval strategies.

Four strategies are implemented on purpose, not for flexibility's sake: the
plan commits to A/B testing whether topic-similar exemplar selection actually
helps. arXiv:2509.14543 found that content-similarity selection *reduced*
style-attribution accuracy on English informal writing (it narrows stylistic
diversity), while the RAG literature assumes it helps. That contradiction has
to be settled on this corpus, so `topical` and `random` must be swappable
with everything else held constant.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from corpus.models import Article
from corpus.services import index


@dataclass
class Exemplar:
    article: Article
    score: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "id": self.article.id,
            "title": self.article.title,
            "author": self.article.author.name,
            "score": round(self.score, 4),
            "reason": self.reason,
            "url": self.article.url,
        }


def _base_queryset(outlet_id: int, author_id: int | None, min_chars: int):
    qs = Article.objects.filter(outlet_id=outlet_id, char_count__gte=min_chars)
    if author_id:
        qs = qs.filter(author_id=author_id)
    return qs


def retrieve(
    strategy: str,
    outlet_id: int,
    query_text: str,
    count: int = 4,
    author_id: int | None = None,
    min_chars: int = 400,
    seed: int | None = None,
) -> list[Exemplar]:
    """Pick `count` style exemplars using the named strategy."""
    if strategy == "none" or count <= 0:
        return []

    qs = _base_queryset(outlet_id, author_id, min_chars)
    if strategy == "random":
        return _random(qs, count, seed)
    if strategy == "topical":
        return _topical(qs, query_text, count)
    if strategy == "typical":
        return _typical(qs, query_text, count, outlet_id)
    if strategy == "hybrid":
        half = max(1, count // 2)
        topical = _topical(qs, query_text, half)
        used = {e.article.id for e in topical}
        rnd = _random(qs.exclude(id__in=used), count - len(topical), seed)
        return topical + rnd
    raise ValueError(f"未知的檢索策略：{strategy}")


def _random(qs, count: int, seed: int | None) -> list[Exemplar]:
    ids = list(qs.values_list("id", flat=True))
    if not ids:
        return []
    rng = random.Random(seed)
    picked = rng.sample(ids, min(count, len(ids)))
    articles = {a.id: a for a in Article.objects.filter(id__in=picked).select_related("author")}
    return [Exemplar(articles[i], 0.0, "隨機抽樣") for i in picked if i in articles]


def _typical(qs, query_text: str, count: int, outlet_id: int,
             pool_factor: int = 4, topic_weight: float = 0.4) -> list[Exemplar]:
    """Retrieve topically, then re-rank by how typical of the house voice.

    The A/B result motivating this: topical exemplars scored 0.84 similarity to
    the brief yet produced no style gain over random ones. A plausible reason is
    that an article about the same campaign is not necessarily written in the
    outlet's most characteristic voice — a one-off listicle or a wire rewrite
    can be topically perfect and stylistically atypical.

    So the pool is gathered by topic and then reordered by distance to the
    outlet centroid, with topic deliberately the minority weight: relevance
    only has to be good enough to keep the vocabulary domain right.
    """
    pool = _topical(qs, query_text, count * pool_factor)
    if not pool:
        return []

    typ = index.typicality([e.article.id for e in pool], outlet_id)
    if not typ:
        return pool[:count]

    scored = []
    for e in pool:
        t = typ.get(e.article.id, 0.0)
        blended = topic_weight * e.score + (1 - topic_weight) * t
        scored.append((blended, t, e))
    scored.sort(key=lambda x: -x[0])

    return [
        Exemplar(e.article, blended,
                 f"綜合 {blended:.3f}（主題 {e.score:.3f} / 文體代表性 {t:.3f}）")
        for blended, t, e in scored[:count]
    ]


def _topical(qs, query_text: str, count: int) -> list[Exemplar]:
    from core import embeddings

    if not index.is_built():
        raise RuntimeError(
            "向量索引尚未建立，無法使用主題相似檢索。"
            "請先執行：python manage.py build_index --outlet <媒體>"
        )

    # `.order_by()` drops Article.Meta.ordering — sorting tens of thousands of
    # rows here is wasted work when all we need is the set of candidate rows.
    rows = np.asarray(
        list(qs.order_by().exclude(vector_row__isnull=True)
             .values_list("vector_row", flat=True)),
        dtype=np.int64,
    )
    if not rows.size:
        raise RuntimeError("符合條件的文章都還沒建立向量，請先擴大 build_index 的範圍。")

    qvec = embeddings.embed_query(query_text)
    hits = index.search_rows(qvec, rows, count)

    ids, _ = index.load()
    id_of_row = {int(r): int(ids[r]) for r, _ in hits}
    article_ids = [id_of_row[r] for r, _ in hits]
    articles = {a.id: a for a in Article.objects.filter(id__in=article_ids).select_related("author")}

    out = []
    for row, score in hits:
        art = articles.get(id_of_row[row])
        if art:
            out.append(Exemplar(art, score, f"主題相似度 {score:.3f}"))
    return out
