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
