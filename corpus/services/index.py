"""Vector index over the style corpus.

Deliberately not a vector database. At this corpus size (~56k articles) a
single normalised float32 matrix plus a numpy matmul answers a query in
milliseconds, and it keeps the whole retrieval path inspectable — which
matters because the A/B experiment has to compare retrieval strategies
honestly rather than through an opaque library.

Layout under `var/index/`:
    articles.npy      float32 (N, dims), L2-normalised rows
    articles_ids.npy  int64 (N,), Article primary keys aligned to rows
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from django.conf import settings
from django.utils import timezone

from corpus.models import Article, EmbeddingIndex

MATRIX_PATH = "articles.npy"
IDS_PATH = "articles_ids.npy"
CHUNK = 2000  # articles loaded into memory at a time

_cache: dict[str, object] = {}


def index_dir() -> Path:
    d = Path(settings.INDEX_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def indexing_text(title: str, body: str, max_chars: int | None = None) -> str:
    """What actually gets embedded.

    Title plus the lead is a better topical fingerprint than the whole body:
    the tail of a fashion article is mostly product specs and pricing, which
    blurs the topic signal and costs tokens. Length is configurable because
    under a requests-per-minute cap it directly sets indexing throughput —
    shorter texts pack more articles into each request.
    """
    limit = max_chars if max_chars is not None else settings.EMBED_INDEX_CHARS
    return f"{title}\n{body[:limit]}"


def _target_ids(outlet: str | None, limit_per_author: int) -> list[int]:
    qs = Article.objects.all()
    if outlet:
        qs = qs.filter(outlet__name=outlet)

    if not limit_per_author:
        return list(qs.order_by("id").values_list("id", flat=True))

    keep: list[int] = []
    # `.order_by()` clears Article.Meta.ordering first: Django appends ordering
    # columns to a DISTINCT select, which would make every row look distinct
    # and turn this into one subquery per article instead of per author.
    for author_id in qs.order_by().values_list("author_id", flat=True).distinct():
        keep.extend(
            qs.filter(author_id=author_id)
            .order_by("-published_on", "-id")
            .values_list("id", flat=True)[:limit_per_author]
        )
    return sorted(keep)


def build(
    outlet: str | None = None,
    limit_per_author: int = 0,
    rebuild: bool = False,
    progress=None,
) -> EmbeddingIndex:
    """Embed articles and write the matrix. Incremental unless `rebuild`.

    The pending set is diffed in Python rather than with a SQL `NOT IN`:
    at this scale the id list would blow past SQLite's bound-parameter limit.
    Articles are then loaded in chunks so peak memory stays bounded instead of
    holding every article body at once.
    """
    from core import embeddings

    existing_ids, existing_matrix = _load_raw()
    if rebuild:
        existing_ids, existing_matrix = np.zeros(0, dtype=np.int64), None

    already = set(existing_ids.tolist()) if existing_ids.size else set()
    todo_ids = [i for i in _target_ids(outlet, limit_per_author) if i not in already]

    new_vectors: list[np.ndarray] = []
    new_ids: list[int] = []
    done_so_far = 0
    total = len(todo_ids)

    for start in range(0, total, CHUNK):
        chunk_ids = todo_ids[start:start + CHUNK]
        rows = list(
            Article.objects.filter(id__in=chunk_ids)
            .order_by("id")
            .values_list("id", "title", "body")
        )
        if not rows:
            continue
        texts = [indexing_text(title, body) for _, title, body in rows]

        def chunk_progress(done, _total, base=done_so_far):
            if progress:
                progress(base + done, total)

        new_vectors.append(
            embeddings.embed_texts(texts, input_type="document", progress=chunk_progress)
        )
        new_ids.extend(r[0] for r in rows)
        done_so_far += len(rows)

    if new_vectors:
        stacked = np.vstack(new_vectors)
        if existing_matrix is not None and existing_matrix.size:
            matrix = np.vstack([np.asarray(existing_matrix), stacked])
            ids = np.concatenate([existing_ids, np.asarray(new_ids, dtype=np.int64)])
        else:
            matrix, ids = stacked, np.asarray(new_ids, dtype=np.int64)
    else:
        matrix = (np.asarray(existing_matrix) if existing_matrix is not None
                  else np.zeros((0, settings.EMBED_DIMENSIONS), dtype=np.float32))
        ids = existing_ids

    d = index_dir()
    np.save(d / MATRIX_PATH, matrix)
    np.save(d / IDS_PATH, ids)
    _cache.clear()

    _write_back_rows(ids, new_ids)

    record, _ = EmbeddingIndex.objects.update_or_create(
        name="articles",
        defaults={
            # The active backend's model, not the API setting — mislabelling
            # which model produced a matrix makes it impossible to tell later
            # whether an index needs rebuilding after a backend switch.
            "model": embeddings.active_model_name(),
            "dimensions": int(matrix.shape[1]) if matrix.size else settings.EMBED_DIMENSIONS,
            "vector_count": int(matrix.shape[0]),
            "note": f"本次新增 {len(new_ids)} 筆",
        },
    )
    return record


def _write_back_rows(all_ids: np.ndarray, changed_ids: list[int]) -> None:
    """Point each Article at its matrix row so retrieval can gather directly."""
    if not len(all_ids):
        return
    row_of = {int(aid): i for i, aid in enumerate(all_ids.tolist())}
    now = timezone.now()
    targets = changed_ids or list(row_of)

    for start in range(0, len(targets), 500):
        batch_ids = targets[start:start + 500]
        objs = list(Article.objects.filter(id__in=batch_ids).only("id"))
        for obj in objs:
            obj.vector_row = row_of.get(obj.id)
            obj.indexed_at = now
        Article.objects.bulk_update(objs, ["vector_row", "indexed_at"], batch_size=500)


def _load_raw() -> tuple[np.ndarray, np.ndarray | None]:
    d = index_dir()
    mpath, ipath = d / MATRIX_PATH, d / IDS_PATH
    if not mpath.exists() or not ipath.exists():
        return np.zeros(0, dtype=np.int64), None
    return np.load(ipath), np.load(mpath, mmap_mode="r")


def load() -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, matrix), cached across calls in-process."""
    if "ids" not in _cache:
        ids, matrix = _load_raw()
        if matrix is None:
            matrix = np.zeros((0, settings.EMBED_DIMENSIONS), dtype=np.float32)
        _cache["ids"] = ids
        _cache["matrix"] = matrix
    return _cache["ids"], _cache["matrix"]  # type: ignore[return-value]


def is_built() -> bool:
    ids, _ = load()
    return bool(ids.size)


def search_rows(query_vec: np.ndarray, rows: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Cosine-rank a subset of rows. Returns [(row, score)] best first."""
    _, matrix = load()
    if not rows.size or not matrix.size:
        return []
    subset = np.asarray(matrix[rows])
    scores = subset @ query_vec.astype(np.float32)
    k = min(top_k, scores.shape[0])
    if k < scores.shape[0]:
        top = np.argpartition(-scores, k - 1)[:k]
    else:
        top = np.arange(scores.shape[0])
    top = top[np.argsort(-scores[top])]
    return [(int(rows[i]), float(scores[i])) for i in top]


def outlet_centroid(outlet_id: int, author_id: int | None = None,
                    sample: int = 4000) -> np.ndarray | None:
    """The stylistic centre of mass to measure typicality against, cached.

    Scoped to the author when a run targets one. Measuring an author's articles
    against the *outlet* average is backwards for author imitation: it would
    rank that writer's most characteristic pieces as least typical, precisely
    because what makes them characteristic is departing from the house mean.
    """
    key = f"centroid:{outlet_id}:{author_id}:{sample}"
    if key not in _cache:
        qs = Article.objects.filter(outlet_id=outlet_id)
        if author_id:
            qs = qs.filter(author_id=author_id)
        ids = list(
            qs.exclude(vector_row__isnull=True).order_by()
            .values_list("id", flat=True)[:sample]
        )
        _cache[key] = centroid(ids)
    return _cache[key]  # type: ignore[return-value]


def typicality(article_ids: list[int], outlet_id: int,
               author_id: int | None = None) -> dict[int, float]:
    """Cosine of each article to its outlet (or author) centroid, by article id."""
    centre = outlet_centroid(outlet_id, author_id)
    ids, matrix = load()
    if centre is None or not ids.size:
        return {}
    row_of = {int(a): i for i, a in enumerate(ids.tolist())}
    out = {}
    for aid in article_ids:
        row = row_of.get(aid)
        if row is not None:
            out[aid] = float(np.asarray(matrix[row]) @ centre)
    return out


def centroid(article_ids: list[int]) -> np.ndarray | None:
    """Mean normalised vector of the given articles — a style "centre of mass".

    Used by the evaluator as the reference point for "how close is this draft
    to how the outlet normally writes".
    """
    ids, matrix = load()
    if not ids.size:
        return None
    row_of = {int(a): i for i, a in enumerate(ids.tolist())}
    rows = [row_of[a] for a in article_ids if a in row_of]
    if not rows:
        return None
    mean = np.asarray(matrix[rows]).mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean
