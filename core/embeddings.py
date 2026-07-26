"""Embedding layer.

Verified against the Azure AI Services endpoint on 2026-07-26:
  * `text-embedding-3-*` are NOT deployed on this resource (`unknown_model`).
  * `embed-v-4-0` (Cohere Embed v4) works: 1536 dims natively, honours the
    `dimensions` parameter, and accepts `input_type` of 'text'/'query'/'document'.
  * Two independent limits apply per request, and both must be respected:
      - at most 96 inputs;
      - at most 8000 tokens **for the whole request body**, not per input.
    The token cap is the binding one in practice. Short probe strings sail past
    96 inputs (~1.3k tokens total), but real articles run ~480 tokens each, so a
    96-input batch lands near 46k tokens and the service returns 413
    `tokens_limit_reached`. Batching is therefore driven by a token budget.

`input_type` matters for retrieval quality: Cohere embeds documents and queries
into deliberately asymmetric spaces, so indexing must use 'document' and
searching must use 'query'.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from django.conf import settings
from openai import OpenAI

MAX_BATCH = 96
MAX_REQUEST_TOKENS = 8000
# Leave headroom: the estimate below is approximate and a 413 costs a full
# retry cycle, so aim well under the hard cap.
TOKEN_BUDGET = 6500
# Measured on this endpoint: 2000 Chinese characters billed as 1600 tokens.
# Rounded up, plus a fixed per-input overhead.
TOKENS_PER_CHAR = 0.9
PER_INPUT_OVERHEAD = 8
# A single input must fit the budget on its own.
MAX_CHARS_PER_INPUT = int(TOKEN_BUDGET / TOKENS_PER_CHAR)


class _RateLimiter:
    """Sliding-window limiter shared by all embedding threads.

    The deployment is on a free tier capped at 10 requests per 60 seconds
    (429 `RateLimitReached`). Waiting for a slot is far cheaper than issuing
    the request and burning a retry cycle on the rejection — and because the
    cap counts *requests*, not tokens, the way to index faster is to pack each
    request closer to the 8000-token ceiling, not to raise concurrency.
    """

    def __init__(self, rpm: int, window: float = 60.0):
        self.rpm = max(rpm, 1)
        self.window = window
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self.window:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = self.window - (now - self._times[0]) + 0.05
            time.sleep(max(wait, 0.05))


@lru_cache(maxsize=1)
def get_limiter() -> _RateLimiter:
    return _RateLimiter(getattr(settings, "EMBED_RPM", 10))


def estimate_tokens(text: str) -> int:
    return int(len(text) * TOKENS_PER_CHAR) + PER_INPUT_OVERHEAD


def plan_batches(texts: list[str], max_items: int | None = None) -> list[list[int]]:
    """Group text indices into requests that respect both service limits."""
    limit = min(max_items or settings.EMBED_BATCH_SIZE, MAX_BATCH)
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for i, text in enumerate(texts):
        cost = estimate_tokens(text)
        if current and (len(current) >= limit or current_tokens + cost > TOKEN_BUDGET):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not settings.LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY 未設定。請在專案根目錄的 .env 填入金鑰（可參考 .env.example）。"
        )
    return OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        timeout=settings.LLM_TIMEOUT,
    )


def _embed_batch(texts: list[str], input_type: str, retries: int = 4) -> list[list[float]]:
    client = get_client()
    kwargs = {"model": settings.EMBED_MODEL, "input": texts}
    if settings.EMBED_DIMENSIONS:
        kwargs["dimensions"] = settings.EMBED_DIMENSIONS
    if input_type:
        kwargs["extra_body"] = {"input_type": input_type}

    limiter = get_limiter()
    last_error = None
    for attempt in range(retries):
        try:
            limiter.acquire()
            resp = client.embeddings.create(**kwargs)
            # The API does not guarantee ordering, so sort by index.
            ordered = sorted(resp.data, key=lambda d: d.index)
            return [d.embedding for d in ordered]
        except Exception as exc:  # noqa: BLE001 - retry on any transport/rate error
            last_error = exc
            # A 429 means the window is genuinely full; short exponential
            # backoff would just burn the remaining retries, so sit out a
            # whole window instead.
            if "RateLimitReached" in str(exc) or "429" in str(exc):
                # Sit out a full window. The upstream counter may be a tumbling
                # window rather than a sliding one, so a partial wait can land
                # straight back in the same saturated bucket.
                time.sleep(limiter.window)
            else:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"embedding 失敗（重試 {retries} 次）：{last_error}")


def backend() -> str:
    return (getattr(settings, "EMBED_BACKEND", "api") or "api").lower()


def active_model_name() -> str:
    if backend() == "local":
        return getattr(settings, "EMBED_LOCAL_MODEL", "BAAI/bge-m3")
    return settings.EMBED_MODEL


def vector_dimensions() -> int:
    if backend() == "local":
        from core import local_embeddings

        return local_embeddings.dimensions()
    return settings.EMBED_DIMENSIONS


def embed_texts(
    texts: list[str],
    input_type: str = "document",
    concurrency: int | None = None,
    progress=None,
) -> np.ndarray:
    """Embed a list of texts, returning an L2-normalised float32 matrix.

    Normalising at write time makes cosine similarity a plain dot product,
    which keeps the search path a single numpy matmul.
    """
    if backend() == "local":
        from core import local_embeddings

        return local_embeddings.embed_texts(texts, input_type, progress=progress)

    if not texts:
        return np.zeros((0, settings.EMBED_DIMENSIONS), dtype=np.float32)

    texts = [t[:MAX_CHARS_PER_INPUT] for t in texts]
    index_batches = plan_batches(texts)
    results: list[list[list[float]]] = [None] * len(index_batches)  # type: ignore[list-item]

    done = 0
    lock = threading.Lock()

    def run(i_batch):
        nonlocal done
        i, idxs = i_batch
        results[i] = _embed_batch([texts[j] for j in idxs], input_type)
        with lock:
            done += len(idxs)
            if progress:
                progress(done, len(texts))

    workers = concurrency or settings.EMBED_CONCURRENCY
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, enumerate(index_batches)))
    else:
        for item in enumerate(index_batches):
            run(item)

    flat = [vec for batch in results for vec in batch]
    matrix = np.asarray(flat, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query (asymmetric: uses input_type='query')."""
    return embed_texts([text], input_type="query", concurrency=1)[0]
