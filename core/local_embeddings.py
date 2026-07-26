"""Local (on-GPU) embedding backend.

Why this exists: the Azure Cohere deployment is on a free tier capped at
50 requests per day, which works out to ~850 articles/day — 20 days for
COOL-STYLE alone and 54 days for the full corpus. That makes topical retrieval
untestable in practice. A local model has no quota at all and, on the RTX 4090
in this machine, embeds the whole corpus in minutes.

BGE-M3 is the default because this corpus is genuinely mixed-language: Chinese
prose densely interleaved with English brand and model names ("sacai x Nike
LDWaffle", "204L"). A Chinese-only model would embed those tokens poorly, and
they carry a lot of the topical signal.

Unlike the earlier BGE v1.5 generation, M3 needs no query instruction prefix,
so `input_type` has no effect here — the API is kept identical to the remote
backend only so callers do not have to care which one is active.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from django.conf import settings


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    name = getattr(settings, "EMBED_LOCAL_MODEL", "BAAI/bge-m3")
    device = getattr(settings, "EMBED_LOCAL_DEVICE", "cuda")
    try:
        return SentenceTransformer(name, device=device)
    except Exception:
        # A machine without a working CUDA runtime should still be able to
        # index, just slower — failing outright would be worse.
        if device != "cpu":
            return SentenceTransformer(name, device="cpu")
        raise


def dimensions() -> int:
    return int(get_model().get_sentence_embedding_dimension())


def embed_texts(
    texts: list[str],
    input_type: str = "document",  # noqa: ARG001 - kept for API symmetry
    progress=None,
) -> np.ndarray:
    """Embed texts locally, returning an L2-normalised float32 matrix."""
    if not texts:
        return np.zeros((0, dimensions()), dtype=np.float32)

    model = get_model()
    batch = int(getattr(settings, "EMBED_LOCAL_BATCH", 64))
    out: list[np.ndarray] = []

    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        vectors = model.encode(
            chunk,
            batch_size=batch,
            normalize_embeddings=True,   # cosine similarity becomes a dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out.append(np.asarray(vectors, dtype=np.float32))
        if progress:
            progress(min(start + batch, len(texts)), len(texts))

    return np.vstack(out)
