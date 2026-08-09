"""How long a model call may take, and how long a job may look alive.

These numbers used to be scattered as literals across five modules — 600 in
`generate`, 400 in `pipeline`, 240 in `ppt_extract`, 240 and 300 as default
arguments in `facts_update`, 120 in `.env`. Same kind of work, different
numbers, none of them changeable without a deploy, and 任務三's recommendation
("raise fact extraction from 240s") was not actionable because the value was
buried in a function body.

Three families, because three is how many genuinely differ: writing is the
long one, looking at a picture is the short one, and reading a deck sits in
between. Everything within a family now shares a number an operator can change
at /manage/advanced/, with the `.env` value as the fallback for when the
settings row cannot be read.

**The staleness thresholds are derived here too, and that is the point.**
`reap_stale` marks anything older than its threshold as failed, so a threshold
below the time a healthy job may legitimately take turns a slow run into a
false failure. Leaving both sides adjustable would invite exactly that pair.
Deriving one from the other makes the inconsistency unrepresentable.
"""
from __future__ import annotations

from datetime import timedelta

from django.conf import settings

# field on SiteSettings -> (.env setting used as fallback, default)
_SOURCES = {
    "generate": ("GENERATE_TIMEOUT", 600),
    "vision": ("LOCAL_LLM_VISION_TIMEOUT", 120),
    "extract": ("EXTRACT_TIMEOUT", 240),
}
_FIELDS = {
    "generate": "generate_timeout_seconds",
    "vision": "vision_timeout_seconds",
    "extract": "extract_timeout_seconds",
}


def _seconds(kind: str) -> int:
    """One family's timeout, from 高級設定 with `.env` as the fallback.

    Lazy import so `core` carries no load-time dependency on `studio`, the
    same shape `core.llm._backend()` uses.
    """
    env_name, default = _SOURCES[kind]
    try:
        from studio.models import SiteSettings

        value = int(getattr(SiteSettings.load(), _FIELDS[kind]) or 0)
        if value > 0:
            return value
    except Exception:  # noqa: BLE001 - a broken settings row must not stop work
        pass
    return int(getattr(settings, env_name, default) or default)


def generate() -> int:
    """Writing a draft, rewriting one, and the staged pipeline's stages."""
    return _seconds("generate")


def vision() -> int:
    """Classifying one picture."""
    return _seconds("vision")


def extract() -> int:
    """Reading facts out of a deck, correcting them, merging a new file in."""
    return _seconds("extract")


# A staged run makes four model calls that can each approach the timeout —
# notes, outline, body, revision. Retrieval and re-ranking are stages too but
# touch no model and finish in milliseconds.
_GENERATE_CALLS = 4
# One extraction per upload batch; the multiplier is headroom for the merge
# path and for a retry landing inside the same window.
_EXTRACT_CALLS = 3


def _gate_wait() -> float:
    """The longest a call may sit waiting for a free model slot.

    This has to be counted: `started_at` is stamped when a worker picks the job
    up, *before* it queues for the model, so gate waiting happens inside the
    window `reap_stale` measures. Without this term the upload threshold (20
    minutes) sat one minute above the worst legitimate wait (15 minutes of gate
    plus a 4-minute extraction) — a queue one job deeper than expected would
    have started failing healthy uploads.
    """
    from core import llm

    return llm.GATE_MAX_WAIT


def generation_stale() -> timedelta:
    """When a generation run cannot still be running.

    The floor is the empirical one it has always had: the slowest staged run
    with a rewrite observed here took about 32 minutes, and 45 leaves room
    above that. The derived term only takes over if the timeout is raised.
    """
    derived = _gate_wait() + _GENERATE_CALLS * generate()
    return timedelta(seconds=max(45 * 60, derived))


def ingest_stale() -> timedelta:
    """When an upload batch cannot still be processing.

    Much cheaper work than a generation run — a few file parses and one
    extraction — hence a much shorter floor.
    """
    derived = _gate_wait() + _EXTRACT_CALLS * extract()
    return timedelta(seconds=max(20 * 60, derived))
