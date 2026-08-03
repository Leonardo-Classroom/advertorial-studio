"""Run generations off the request thread, a couple at a time.

Generation used to happen inside the POST that asked for it: 1.5–3 minutes of
the browser sitting on a submitted form, and a page that could not be left. Now
the request returns as soon as the run row exists and a worker thread does the
work, which is what makes the new tab possible — and, with it, makes a second
click possible while the first is still going.

That is the reason for the limit. Nothing stops someone pressing 產出稿件 five
times; without a gate that is five concurrent model calls, five times the spend,
and five threads writing to the same SQLite file. Beyond `max_parallel_runs` the
extra ones wait their turn as `pending`, which the interface shows as 排隊中.

Threads, not a job queue. At this scale Celery and Redis would be two more
things to run and monitor for a workload of a few drafts a day. The cost is that
a server restart kills anything in flight — so a run records when it started,
and one that has been running impossibly long is marked failed rather than left
spinning on someone's screen forever.
"""
from __future__ import annotations

import threading
from datetime import timedelta

from django.db import connections
from django.utils import timezone

# A run cannot legitimately take this long: the slowest observed staged run with
# a rewrite was around 32 minutes on a cold local embedding model, and the model
# calls themselves time out well before it.
STALE_AFTER = timedelta(minutes=45)

_gate = threading.Condition()
_active = 0


def _limit() -> int:
    """Read fresh each time, so changing it in 高級 takes effect immediately."""
    from studio.models import SiteSettings

    try:
        return max(1, SiteSettings.load().max_parallel_runs)
    except Exception:  # noqa: BLE001 - a broken settings row must not block work
        return 1


def _acquire() -> None:
    global _active
    with _gate:
        while _active >= _limit():
            _gate.wait()
        _active += 1


def _release() -> None:
    global _active
    with _gate:
        _active -= 1
        _gate.notify()


def active_count() -> int:
    with _gate:
        return _active


def _work(run_id: int) -> None:
    from studio.models import GenerationRun
    from studio.services import generate as generate_service

    _acquire()
    try:
        run = GenerationRun.objects.filter(pk=run_id).first()
        if run is None or run.status != "pending":
            return                      # cancelled, deleted, or already picked up
        run.started_at = timezone.now()
        run.save(update_fields=["started_at"])
        generate_service.run_generation(run)
        if run.status == "done":
            from studio.services import drafts

            drafts.ensure_first_version(run)
    except Exception as exc:  # noqa: BLE001 - a thread that dies silently leaves
        # the run stuck on 生成中 with nothing to explain it.
        GenerationRun.objects.filter(pk=run_id).update(
            status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        _release()
        # Worker threads get their own database connection; leaving them open
        # holds SQLite handles for the life of the process.
        connections.close_all()


def submit(run: "object") -> None:
    """Queue a run. Returns immediately; the row carries the outcome."""
    threading.Thread(target=_work, args=(run.pk,), daemon=True,
                     name=f"generate-{run.pk}").start()


def _rewrite(run_id: int, base_id: int, feedback: str) -> None:
    from studio.models import DraftVersion, GenerationRun
    from studio.services import drafts
    from studio.services import generate as generate_service

    _acquire()
    try:
        run = GenerationRun.objects.filter(pk=run_id).first()
        base = DraftVersion.objects.filter(pk=base_id).first()
        if run is None or base is None:
            return
        revision = generate_service.run_revision(run, feedback, previous_text=base.text)
        drafts.add_version(run, revision.output, source="revise",
                           user_input=feedback, parent=base)
        GenerationRun.objects.filter(pk=run_id).update(status="done")
    except Exception as exc:  # noqa: BLE001 - see _work
        GenerationRun.objects.filter(pk=run_id).update(
            status="failed", error=f"重寫失敗：{type(exc).__name__}: {exc}")
    finally:
        _release()
        connections.close_all()


def submit_rewrite(run, base, feedback: str) -> None:
    """Queue a rewrite of one draft version, against the user's own notes."""
    run.status = "running"
    run.started_at = timezone.now()
    run.error = ""
    run.save(update_fields=["status", "started_at", "error"])
    threading.Thread(target=_rewrite, args=(run.pk, base.pk, feedback), daemon=True,
                     name=f"rewrite-{run.pk}").start()


def reap_stale(runs) -> None:
    """Mark as failed anything that cannot still be running.

    Called from the pages that display run status, because there is no other
    heartbeat: a thread killed by a restart never gets to update its own row.
    """
    cutoff = timezone.now() - STALE_AFTER
    for run in runs:
        if not run.in_progress:
            continue
        started = run.started_at or run.created_at
        if started < cutoff:
            run.status = "failed"
            run.error = run.error or "產稿中斷（可能是伺服器重新啟動）。請重新產稿。"
            run.save(update_fields=["status", "error"])
