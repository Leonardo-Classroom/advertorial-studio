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

**A bounded queue with a worker pool, not a thread per request.** The earlier
version started a daemon thread for every submission and let the surplus park on
a condition variable. The gate held concurrency to `max_parallel_runs`, but
nothing held the *thread count*: N clicks meant N threads, each with its own 8MB
stack, so a user who kept pressing the button could turn a few hundred requests
into gigabytes of parked stacks (任務一 發現 2). Now submissions go into a queue
of bounded depth and at most `max_parallel_runs` workers ever exist. A queue that
is genuinely full is refused at the door, where the user can be told, rather than
absorbed silently until the process dies.

Threads, not a job queue service. At this scale Celery and Redis would be two
more things to run and monitor for a workload of a few drafts a day. The cost is
that a server restart kills anything in flight and empties the queue — so a run
records when it started, and one that cannot still be running is marked failed
rather than left spinning on someone's screen forever (`reap_stale`).
"""
from __future__ import annotations

import queue
import threading
from datetime import timedelta

from django.db import connections
from django.utils import timezone

# How long a run may look alive before it is presumed dead. Derived from the
# generation timeout rather than fixed beside it (`core.timeouts`): the two
# have to stay consistent, and the only way to guarantee that is to compute
# one from the other. Defaults to the 45 minutes it has always been — the
# slowest observed staged run with a rewrite took about 32 minutes — and grows
# only if someone raises the timeout it depends on.
def _stale_after() -> timedelta:
    from core import timeouts

    return timeouts.generation_stale()

# How many submissions may wait. At the observed 30-120s per run this is already
# hours of backlog, which is well past the point where the honest answer is
# "come back later" rather than a queue position nobody will wait out. The point
# of the number is that one exists: memory now grows with the queue (one small
# tuple per entry) instead of with parked thread stacks.
QUEUE_MAX = 100

# How long a worker waits for something to do before retiring. Long enough to
# stay warm across the gaps in a working session, short enough that an idle
# process is not holding threads open all night.
WORKER_IDLE_TIMEOUT = 60.0

_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=QUEUE_MAX)

# Guards `_workers`, and pairs the "am I surplus?" check with the decrement so a
# retiring worker cannot race a submission into leaving the queue unattended.
_pool_lock = threading.Lock()
_workers = 0

_active_lock = threading.Lock()
_active = 0


def _limit() -> int:
    """Read fresh each time, so changing it in 高級 does not need a restart.

    Raising it takes effect on the next submission (`_ensure_workers` tops the
    pool up); lowering it takes effect as each worker finishes its current job
    and finds itself surplus. Neither interrupts work already in flight.

    Always called *outside* `_pool_lock`: this is a database query, and the old
    version ran it inside the gate's wait loop, so every wakeup did DB I/O while
    holding a threading lock (任務一 發現 4).
    """
    from studio.models import SiteSettings

    try:
        return max(1, SiteSettings.load().max_parallel_runs)
    except Exception:  # noqa: BLE001 - a broken settings row must not block work
        return 1


def _ensure_workers() -> None:
    """Top the pool up to the current limit. Cheap and idempotent."""
    global _workers

    limit = _limit()
    with _pool_lock:
        while _workers < limit:
            _workers += 1
            threading.Thread(target=_worker_loop, daemon=True,
                             name=f"run-worker-{_workers}").start()


def _worker_loop() -> None:
    """Take tasks until the queue goes quiet or this worker becomes surplus."""
    global _workers

    while True:
        try:
            task = _queue.get(timeout=WORKER_IDLE_TIMEOUT)
        except queue.Empty:
            with _pool_lock:
                # Re-check under the lock: a submission may have landed between
                # the timeout firing and this point, and `_ensure_workers` may
                # already have decided the pool was full and declined to spawn.
                if _queue.empty():
                    _workers -= 1
                    return
            continue

        try:
            _run_task(task)
        finally:
            _queue.task_done()
            # Worker threads get their own database connection; leaving them
            # open holds SQLite handles for the life of the process.
            connections.close_all()

        # Retire if 高級 lowered the limit while this job was running.
        limit = _limit()
        with _pool_lock:
            if _workers > limit:
                _workers -= 1
                return


def _run_task(task: tuple) -> None:
    global _active

    kind = task[0]
    with _active_lock:
        _active += 1
    try:
        if kind == "generate":
            _work(task[1])
        elif kind == "rewrite":
            _rewrite(task[1], task[2], task[3])
    finally:
        with _active_lock:
            _active -= 1


def _enqueue(task: tuple) -> bool:
    """Put a task in the queue. False means the queue is full — say so."""
    try:
        _queue.put_nowait(task)
    except queue.Full:
        return False
    _ensure_workers()
    return True


def active_count() -> int:
    """Runs executing right now (not counting those waiting in the queue)."""
    with _active_lock:
        return _active


def queue_depth() -> int:
    """Submissions waiting for a worker."""
    return _queue.qsize()


def _work(run_id: int) -> None:
    from studio.models import GenerationRun
    from studio.services import generate as generate_service

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


def submit(run: "object") -> bool:
    """Queue a run. Returns immediately; the row carries the outcome.

    False means the queue was full and nothing was scheduled — the caller has to
    tell the user, because the run row on its own would sit at 排隊中 forever.
    """
    return _enqueue(("generate", run.pk))


def _rewrite(run_id: int, base_id: int, feedback: str) -> None:
    from studio.models import DraftVersion, GenerationRun
    from studio.services import drafts
    from studio.services import generate as generate_service

    try:
        run = GenerationRun.objects.filter(pk=run_id).first()
        base = DraftVersion.objects.filter(pk=base_id).first()
        if run is None or base is None:
            return
        # Stamped here rather than at submission, matching `_work`: until a
        # worker picks the task up this run is queued, not running, and
        # `reap_stale` measures from `started_at`. Stamping it early meant a
        # rewrite that waited out a busy queue could be declared timed out
        # before it had run for a second (任務一 發現 1).
        run.status = "running"
        run.started_at = timezone.now()
        run.save(update_fields=["status", "started_at"])
        revision = generate_service.run_revision(run, feedback, previous_text=base.text)
        drafts.add_version(run, revision.output, source="revise",
                           user_input=feedback, parent=base)
        GenerationRun.objects.filter(pk=run_id).update(status="done")
    except Exception as exc:  # noqa: BLE001 - see _work
        GenerationRun.objects.filter(pk=run_id).update(
            status="failed", error=f"重寫失敗：{type(exc).__name__}: {exc}")


def submit_rewrite(run, base, feedback: str) -> bool:
    """Queue a rewrite of one draft version, against the user's own notes.

    The run goes back to `pending` — 排隊中 — and the worker promotes it to
    `running` when it actually starts. False means the queue was full; the run
    is left exactly as it was found.
    """
    previous_status = run.status
    run.status = "pending"
    run.started_at = None
    run.error = ""
    run.save(update_fields=["status", "started_at", "error"])
    if _enqueue(("rewrite", run.pk, base.pk, feedback)):
        return True
    run.status = previous_status
    run.save(update_fields=["status"])
    return False


def reap_stale(runs) -> None:
    """Mark as failed anything that cannot still be running.

    Called from the pages that display run status, because there is no other
    heartbeat: a thread killed by a restart never gets to update its own row,
    and a restart also empties the queue under anything still waiting in it.
    """
    cutoff = timezone.now() - _stale_after()
    for run in runs:
        if not run.in_progress:
            continue
        started = run.started_at or run.created_at
        if started < cutoff:
            run.status = "failed"
            run.error = run.error or "產稿中斷（可能是伺服器重新啟動）。請重新產稿。"
            run.save(update_fields=["status", "error"])
