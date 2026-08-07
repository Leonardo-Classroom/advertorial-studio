"""Process an uploaded Brief's source files off the request thread.

Structural twin of `studio/services/runner.py`, one level simpler: an upload
batch is a handful of file parses plus one fact-extraction call, not a
multi-stage generation pipeline, so there is no concurrency gate here — every
batch just gets its own thread. If upload volume ever makes that matter, the
gate in `runner.py` (`_acquire`/`_release`/`SiteSettings.max_parallel_runs`)
is the pattern to copy in, not reinvent.

A batch is one Brief's worth of `BriefSourceFile` rows, processed in `order`.
One file's failure is recorded on its own row and does not stop the rest — the
same principle `briefs/services/images.py` already applies one level down, to
individual pictures within a single file.
"""
from __future__ import annotations

import threading
from datetime import timedelta

from django.db import connections
from django.utils import timezone

# A file parse plus one facts call — much cheaper than a generation run, so a
# much shorter budget than `runner.py`'s 45 minutes before something is
# declared stuck rather than merely slow.
STALE_AFTER = timedelta(minutes=20)


def _mark_started(brief) -> None:
    """Flag the brief as working, and reset the clock `reap_stale_brief` reads.

    `updated_at` has to move by hand: `auto_now` fires on `save()`, not on the
    `update()` used here, and without it a retry on a brief last touched days
    ago is declared stale by the very first poll — the waiting card vanishes
    and announces a timeout while the thread it was watching runs on happily.
    """
    type(brief).objects.filter(pk=brief.pk).update(
        processing=True, updated_at=timezone.now())


def _process_one(source_file) -> None:
    from briefs.services import source_extract

    source_file.status = "processing"
    source_file.started_at = timezone.now()
    source_file.save(update_fields=["status", "started_at"])
    try:
        text, count = source_extract.extract_text(source_file.file.path, source_file.format)
        source_file.raw_text = text
        source_file.page_count = count
        source_file.status = "done"
        source_file.save(update_fields=["raw_text", "page_count", "status", "updated_at"])
    except Exception as exc:  # noqa: BLE001 - one file must not abort the batch
        source_file.status = "failed"
        source_file.error = f"{type(exc).__name__}: {exc}"
        source_file.save(update_fields=["status", "error", "updated_at"])


def _extract_facts_and_images(brief) -> None:
    from briefs.services import images as image_service
    from briefs.services import ppt_extract

    brief.recompute_from_source_files()
    if not brief.source_files.filter(status="done").exists():
        return  # every file failed; only stub markers, nothing to extract from
    facts = ppt_extract.extract_facts(brief.raw_text)
    brief.add_facts_version(facts, source="extract")
    # Pictures are an extra, and their failure is not the batch's failure: the
    # facts are already saved and a draft can be written without illustrations.
    # Letting this raise would report a fully usable brief as 批次處理失敗.
    #
    # Extraction/filtering is local and cheap, so it always runs — the operator
    # should see what pictures the deck has regardless of whether they asked
    # for AI judgement on them. `parse_images` now only decides whether the
    # (slow, costs a vision call per picture) classification pass runs right
    # away; skipped here, it stays available later via image_service.classify_missing.
    try:
        image_service.ingest(brief, run_classify=brief.parse_images)
    except Exception as exc:  # noqa: BLE001 - see above
        type(brief).objects.filter(pk=brief.pk).update(
            note=f"圖片解析失敗：{type(exc).__name__}: {exc}（文字內容不受影響，稿子仍可正常產出）")


def _work(brief_id: int) -> None:
    from briefs.models import Brief

    try:
        brief = Brief.objects.filter(pk=brief_id).first()
        if brief is None:
            return  # deleted while queued
        for source_file in brief.source_files.order_by("order", "pk"):
            _process_one(source_file)
        _extract_facts_and_images(brief)
    except Exception as exc:  # noqa: BLE001 - a thread that dies silently leaves
        # every file stuck on 處理中 with nothing to explain it.
        from briefs.models import Brief as BriefModel

        BriefModel.objects.filter(pk=brief_id).update(
            note=f"批次處理失敗：{type(exc).__name__}: {exc}")
    finally:
        from briefs.models import Brief as BriefModel

        BriefModel.objects.filter(pk=brief_id).update(processing=False)
        connections.close_all()


def submit(brief) -> None:
    """Queue a brief's source files for processing. Returns immediately.

    `processing` is flipped before the thread starts, not inside it — the
    caller redirects to a page that polls this flag right away, and a race
    where that first poll lands before the thread has run at all would show
    the batch as finished when it has not even begun.
    """
    _mark_started(brief)
    threading.Thread(target=_work, args=(brief.pk,), daemon=True,
                     name=f"ingest-{brief.pk}").start()


def _retry_facts(brief_id: int) -> None:
    from briefs.models import Brief

    try:
        brief = Brief.objects.filter(pk=brief_id).first()
        if brief is None:
            return
        _extract_facts_and_images(brief)
    except Exception as exc:  # noqa: BLE001 - see _work
        from briefs.models import Brief as BriefModel

        BriefModel.objects.filter(pk=brief_id).update(
            note=f"重新抽取失敗：{type(exc).__name__}: {exc}")
    finally:
        from briefs.models import Brief as BriefModel

        BriefModel.objects.filter(pk=brief_id).update(processing=False)
        connections.close_all()


def retry_facts(brief) -> None:
    """Re-run fact extraction (and image ingest) without re-parsing files.

    For the case where every file parsed fine but the `extract_facts()` call
    itself failed — there is nothing wrong with the files, so reprocessing
    them would just spend the parse cost again for no reason.
    """
    _mark_started(brief)
    threading.Thread(target=_retry_facts, args=(brief.pk,), daemon=True,
                     name=f"retry-facts-{brief.pk}").start()


def reap_stale(source_files) -> None:
    """Mark as failed anything that cannot still be processing.

    Called from the pages that display upload status, because there is no
    other heartbeat: a thread killed by a server restart never gets to update
    its own row.
    """
    cutoff = timezone.now() - STALE_AFTER
    for source_file in source_files:
        if not source_file.in_progress:
            continue
        started = source_file.started_at or source_file.created_at
        if started < cutoff:
            source_file.status = "failed"
            source_file.error = (source_file.error
                                 or "處理中斷（可能是伺服器重新啟動）。請重新上傳這個檔案。")
            source_file.save(update_fields=["status", "error"])


def reap_stale_brief(brief) -> None:
    """Clear a stuck `processing` flag left behind by a killed worker thread.

    Per-file `reap_stale` covers a file stuck mid-parse; this covers the other
    gap — every file already terminal, but the one trailing `extract_facts()`
    call never got to flip `processing` back off because the thread died with
    the process. `updated_at` is the proxy for "last time this row moved
    forward": nothing touches it between `submit()`/`retry_facts()` setting
    `processing=True` and the background job's own next write.
    """
    if not brief.processing:
        return
    if any(f.in_progress for f in brief.source_files.all()):
        return  # a file is still genuinely working; not stuck yet
    if brief.updated_at >= timezone.now() - STALE_AFTER:
        return
    brief.processing = False
    brief.note = brief.note or "處理逾時中斷（可能是伺服器重新啟動）。請重新整理後再試一次。"
    brief.save(update_fields=["processing", "note", "updated_at"])
