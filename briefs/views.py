import json

from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from accounts.decorators import staff_required
from briefs.models import FACT_SCHEMA_HINT, Brief
from briefs.services import ingest_runner
from briefs.services import upload as upload_service


@staff_required
def brief_list(request):
    return render(request, "briefs/list.html", {
        "section": "briefs",
        "briefs": Brief.objects.all(),
    })


@staff_required
def brief_upload(request):
    """Same multi-file, async path the portal uses — see `briefs/services/upload.py`.

    Staff tooling used to keep its own copy of "make a Brief, save the file,
    parse it", which only ever drifted from the portal's version. Both now
    call the one shared implementation and differ only in where they redirect.
    """
    if request.method == "POST":
        files = request.FILES.getlist("source_files")
        try:
            brief = upload_service.create_brief(
                owner=request.user,
                title=request.POST.get("title") or "",
                files=files,
                parse_images=request.POST.get("parse_images") == "1",
            )
        except ValueError as exc:  # noqa: BLE001 - the operator needs the reason
            messages.error(request, str(exc))
            return redirect("briefs:upload")
        return redirect("briefs:detail", pk=brief.pk)

    return render(request, "briefs/upload.html", {"section": "briefs"})


@staff_required
def brief_detail(request, pk):
    brief = get_object_or_404(Brief, pk=pk)
    source_files = list(brief.source_files.order_by("order", "pk"))
    ingest_runner.reap_stale(source_files)
    ingest_runner.reap_stale_brief(brief)
    return render(request, "briefs/detail.html", {
        "section": "briefs",
        "brief": brief,
        "source_files": source_files,
        "files_in_progress": brief.processing,
        "facts_json": json.dumps(brief.facts or FACT_SCHEMA_HINT, ensure_ascii=False, indent=2),
        "brief_images": brief.images.order_by("source_file_id", "slide_index", "pk"),
    })


@staff_required
def brief_upload_status(request, pk):
    """Where an upload/retry batch has got to — polled by the waiting card."""
    brief = get_object_or_404(Brief, pk=pk)
    return JsonResponse(upload_service.status_payload(brief))


def _ingest_images(request, brief) -> None:
    """Parse the deck's pictures and report what survived the filter."""
    from briefs.services import images as image_service

    try:
        summary = image_service.ingest(brief)
    except Exception as exc:  # noqa: BLE001 - the text path already succeeded
        messages.warning(request, f"圖片解析失敗：{exc}（文字內容不受影響）")
        return

    if not summary["stored"]:
        messages.info(request, "沒有找到可用的圖片素材（多半是表格、logo 或裝飾圖）。")
        return

    messages.success(
        request,
        f"圖片解析完成：簡報共 {summary['found']} 張圖，"
        f"規則過濾後剩 {summary['kept']} 張"
        f"（重複 {summary['duplicate']}、近似重複 {summary['near_duplicate']}、"
        f"向量圖 {summary['vector']}、過小 {summary['too_small']}），"
        f"送辨識 {summary['stored']} 張，判定可用 {summary['usable']} 張。請逐張核對。")


@staff_required
def brief_images(request, pk):
    """Re-parse pictures, or record the operator's approvals."""
    brief = get_object_or_404(Brief, pk=pk)
    if request.method != "POST":
        return redirect("briefs:detail", pk=pk)

    if request.POST.get("action") == "parse":
        if not brief.facts:
            messages.error(request, "請先抽取事實再解析圖片——分類需要知道品牌是誰，才擋得掉競品照。")
            return redirect("briefs:detail", pk=pk)
        brief.parse_images = True
        brief.save(update_fields=["parse_images", "updated_at"])
        _ingest_images(request, brief)
        return redirect("briefs:detail", pk=pk)

    approved = set(request.POST.getlist("approved"))
    for image in brief.images.all():
        key = str(image.pk)
        image.approved = key in approved
        image.caption = (request.POST.get(f"caption_{key}") or "").strip()[:200]
        image.save(update_fields=["approved", "caption"])

    # The review page saves each change as it happens; answer it quietly.
    if request.headers.get("X-Requested-With") == "fetch":
        return HttpResponse(status=204)

    messages.success(request, f"已確認 {len(approved)} 張圖片可用於稿件。")
    return redirect("briefs:detail", pk=pk)


@staff_required
def brief_extract(request, pk):
    """Re-run fact extraction (and image ingest) without re-parsing files.

    Runs in the background via `ingest_runner.retry_facts`, same as the
    portal's retry — the detail page's waiting card picks up the progress.
    """
    brief = get_object_or_404(Brief, pk=pk)
    if request.method != "POST":
        return redirect("briefs:detail", pk=pk)

    if not brief.source_files.filter(status="done").exists():
        messages.error(request, "這份簡報沒有可用的文字內容，無法抽取。")
        return redirect("briefs:detail", pk=pk)

    ingest_runner.retry_facts(brief)
    messages.info(request, "已在背景重新抽取，完成後這一頁會自動更新。")
    return redirect("briefs:detail", pk=pk)


@staff_required
def brief_save_facts(request, pk):
    brief = get_object_or_404(Brief, pk=pk)
    if request.method != "POST":
        return redirect("briefs:detail", pk=pk)

    raw = request.POST.get("facts", "").strip()
    try:
        facts = json.loads(raw)
    except json.JSONDecodeError as exc:
        messages.error(request, f"JSON 格式錯誤，未儲存：{exc}")
        return redirect("briefs:detail", pk=pk)

    # Staff edits are versioned too. The portal picks which version a draft is
    # written from, so a hand edit that silently replaced the current facts
    # would leave those drafts pointing at a version that no longer says what
    # they were written from.
    version = brief.add_facts_version(facts, source="staff_edit")
    if request.POST.get("confirm") == "1":
        brief.status = "confirmed"
        brief.save(update_fields=["status", "updated_at"])
        messages.success(request, f"事實已確認，目前是 {version.label}。")
    else:
        messages.success(request, f"已儲存為 {version.label}。")
    return redirect("briefs:detail", pk=pk)
