import json

from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from accounts.decorators import staff_required
from briefs.models import FACT_SCHEMA_HINT, Brief
from briefs.services import ppt_extract


@staff_required
def brief_list(request):
    return render(request, "briefs/list.html", {
        "section": "briefs",
        "briefs": Brief.objects.all(),
    })


@staff_required
def brief_upload(request):
    if request.method == "POST":
        upload = request.FILES.get("source_file")
        title = (request.POST.get("title") or "").strip()
        if not upload:
            messages.error(request, "請選擇一個 .pptx 檔案。")
            return redirect("briefs:upload")

        brief = Brief.objects.create(
            owner=request.user,
            title=title or upload.name.rsplit(".", 1)[0],
            source_file=upload,
            parse_images=request.POST.get("parse_images") == "1",
        )
        try:
            raw, slides = ppt_extract.extract_text(brief.source_file.path)
            brief.raw_text = raw
            brief.slide_count = slides
            brief.save(update_fields=["raw_text", "slide_count"])
            messages.success(
                request,
                f"已讀取 {slides} 張投影片。接著按「用 AI 抽取事實」，"
                "抽完務必人工核對——這份事實是之後所有稿件唯一的事實來源。",
            )
        except Exception as exc:  # noqa: BLE001 - show the operator what broke
            messages.error(request, f"解析簡報失敗：{exc}")
        return redirect("briefs:detail", pk=brief.pk)

    return render(request, "briefs/upload.html", {"section": "briefs"})


@staff_required
def brief_detail(request, pk):
    brief = get_object_or_404(Brief, pk=pk)
    return render(request, "briefs/detail.html", {
        "section": "briefs",
        "brief": brief,
        "facts_json": json.dumps(brief.facts or FACT_SCHEMA_HINT, ensure_ascii=False, indent=2),
        "brief_images": brief.images.all(),
    })


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
    brief = get_object_or_404(Brief, pk=pk)
    if request.method != "POST":
        return redirect("briefs:detail", pk=pk)

    if not brief.raw_text.strip():
        messages.error(request, "這份簡報沒有可用的文字內容，無法抽取。")
        return redirect("briefs:detail", pk=pk)

    try:
        facts = ppt_extract.extract_facts(brief.raw_text)
    except Exception as exc:  # noqa: BLE001
        messages.error(request, f"抽取失敗：{exc}")
        return redirect("briefs:detail", pk=pk)

    if "_raw" in facts:
        messages.warning(request, "模型沒有回傳結構化 JSON，已保留原始輸出供你手動整理。")
    brief.add_facts_version(facts, source="extract")

    uncertain = facts.get("uncertain") or []
    if uncertain:
        messages.warning(request, f"模型標記了 {len(uncertain)} 處不確定的地方，請往下捲動確認。")
    messages.success(request, "已抽取。請逐項核對後再按「確認事實」。")

    # The upload asked for pictures; the brand needed to judge them only exists
    # now, so this is the earliest point the request can honestly be served.
    if brief.parse_images and not brief.images.exists():
        _ingest_images(request, brief)
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
