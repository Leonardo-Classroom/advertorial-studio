import json

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render

from briefs.models import FACT_SCHEMA_HINT, Brief
from briefs.services import ppt_extract


def brief_list(request):
    return render(request, "briefs/list.html", {
        "section": "briefs",
        "briefs": Brief.objects.all(),
    })


def brief_upload(request):
    if request.method == "POST":
        upload = request.FILES.get("source_file")
        title = (request.POST.get("title") or "").strip()
        if not upload:
            messages.error(request, "請選擇一個 .pptx 檔案。")
            return redirect("briefs:upload")

        brief = Brief.objects.create(
            title=title or upload.name.rsplit(".", 1)[0],
            source_file=upload,
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


def brief_detail(request, pk):
    brief = get_object_or_404(Brief, pk=pk)
    return render(request, "briefs/detail.html", {
        "section": "briefs",
        "brief": brief,
        "facts_json": json.dumps(brief.facts or FACT_SCHEMA_HINT, ensure_ascii=False, indent=2),
    })


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
    brief.facts = facts
    brief.status = "extracted"
    brief.save(update_fields=["facts", "status", "updated_at"])

    uncertain = facts.get("uncertain") or []
    if uncertain:
        messages.warning(request, f"模型標記了 {len(uncertain)} 處不確定的地方，請往下捲動確認。")
    messages.success(request, "已抽取。請逐項核對後再按「確認事實」。")
    return redirect("briefs:detail", pk=pk)


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

    brief.facts = facts
    if request.POST.get("confirm") == "1":
        brief.status = "confirmed"
        messages.success(request, "事實已確認。現在可以開始生成廣編稿。")
    else:
        messages.success(request, "已儲存。")
    brief.save(update_fields=["facts", "status", "updated_at"])
    return redirect("briefs:detail", pk=pk)
