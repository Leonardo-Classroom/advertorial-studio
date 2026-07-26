"""The portal: what someone who just wants a draft sees.

Deliberately narrower than the staff tooling under /manage/. A portal user
picks a target style and gets a draft; retrieval strategy, generation mode,
rewrite budget and the whole evaluation apparatus stay out of sight, set to
the values the experiments settled on. Exposing those knobs here would ask
users to make calls that took a dozen A/B runs to answer.

Everything is scoped to `request.user`: a portal user sees only their own
briefs and drafts. Staff see everything through /manage/ instead.
"""
from __future__ import annotations

import json

from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.text import slugify

from briefs.models import Brief
from briefs.services import ppt_extract
from corpus.models import StyleGuide
from studio.models import GenerationRun
from studio.services import generate as generate_service


def _my_briefs(request):
    return Brief.objects.filter(owner=request.user)


def _my_runs(request):
    return GenerationRun.objects.filter(owner=request.user)


@login_required
def home(request):
    return render(request, "portal/home.html", {
        "nav": "home",
        "briefs": _my_briefs(request)[:10],
        "runs": _my_runs(request).select_related("brief", "outlet")[:10],
        "brief_count": _my_briefs(request).count(),
        "run_count": _my_runs(request).count(),
    })


@login_required
def upload(request):
    if request.method == "POST":
        upload_file = request.FILES.get("source_file")
        if not upload_file:
            messages.error(request, "請選擇一個 .pptx 檔案。")
            return redirect("portal:upload")
        if not upload_file.name.lower().endswith(".pptx"):
            messages.error(request, "只支援 .pptx 格式。")
            return redirect("portal:upload")

        brief = Brief.objects.create(
            owner=request.user,
            title=(request.POST.get("title") or "").strip() or upload_file.name.rsplit(".", 1)[0],
            source_file=upload_file,
        )
        try:
            raw, slides = ppt_extract.extract_text(brief.source_file.path)
            brief.raw_text = raw
            brief.slide_count = slides
            brief.save(update_fields=["raw_text", "slide_count"])
        except Exception as exc:  # noqa: BLE001 - the operator needs the reason
            messages.error(request, f"讀取簡報失敗：{exc}")
            return redirect("portal:brief_detail", pk=brief.pk)

        # Extract immediately: making the user press a second button to get
        # anything out of their own upload is friction with no upside.
        try:
            brief.facts = ppt_extract.extract_facts(raw)
            brief.status = "extracted"
            brief.save(update_fields=["facts", "status", "updated_at"])
        except Exception as exc:  # noqa: BLE001
            messages.warning(request, f"已讀取 {slides} 張投影片，但自動抽取失敗：{exc}。可在下方手動重試。")
            return redirect("portal:brief_detail", pk=brief.pk)

        messages.success(request, f"已讀取 {slides} 張投影片並抽出內容。請核對後再產稿。")
        return redirect("portal:brief_detail", pk=brief.pk)

    return render(request, "portal/upload.html", {"nav": "upload"})


@login_required
def brief_detail(request, pk):
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    guides = StyleGuide.objects.filter(is_active=True).select_related("outlet", "author")
    return render(request, "portal/brief_detail.html", {
        "nav": "briefs",
        "brief": brief,
        "facts_json": json.dumps(brief.facts or {}, ensure_ascii=False, indent=2),
        "uncertain": (brief.facts or {}).get("uncertain") or [],
        "guides": guides,
        "runs": brief.runs.filter(owner=request.user),
    })


@login_required
def brief_extract(request, pk):
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)
    if not brief.raw_text.strip():
        messages.error(request, "這份簡報沒有可讀取的文字內容。")
        return redirect("portal:brief_detail", pk=pk)
    try:
        brief.facts = ppt_extract.extract_facts(brief.raw_text)
        brief.status = "extracted"
        brief.save(update_fields=["facts", "status", "updated_at"])
        messages.success(request, "已重新抽取，請核對。")
    except Exception as exc:  # noqa: BLE001
        messages.error(request, f"抽取失敗：{exc}")
    return redirect("portal:brief_detail", pk=pk)


@login_required
def brief_save_facts(request, pk):
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)
    try:
        brief.facts = json.loads(request.POST.get("facts", "").strip())
    except json.JSONDecodeError as exc:
        messages.error(request, f"格式錯誤，未儲存：{exc}")
        return redirect("portal:brief_detail", pk=pk)

    brief.status = "confirmed" if request.POST.get("confirm") == "1" else brief.status
    brief.save(update_fields=["facts", "status", "updated_at"])
    messages.success(request, "已確認，可以產稿了。" if brief.status == "confirmed" else "已儲存。")
    return redirect("portal:brief_detail", pk=pk)


@login_required
def generate(request, pk):
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)

    guide = get_object_or_404(StyleGuide.objects.filter(is_active=True),
                              pk=request.POST.get("style_guide"))
    review_outline = request.POST.get("review_outline") == "on"

    run = GenerationRun.objects.create(
        owner=request.user,
        brief=brief,
        outlet=guide.outlet,
        author=guide.author,
        style_guide=guide,
        # Fixed to what the experiments settled on. Not user-facing choices.
        mode="staged",
        retrieval_strategy="typical",
        exemplar_count=4,
        max_rewrites=1,
    )
    generate_service.run_generation(run, stop_after_outline=review_outline)

    if run.status == "failed":
        messages.error(request, f"產稿失敗：{run.error}")
    elif review_outline:
        messages.success(request, "大綱已產出。確認方向後再按「依大綱寫出正文」。")
    else:
        messages.success(request, "稿件已產出。")
    return redirect("portal:draft_detail", pk=run.pk)


@login_required
def draft_detail(request, pk):
    run = get_object_or_404(_my_runs(request).select_related("brief", "outlet", "style_guide"), pk=pk)
    return render(request, "portal/draft_detail.html", {
        "nav": "drafts",
        "run": run,
        "revisions": run.revisions.filter(accepted=True),
    })


@login_required
def draft_outline(request, pk):
    run = get_object_or_404(_my_runs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:draft_detail", pk=pk)

    run.outline = request.POST.get("outline", run.outline)
    run.outline_approved = True
    run.save(update_fields=["outline", "outline_approved"])

    if request.POST.get("action") == "generate":
        from studio.services import pipeline

        pipeline.continue_from_outline(run)
        if run.status == "failed":
            messages.error(request, f"產稿失敗：{run.error}")
        else:
            messages.success(request, "已依大綱寫出正文。")
    else:
        messages.success(request, "大綱已儲存。")
    return redirect("portal:draft_detail", pk=pk)


@login_required
def draft_revise(request, pk):
    run = get_object_or_404(_my_runs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:draft_detail", pk=pk)

    feedback = (request.POST.get("feedback") or "").strip()
    if not feedback:
        messages.error(request, "請寫下要修改的地方。")
        return redirect("portal:draft_detail", pk=pk)

    revision = generate_service.run_revision(run, feedback)
    messages.success(request, f"已依你的意見產出第 {revision.round} 稿。")
    return redirect("portal:draft_detail", pk=pk)


@login_required
def draft_download(request, pk):
    run = get_object_or_404(_my_runs(request), pk=pk)
    name = slugify(run.brief.title) or f"draft-{run.pk}"
    response = HttpResponse(run.latest_text, content_type="text/markdown; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{name}-{run.pk}.md"'
    return response


@login_required
def change_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  # keep the session alive
            messages.success(request, "密碼已更新。")
            return redirect("portal:home")
    else:
        form = PasswordChangeForm(request.user)
    return render(request, "portal/change_password.html", {"nav": "", "form": form})
