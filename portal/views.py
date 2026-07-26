"""The portal: what someone who just wants a draft sees.

Narrower than the staff tooling under /manage/, but not by much: the portal
exposes the target style, retrieval strategy and rewrite budget, while the
generation mode and the whole evaluation apparatus stay behind /manage/.

Every exposed control carries what the experiments actually measured, so a
choice is never made blind — retrieval strategy made no measurable difference
across a dozen A/B runs, and a second rewrite was accepted zero times out of
twelve. Values arriving from the form are validated and clamped rather than
trusted.

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
from studio.models import SELECTABLE_STRATEGIES, GenerationRun
from studio.services import generate as generate_service


# The feedback-driven rewrite is not finished, so the portal does not offer it
# yet. Both the form and the view read this: a disabled button is only a hint to
# the browser, and the URL stays reachable by hand or from a tab opened before
# the change, so the refusal has to exist server-side to mean anything. Staff
# keep the loop under /manage/, which is where it is being worked on.
# Flip to True to release it — the form re-enables with it.
REVISION_ENABLED = False


def _my_briefs(request):
    return Brief.objects.filter(owner=request.user)


def _ingest_images(request, brief) -> None:
    """Parse the deck's pictures, reporting what survived filtering.

    The counts are worth showing rather than hiding: a deck yields hundreds of
    embedded images and only a handful are article material, so "12 張裡有 6 張可用"
    tells the user the filter worked instead of leaving them wondering why their
    358-picture deck produced six thumbnails.
    """
    from briefs.services import images as image_service

    try:
        summary = image_service.ingest(brief)
    except Exception as exc:  # noqa: BLE001 - text extraction already succeeded
        messages.warning(request, f"圖片解析失敗：{exc}。文字內容不受影響，稿子仍可正常產出。")
        return

    if not summary["stored"]:
        messages.info(request, "簡報裡沒有找到適合放進稿子的圖片（多半是表格、logo 或裝飾圖）。")
        return

    note = (f"圖片解析完成：{summary['found']} 張中篩出 {summary['stored']} 張候選，"
            f"其中 {summary['usable']} 張判定可用。請在下方核對後再產稿。")
    if summary["truncated"]:
        note += f"（另有 {summary['truncated']} 張較小的圖未送辨識）"
    messages.success(request, note)


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
            parse_images=request.POST.get("parse_images") == "1",
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
        # After the facts, never before: classifying a picture needs to know
        # whose campaign this is, or a competitor's shoe reads as usable material.
        if brief.parse_images:
            _ingest_images(request, brief)
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
        "strategies": SELECTABLE_STRATEGIES,
        "runs": brief.runs.filter(owner=request.user),
        "brief_images": brief.images.all(),
    })


@login_required
def brief_images(request, pk):
    """Re-parse the deck's pictures, or save the operator's approvals.

    Both live on one endpoint because they are the same decision from the user's
    side: what may this draft illustrate itself with.
    """
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)

    if request.POST.get("action") == "parse":
        brief.parse_images = True
        brief.save(update_fields=["parse_images", "updated_at"])
        _ingest_images(request, brief)
        return redirect("portal:brief_detail", pk=pk)

    approved = set(request.POST.getlist("approved"))
    for image in brief.images.all():
        key = str(image.pk)
        image.approved = key in approved
        image.caption = (request.POST.get(f"caption_{key}") or "").strip()[:200]
        image.save(update_fields=["approved", "caption"])

    messages.success(request, f"已確認 {len(approved)} 張圖片可用於稿件。")
    return redirect("portal:brief_detail", pk=pk)


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

    # Clamped rather than trusted: these arrive from a form and a rewrite budget
    # of 50 would tie up the model for an hour.
    strategy = request.POST.get("retrieval_strategy", "typical")
    if strategy not in dict(SELECTABLE_STRATEGIES):
        strategy = "typical"
    try:
        rewrites = int(request.POST.get("max_rewrites") or 1)
    except ValueError:
        rewrites = 1

    run = GenerationRun.objects.create(
        owner=request.user,
        brief=brief,
        outlet=guide.outlet,
        author=guide.author,
        style_guide=guide,
        mode="staged",
        retrieval_strategy=strategy,
        exemplar_count=4,
        max_rewrites=max(0, min(rewrites, 3)),
    )
    generate_service.run_generation(run, stop_after_outline=review_outline)

    if run.status == "failed":
        messages.error(request, f"產稿失敗：{run.error}")
    elif review_outline:
        messages.success(request, "大綱已產出。確認方向後再按「依大綱寫出正文」。")
    else:
        messages.success(request, "稿件已產出。")
    return redirect("portal:draft_detail", pk=run.pk)


def _delivered_draft(run) -> str:
    """The text the run actually hands over as its first draft.

    The rewrite budget makes the generator revise its own work before handing
    anything back. That is the generator finishing the job, not a second draft:
    the user asked for a piece of copy, and showing them the version that exists
    only because the machine disagreed with itself invites them to compare two
    texts they never asked to choose between. So the accepted auto-rewrite *is*
    the first draft here.

    /manage/ still shows every round — that view exists to inspect the
    machinery, and folding rounds together there would hide what an experiment
    is measuring.
    """
    last_auto = (run.revisions.filter(source="auto", accepted=True)
                 .order_by("-round").first())
    return last_auto.output if last_auto else run.output


def _human_revisions(run) -> list:
    """The user's own revisions, renumbered as if the auto rounds never existed.

    Their stored `round` counts auto-rewrites, so the first revision a user asks
    for lands on round 3 and would read as 第 3 稿 on a page showing only two
    versions. The numbering has to match what is on screen.
    """
    revisions = list(run.revisions.filter(accepted=True, source="human").order_by("round"))
    for i, revision in enumerate(revisions, start=2):
        revision.display_round = i
    return revisions


@login_required
def draft_detail(request, pk):
    run = get_object_or_404(_my_runs(request).select_related("brief", "outlet", "style_guide"), pk=pk)
    return render(request, "portal/draft_detail.html", {
        "nav": "drafts",
        "run": run,
        "draft_text": _delivered_draft(run),
        "revisions": _human_revisions(run),
        "revision_enabled": REVISION_ENABLED,
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

    if not REVISION_ENABLED:
        messages.info(request, "「依意見重寫」還在開發中，暫時無法使用。")
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
