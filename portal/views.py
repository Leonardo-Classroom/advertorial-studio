"""The portal: what someone who just wants a draft sees.

Much narrower than the staff tooling under /manage/. It asks for two things —
which media style to imitate, and which version of the brief's facts to write
from — and decides everything else from staff defaults. The knobs it used to
offer (retrieval strategy, rewrite budget, pausing at the outline) were removed
because the experiments could not separate them: three retrieval strategies
scored within 0.02 of each other across 81 runs, and a second rewrite was
accepted zero times out of twelve. A control that cannot change the outcome
still costs the user the time it takes to wonder about it.

Facts are versioned rather than edited in place: corrections arrive as a
sentence, a model folds them in, and the user confirms the diff. Nothing here
writes a version without that confirmation.

Everything is scoped to `request.user`: a portal user sees only their own
briefs and drafts. Staff see everything through /manage/ instead.
"""
from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.text import slugify

from briefs.models import Brief
from briefs.services import facts_text, facts_update, ppt_extract
from corpus.models import StyleGuide
from studio.models import GenerationRun, Revision, SiteSettings
from studio.services import draft_edit as draft_edit_service
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


def _listed_briefs(request):
    """What the user sees as "my briefs".

    A brief whose extraction failed has a file and nothing else — it cannot be
    worked on, only retried from the upload page. Listing it would put a dead
    row in front of the user with no way to act on it.
    """
    return _my_briefs(request).filter(fact_versions__isnull=False).distinct()


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
        "briefs": _listed_briefs(request)[:10],
        "runs": _my_runs(request).select_related("brief", "outlet")[:10],
        "brief_count": _listed_briefs(request).count(),
        "run_count": _my_runs(request).count(),
    })


def _extract_facts_or_retry(request, brief):
    """Turn the deck text into v1 of the facts, or send the user back to retry.

    Extraction failures stay on the upload page. There is nothing to do about
    one from inside a brief — the brief has no content yet — and the previous
    behaviour of dropping the user into a detail page full of blanks, with a
    「重新讀取」button as the only way out, made a model timeout look like a
    broken upload.
    """
    try:
        facts = ppt_extract.extract_facts(brief.raw_text)
    except Exception as exc:  # noqa: BLE001 - the user needs the reason
        messages.error(request, f"讀取內容失敗：{exc}")
        return redirect(f"{reverse('portal:upload')}?retry={brief.pk}")

    brief.add_facts_version(facts, source="extract")
    messages.success(request, f"已讀取 {brief.slide_count} 張投影片並抽出內容。請核對後再產稿。")
    # After the facts, never before: classifying a picture needs to know
    # whose campaign this is, or a competitor's shoe reads as usable material.
    if brief.parse_images:
        _ingest_images(request, brief)
    return redirect("portal:brief_detail", pk=brief.pk)


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
        except Exception as exc:  # noqa: BLE001 - the user needs the reason
            # The file itself is unreadable, so there is nothing to retry and
            # nothing worth keeping. Say so and let them upload another.
            brief.source_file.delete(save=False)
            brief.delete()
            messages.error(request, f"這個檔案讀不開（{exc}）。請確認是完整的 .pptx 後重新上傳。")
            return redirect("portal:upload")

        # Extract immediately: making the user press a second button to get
        # anything out of their own upload is friction with no upside.
        return _extract_facts_or_retry(request, brief)

    # A retry carries the brief whose text was already parsed: the file is fine,
    # only the model call failed, so re-uploading would redo the parse for nothing.
    retry_brief = None
    retry_pk = request.GET.get("retry")
    if retry_pk:
        retry_brief = _my_briefs(request).filter(
            pk=retry_pk, fact_versions__isnull=True).first()

    return render(request, "portal/upload.html", {
        "nav": "upload",
        "retry_brief": retry_brief,
    })


@login_required
def upload_retry(request, pk):
    """Run extraction again on a deck that is already uploaded and parsed."""
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:upload")
    if not brief.raw_text.strip():
        messages.error(request, "這份簡報沒有可讀取的文字內容，請換一個檔案。")
        return redirect("portal:upload")
    return _extract_facts_or_retry(request, brief)


@login_required
def brief_detail(request, pk):
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    versions = list(brief.fact_versions.all())
    latest = versions[0] if versions else None

    # Viewing an old version and generating from one are separate choices, so
    # they get separate controls: looking back at v1 to see what changed should
    # not quietly arm the generate button with stale facts.
    viewing = latest
    wanted = request.GET.get("v")
    if wanted:
        viewing = next((v for v in versions if str(v.version) == wanted), latest)

    facts = viewing.data if viewing else {}

    # Optional, and off unless asked for: most visits are to read what the
    # brief says, and a page permanently striped red and green reads as a
    # problem report rather than as the facts a draft will be written from.
    show_diff = request.GET.get("diff") == "1"
    others = [v for v in versions if viewing and v.pk != viewing.pk]
    base = None
    if show_diff and others:
        wanted_base = request.GET.get("base")
        base = (next((v for v in others if str(v.version) == wanted_base), None)
                # v1 by default rather than the version before this one: the
                # question people ask of a corrected brief is what it says now
                # against what came out of the deck, not what the last edit did.
                or next((v for v in others if v.version == 1), None)
                or others[-1])

    context = {
        "nav": "briefs",
        "brief": brief,
        "versions": versions,
        "viewing": viewing,
        "latest": latest,
        "base": base,
        "base_options": others,
        "show_diff": show_diff,
        "marked_sections": (facts_text.to_sections_marked(base.data, facts) if base else None),
        "marked_uncertain": (facts_text.uncertain_marked(base.data, facts) if base else None),
        "sections": facts_text.to_sections(facts),
        "uncertain": facts_text.uncertain_items(facts),
        "latest_uncertain": facts_text.uncertain_items(latest.data if latest else {}),
        "guides": StyleGuide.objects.filter(is_active=True).select_related("outlet", "author"),
        "confirm_updates": SiteSettings.load().confirm_fact_updates,
        "runs": brief.runs.filter(owner=request.user),
        "brief_images": brief.images.all(),
    }
    # Switching version or turning the comparison on replaces this one card, not
    # the page: the correction box beside it is often half-written by then.
    if request.GET.get("partial") == "1":
        return render(request, "portal/_facts_pane.html", context)
    return render(request, "portal/brief_detail.html", context)


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

    # Ticking a box saves itself, so this arrives once per change. A redirect
    # would make the page announce "已確認 12 張" every time someone changed
    # their mind about one picture.
    if request.headers.get("X-Requested-With") == "fetch":
        return HttpResponse(status=204)

    messages.success(request, f"已確認 {len(approved)} 張圖片可用於稿件。")
    return redirect("portal:brief_detail", pk=pk)


def _pending_key(pk) -> str:
    return f"facts_pending:{pk}"


@login_required
def brief_facts_update(request, pk):
    """Merge a plain-language correction into a new version of the facts.

    The merge is a model call over fields a draft has to reproduce verbatim, so
    it can quietly get something wrong. Whether that warrants stopping the user
    to approve a diff is a staff setting (`confirm_fact_updates`), off by
    default: nothing is overwritten either way — a bad merge is undone by
    generating from the previous version, which is still there.
    """
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)

    user_input = (request.POST.get("correction") or "").strip()
    if not user_input:
        messages.error(request, "請先寫下要更正什麼。")
        return redirect("portal:brief_detail", pk=pk)

    versions = list(brief.fact_versions.all())
    if not versions:
        messages.error(request, "這份簡報還沒有抽出內容。")
        return redirect("portal:brief_detail", pk=pk)

    # Which version the correction is applied to. The newest by default, but a
    # deliberate choice: picking an older one is how you carry a good early
    # version forward instead of correcting the same mistake twice. Nothing is
    # lost either way — the versions in between stay exactly where they are.
    wanted = request.POST.get("base_version")
    current = next((v for v in versions if str(v.version) == wanted), versions[0])

    try:
        # The history goes in with it, so "改回原本的" has an answer. Without it
        # the model saw one snapshot and could only decline.
        merged = facts_update.propose(current.data or {}, user_input, versions=versions)
    except Exception as exc:  # noqa: BLE001 - the user needs the reason
        messages.error(request, f"更新失敗：{exc}。你剛才輸入的內容沒有送出，請再試一次。")
        return redirect("portal:brief_detail", pk=pk)

    if merged == (current.data or {}):
        messages.info(request, f"這次更正沒有改動 {current.label} 的任何內容。"
                               "如果你是想整版回到某個舊版本，請切到那一版再按「以這一版為準」——"
                               "整版還原是逐字複製，不經過 AI。")
        return redirect("portal:brief_detail", pk=pk)

    if SiteSettings.load().confirm_fact_updates:
        request.session[_pending_key(pk)] = {
            "data": merged, "user_input": user_input, "base": current.pk}
        return redirect("portal:brief_facts_confirm", pk=pk)

    changed = len(facts_text.diff(current.data or {}, merged))
    version = brief.add_facts_version(merged, source="user_update",
                                      user_input=user_input, parent=current)
    messages.success(request, f"已依 {current.label} 更新為 {version.label}，改了 {changed} 個項目。"
                              f"{current.label} 仍然留著，產稿時可以指定用它。")
    return redirect("portal:brief_detail", pk=pk)


@login_required
def brief_facts_confirm(request, pk):
    """Show what the merge changed; write the new version only if accepted."""
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    pending = request.session.get(_pending_key(pk))
    if not pending:
        return redirect("portal:brief_detail", pk=pk)

    # The diff has to be against whatever the merge started from, which is not
    # necessarily the newest version.
    current = (brief.fact_versions.filter(pk=pending.get("base")).first()
               or brief.latest_facts())
    before = current.data if current else {}

    if request.method == "POST":
        del request.session[_pending_key(pk)]
        if request.POST.get("action") != "apply":
            messages.info(request, "已取消，內容維持原樣。")
            return redirect("portal:brief_detail", pk=pk)

        version = brief.add_facts_version(
            pending["data"], source="user_update",
            user_input=pending["user_input"], parent=current)
        messages.success(request, f"已依 {current.label} 更新為 {version.label}。")
        return redirect("portal:brief_detail", pk=pk)

    return render(request, "portal/brief_facts_confirm.html", {
        "nav": "briefs",
        "brief": brief,
        "current": current,
        "user_input": pending["user_input"],
        "rows": facts_text.diff(before, pending["data"]),
    })


@login_required
def generate(request, pk):
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)

    guide = get_object_or_404(StyleGuide.objects.filter(is_active=True),
                              pk=request.POST.get("style_guide"))

    # Which facts to write from is the user's only remaining choice here, and it
    # is a real one: correcting the brand name produces a new version, and a
    # draft made before that correction is a different draft.
    versions = list(brief.fact_versions.all())
    if not versions:
        messages.error(request, "這份簡報還沒有抽出內容。")
        return redirect("portal:brief_detail", pk=pk)
    wanted = request.POST.get("facts_version")
    version = next((v for v in versions if str(v.pk) == wanted), versions[0])

    # The rest are staff defaults now. They were never decisions the person
    # writing a draft could make usefully — see SiteSettings.
    defaults = SiteSettings.load()

    run = GenerationRun.objects.create(
        owner=request.user,
        brief=brief,
        facts_version=version,
        outlet=guide.outlet,
        author=guide.author,
        style_guide=guide,
        mode="staged",
        retrieval_strategy=defaults.retrieval_strategy,
        exemplar_count=defaults.exemplar_count,
        max_rewrites=max(0, min(defaults.max_rewrites, 3)),
    )
    generate_service.run_generation(run)

    if run.status == "failed":
        messages.error(request, f"產稿失敗：{run.error}")
    else:
        messages.success(request, "稿件已產出。")
    return redirect("portal:draft_detail", pk=run.pk)


def _manual_edit(run):
    """The user's own edit of the delivered text, if they made one."""
    return (run.revisions.filter(source="manual", accepted=True)
            .order_by("-round").first())


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
    # A hand edit wins outright: the user has read the delivered text and said
    # what it should say. It is not another draft to compare against, it is the
    # draft.
    edited = _manual_edit(run)
    if edited:
        return edited.output

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
    draft_text = _delivered_draft(run)
    return render(request, "portal/draft_detail.html", {
        "nav": "drafts",
        "run": run,
        "draft_text": draft_text,
        "edit_sections": draft_edit_service.to_fields(draft_text, run.brief),
        "edited": _manual_edit(run),
        "revisions": _human_revisions(run),
        "revision_enabled": REVISION_ENABLED,
    })


@login_required
def draft_edit(request, pk):
    """Save the draft as the user retyped it, paragraph by paragraph."""
    run = get_object_or_404(_my_runs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:draft_detail", pk=pk)

    before = _delivered_draft(run)
    after = draft_edit_service.from_fields(before, request.POST)
    if after.strip() == before.strip():
        messages.info(request, "內容沒有變動，未儲存。")
        return redirect("portal:draft_detail", pk=pk)

    edited = _manual_edit(run)
    if edited:
        # One standing edit, not a chain of them: every save is the same person
        # continuing to fix the same draft, and a stack of 第 4 稿／第 5 稿 for
        # three typo fixes is a worse record than the current text plus the
        # model's original.
        edited.output = after
        edited.save(update_fields=["output"])
    else:
        last = run.revisions.order_by("-round").first()
        Revision.objects.create(
            run=run, round=(last.round + 1) if last else 2,
            source="manual", feedback="（在稿件頁直接編輯）", output=after,
        )
    messages.success(request, "已儲存你的修改。模型原本寫的版本仍保留在紀錄裡。")
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
