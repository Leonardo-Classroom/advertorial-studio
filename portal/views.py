"""The portal: what someone who just wants a draft sees.

Much narrower than the staff tooling under /manage/. It asks for two things —
which media style to imitate, and which version of the brief's facts to write
from — and decides everything else from staff defaults. The knobs it used to
offer (retrieval strategy, rewrite budget, pausing at the outline) were removed
because the experiments could not separate them: three retrieval strategies
scored within 0.02 of each other across 81 runs, and a second rewrite was
accepted zero times out of twelve. A control that cannot change the outcome
still costs the user the time it takes to wonder about it.

Both the facts and the drafts are versioned rather than edited in place.
Corrections arrive as a sentence, a model folds them in, and the result becomes
a new version; nothing is ever overwritten, so every screen can offer "which
version" as an ordinary choice.

Everything is scoped to `request.user`: a portal user sees only their own
briefs and drafts. Staff see everything through /manage/ instead.
"""
from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.text import slugify

from briefs.models import Brief
from briefs.services import facts_edit, facts_text, facts_update
from briefs.services import ingest_runner
from briefs.services import upload as upload_service
from corpus.models import Outlet, StyleGuide
from studio.models import GenerationRun, SiteSettings
from studio.services import draft_edit as draft_edit_service
from studio.services import generate as generate_service
from studio.services import drafts as draft_service
from studio.services import runner


def _my_briefs(request):
    return Brief.objects.filter(owner=request.user)


def _listed_briefs(request):
    """What the user sees as "my briefs" — anything they actually uploaded.

    This used to hide briefs with no facts, on the grounds that a failed
    extraction left a dead row nobody could act on. Under the async upload
    that rule strands people: the upload redirects to the brief and the work
    happens in the background, so a brief with no facts is now either still
    being parsed, or one whose files failed and whose detail page is the only
    place explaining why. Hiding either would take away the page the user was
    just looking at the moment they navigated off it.
    """
    return _my_briefs(request).filter(
        Q(source_files__isnull=False) | Q(fact_versions__isnull=False)).distinct()


def _ingest_images(request, brief) -> None:
    """Parse the deck's pictures, reporting what survived filtering.

    The counts are worth showing rather than hiding: a deck yields hundreds of
    embedded images and only a handful are article material, so "12 張裡有 6 張可用"
    tells the user the filter worked instead of leaving them wondering why their
    358-picture deck produced six thumbnails.
    """
    from briefs.services import images as image_service

    try:
        summary = image_service.ingest(brief, run_classify=False)
    except Exception as exc:  # noqa: BLE001 - text extraction already succeeded
        messages.warning(request, f"圖片解析失敗：{exc}。文字內容不受影響，稿子仍可正常產出。")
        return

    if not summary["stored"]:
        messages.info(request, "簡報裡沒有找到適合放進稿子的圖片（多半是表格、logo 或裝飾圖）。")
        return

    note = (f"圖片解析完成：{summary['found']} 張中篩出 {summary['stored']} 張候選。"
            f"勾選要用的圖之後，按「解析勾選的圖片」補上圖說再產稿。")
    if summary["truncated"]:
        note += f"（另有 {summary['truncated']} 張較小的圖未送辨識）"
    messages.success(request, note)


def _my_runs(request):
    return GenerationRun.objects.filter(owner=request.user)


# Offered page sizes. Whitelisted rather than taken as given: `?per=1000000`
# would otherwise be a way for any logged-in user to ask the server to render
# every row they own in one response.
PAGE_SIZES = (10, 30, 50, 100)
DEFAULT_PAGE_SIZE = 10


def _page_size(request) -> int:
    """The page size to use, remembering the last one the user picked.

    Kept in the session rather than in every link: the choice is a preference
    about how they like to read these lists, so it should survive following a
    link out and coming back, which a URL-only value does not.
    """
    wanted = request.GET.get("per")
    if wanted:
        try:
            chosen = int(wanted)
        except ValueError:
            chosen = None
        if chosen in PAGE_SIZES:
            request.session["page_size"] = chosen
            return chosen

    remembered = request.session.get("page_size")
    return remembered if remembered in PAGE_SIZES else DEFAULT_PAGE_SIZE


def _qs_extra(**params) -> str:
    """The filter/page-size part of the query string, for links that must keep
    it — the sort headers and the pager, which otherwise reset the view the
    moment you use them."""
    from urllib.parse import urlencode

    kept = {k: v for k, v in params.items() if v}
    return ("&" + urlencode(kept)) if kept else ""


BRIEF_SORTS = {
    "name": "title",
    "facts": "newest_facts",
    "time": "created_at",
    # Same reasoning as the draft list: a facts version is what an edit
    # produces here, so its timestamp is the edit. `Brief.updated_at` also
    # moves when a background job flips `processing`.
    "edited": "newest_facts_at",
}

# Which column header sorts by what. `None` means it cannot be done in SQL —
# see the note in `drafts()`. Keys are what appears in the URL, so they stay
# short and stable even if a heading is reworded.
DRAFT_SORTS = {
    "title": None,
    "version": "newest_version",
    "brief": "brief__title",
    "facts": "facts_version__version",
    "outlet": "outlet__name",
    "time": "created_at",
    # When the copy itself last changed — a new draft version is what an edit
    # or a rewrite produces, so its timestamp is the honest answer. `run.
    # updated_at` is not: `auto_now` moves it on any save at all, including the
    # background status flips nobody would call an edit.
    "edited": "newest_draft_at",
}


def _listed_runs(request):
    """Runs worth showing as drafts — everything except the ones that failed.

    A failed run produced no copy, so it has no headline, no version and
    nothing to open; listing it only asks the user to notice a row they can do
    nothing with. Kept in the database either way: the staff run list under
    /manage/ is where failures are meant to be looked at.
    """
    return _my_runs(request).exclude(status="failed")


def _style_guides():
    """The style guides on offer, English-named outlets first.

    `is_active` is the only thing that decides what appears — the same switch
    the guide list under /manage/ shows as 啟用 / 停用. This once also filtered
    out per-author guides, which made that badge a lie: eleven guides were
    marked 啟用 and did not appear here, and the generate endpoint accepted
    them anyway because it only ever checked `is_active`. Two rules for "may
    this be used" is one too many; the author-level guides are simply
    deactivated instead.

    Sorted in Python rather than by the database: ordering "GQ" before
    "工商時報" is a question about scripts, and collation for that is a
    per-database, per-locale setting this project does not pin.
    """
    guides = StyleGuide.objects.filter(is_active=True).select_related("outlet", "author")
    return sorted(guides, key=lambda g: (not g.outlet.name.isascii(), g.outlet.name))

@login_required
def home(request):
    """The user's projects, paginated.

    It used to be a bare `[:10]` — fine while this was a summary card above the
    drafts, useless once it became the whole page: the eleventh project was
    unreachable except by typing its URL, with nothing on screen admitting the
    list had been cut.
    """
    from django.db.models import OuterRef, Subquery

    from briefs.models import BriefFacts

    # The version shown in the 內容版本 column is a property (`latest_facts`),
    # which SQL cannot order by — so the number comes along as an annotation
    # purely so that column can be sorted.
    newest = BriefFacts.objects.filter(brief=OuterRef("pk")).order_by("-version")
    briefs = _listed_briefs(request).annotate(
        newest_facts=Subquery(newest.values("version")[:1]),
        newest_facts_at=Subquery(newest.values("created_at")[:1]))

    # Filters. Only the two that mean something for a project: it has no
    # article headline to search, and no single outlet — a project can have
    # drafts in several styles, so "模仿風格" belongs to the draft list.
    #
    # The status values mirror what the 內容版本 column already shows, so
    # filtering by one is filtering by what is on screen. Expressed against the
    # `newest_facts` subquery rather than by joining `fact_versions`, which
    # would multiply rows and need another `distinct()`.
    sel_name = (request.GET.get("q") or "").strip()
    sel_status = request.GET.get("status") or ""
    if sel_name:
        briefs = briefs.filter(title__icontains=sel_name)
    if sel_status == "done":
        briefs = briefs.filter(newest_facts__isnull=False)
    elif sel_status == "processing":
        briefs = briefs.filter(newest_facts__isnull=True, processing=True)
    elif sel_status == "none":
        briefs = briefs.filter(newest_facts__isnull=True, processing=False)
    else:
        # 未完成 is hidden by default: nothing was extracted, so there is
        # nothing to write from and the row is only in the way.
        #
        # Hidden, not dropped — picking 未完成 above brings them back. That
        # matters more than it looks: `_listed_briefs` deliberately stopped
        # filtering these out because the detail page is the only thing that
        # says *why* an upload failed, and one whose source file failed to
        # parse cannot even be retried. Excluding them outright would make a
        # failed upload silently vanish with no way to find out what happened.
        briefs = briefs.exclude(newest_facts__isnull=True, processing=False)

    key = request.GET.get("sort") or "time"
    if key not in BRIEF_SORTS:
        key = "time"
    descending = (request.GET.get("dir") or ("desc" if key == "time" else "asc")) == "desc"
    field = BRIEF_SORTS[key]
    order = f"-{field}" if descending else field

    per = _page_size(request)
    page = Paginator(briefs.order_by(order, "-created_at"), per).get_page(request.GET.get("page"))
    return render(request, "portal/home.html", {
        "nav": "home",
        "page": page,
        # Built here because `get_elided_page_range` takes an argument, which a
        # template cannot pass.
        "page_range": page.paginator.get_elided_page_range(page.number),
        "pager_qs": _qs_extra(sort=key, dir="desc" if descending else "asc",
                              q=sel_name, status=sel_status, per=per),
        "sort": key,
        "dir": "desc" if descending else "asc",
        "flip": "asc" if descending else "desc",
        "sortable": [("name", "名稱"), ("facts", "內容版本"),
                     ("time", "建立時間"), ("edited", "最後編輯時間")],
        "sel_name": sel_name,
        "sel_status": sel_status,
        "statuses": [("done", "已抽取內容"), ("processing", "處理中"), ("none", "未完成")],
        "per": per,
        "page_sizes": PAGE_SIZES,
        "qs_extra": _qs_extra(q=sel_name, status=sel_status, per=per),
    })


@login_required
def drafts(request):
    """Every draft this user has produced, on a page of its own.

    Each row leads with the article's own headline, so the list can be read as
    "which piece is this" — the project name repeats across every draft made
    from the same deck, and a column of ten identical names identifies nothing.

    The headline lives inside the newest version's text, so it arrives by
    subquery rather than by touching `draft_versions` per row: the latter is a
    query per draft, and this page exists precisely to show a lot of them.

    Named `newest_draft`, not `latest_text` — `GenerationRun` already has a
    `latest_text` property, and annotating over it fails outright ("property
    has no setter") rather than silently shadowing it.
    """
    from django.db.models import OuterRef, Subquery

    from studio.models import DraftVersion

    newest = DraftVersion.objects.filter(run=OuterRef("pk")).order_by("-version")
    qs = (_listed_runs(request)
          .select_related("brief", "outlet", "author", "facts_version")
          .annotate(newest_draft=Subquery(newest.values("text")[:1]),
                    newest_version=Subquery(newest.values("version")[:1]),
                    newest_draft_at=Subquery(newest.values("created_at")[:1])))

    # Filters. The project is typed rather than picked from a list: the list
    # was every project the user has ever made, most of them named after the
    # same deck, and picking "0410_NB 2025…" out of four identical labels is
    # not something a dropdown helps with. The outlet stays a dropdown — there
    # are a handful of them and the names are distinct.
    sel_title = (request.GET.get("q") or "").strip()
    sel_brief = (request.GET.get("brief") or "").strip()
    sel_outlet = request.GET.get("outlet") or ""
    outlet_options = Outlet.objects.filter(pk__in=qs.values("outlet_id")).order_by("name")
    if sel_brief:
        qs = qs.filter(brief__title__icontains=sel_brief)
    if sel_outlet.isdigit():
        qs = qs.filter(outlet_id=sel_outlet)

    per = _page_size(request)

    key = request.GET.get("sort") or "time"
    if key not in DRAFT_SORTS:
        key = "time"
    # Time reads newest-first; a name or a number reads A→Z. Both still flip.
    descending = (request.GET.get("dir") or ("desc" if key == "time" else "asc")) == "desc"

    field = DRAFT_SORTS[key]
    if field is not None:
        # `-created_at` breaks ties so equal keys (two drafts of one project,
        # same outlet) keep a stable, meaningful order instead of an arbitrary
        # one that shuffles between page loads.
        qs = qs.order_by(f"-{field}" if descending else field, "-created_at")

    # Both searching and sorting by headline have to happen in Python: the
    # headline lives inside the draft text, and the text does not start with it
    # — `## FB貼文文案` comes first — so neither an `icontains` on the column nor
    # an `order_by` on it would be about the headline at all. That means loading
    # the user's runs, which `_listed_runs` already scopes this to; if that ever
    # stops being a page-sized number, the fix is to store the headline on
    # `DraftVersion`, not to make SQL guess at it.
    if field is None or sel_title:
        rows = list(qs)
        for run in rows:
            run.headline = draft_service.headline(run.newest_draft)
        if sel_title:
            needle = sel_title.lower()
            rows = [r for r in rows if needle in (r.headline or "").lower()]
        if field is None:
            rows.sort(key=lambda r: r.headline or "", reverse=descending)
        page = Paginator(rows, per).get_page(request.GET.get("page"))
    else:
        page = Paginator(qs, per).get_page(request.GET.get("page"))
        for run in page.object_list:
            run.headline = draft_service.headline(run.newest_draft)

    return render(request, "portal/drafts.html", {
        "nav": "drafts",
        "page": page,
        "page_range": page.paginator.get_elided_page_range(page.number),
        # The pager carries the sort as well as the filters; `qs_extra` must
        # not, or the sort headers would emit two `sort=` parameters.
        "pager_qs": _qs_extra(sort=key, dir="desc" if descending else "asc",
                              q=sel_title, brief=sel_brief, outlet=sel_outlet, per=per),
        "sort": key,
        "dir": "desc" if descending else "asc",
        # Paired with the labels here rather than repeated in the template, so
        # a new sortable column is one entry in `DRAFT_SORTS` plus one here.
        "sortable": [("title", "標題"), ("version", "版本"), ("brief", "專案名稱"),
                     ("facts", "專案版本"), ("outlet", "模仿風格"),
                     ("time", "生成時間"), ("edited", "最後編輯時間")],
        "outlet_options": outlet_options,
        "sel_title": sel_title,
        "sel_brief": sel_brief,
        "sel_outlet": sel_outlet,
        "per": per,
        "page_sizes": PAGE_SIZES,
        "qs_extra": _qs_extra(q=sel_title, brief=sel_brief, outlet=sel_outlet, per=per),
        # What each header should link to: clicking the active column flips it,
        # clicking another starts that one at its own natural direction.
        "flip": "asc" if descending else "desc",
    })


def _needs_facts_retry(request):
    """Briefs with at least one parsed file but no facts version, not currently
    running — the `extract_facts()` call itself failed, so there is nothing
    left to do but let the user ask for it again."""
    return _my_briefs(request).filter(
        fact_versions__isnull=True, processing=False,
        source_files__status="done").distinct()


@login_required
def upload(request):
    if request.method == "POST":
        files = request.FILES.getlist("source_files")
        try:
            brief = upload_service.create_brief(
                owner=request.user,
                title=request.POST.get("title") or "",
                files=files,
            )
        except ValueError as exc:  # noqa: BLE001 - the user needs the reason
            messages.error(request, str(exc))
            return redirect("portal:upload")

        # Processing happens in the background; this page hands straight off
        # to the brief, which shows the real progress instead of a guess.
        return redirect("portal:brief_detail", pk=brief.pk)

    # A retry carries the brief whose files were already parsed: they are
    # fine, only the facts call failed, so re-uploading would redo the parse
    # for nothing.
    retry_brief = None
    retry_pk = request.GET.get("retry")
    if retry_pk:
        retry_brief = _needs_facts_retry(request).filter(pk=retry_pk).first()

    return render(request, "portal/upload.html", {
        "nav": "upload",
        "retry_brief": retry_brief,
    })


@login_required
def upload_retry(request, pk):
    """Run fact extraction again on a brief whose files are already parsed."""
    brief = get_object_or_404(_needs_facts_retry(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:upload")
    ingest_runner.retry_facts(brief)
    return redirect("portal:brief_detail", pk=brief.pk)


@login_required
def source_file_retry(request, pk, file_pk):
    """Re-parse one failed source file and fold its content into the facts."""
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)

    source_file = get_object_or_404(brief.source_files, pk=file_pk)
    if source_file.status != "failed":
        messages.info(request, "這個檔案沒有處理失敗，不需要重新處理。")
        return redirect("portal:brief_detail", pk=pk)

    ingest_runner.retry_source_file(brief, source_file)
    messages.info(request, f"已在背景重新處理《{source_file.original_filename}》，"
                           "成功的話會把它的內容併進抽出的內容，存成新的一版。")
    return redirect("portal:brief_detail", pk=pk)

@login_required
def upload_status(request, pk):
    """Where an upload/retry batch has got to, for the pages waiting on it."""
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    return JsonResponse(upload_service.status_payload(brief))


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

    # The card lists the article's own headline, so each row needs its newest
    # draft version — its text to read the headline out of, and its number for
    # the `p` label. By subquery rather than by touching `draft_versions` per
    # row: same reasoning as the draft list (`portal.views.drafts`).
    from django.db.models import OuterRef, Subquery

    from studio.models import DraftVersion

    newest = DraftVersion.objects.filter(run=OuterRef("pk")).order_by("-version")
    runs = list(brief.runs.filter(owner=request.user)
                .select_related("outlet", "author", "facts_version")
                .annotate(newest_draft=Subquery(newest.values("text")[:1]),
                          newest_version=Subquery(newest.values("version")[:1]),
                          newest_draft_at=Subquery(newest.values("created_at")[:1])))
    for run in runs:
        run.headline = draft_service.headline(run.newest_draft)
    runner.reap_stale(runs)

    source_files = list(brief.source_files.order_by("order", "pk"))
    ingest_runner.reap_stale(source_files)
    ingest_runner.reap_stale_brief(brief)
    # Only offer the retry button when a retry can actually do something —
    # if every file failed to parse there is no text for it to work from, and
    # the per-file 處理失敗 pills already say why.
    needs_facts_retry = (not brief.processing and not versions
                         and any(f.status == "done" for f in source_files))

    context = {
        "nav": "briefs",
        "brief": brief,
        "source_files": source_files,
        "files_in_progress": brief.processing,
        "needs_facts_retry": needs_facts_retry,
        "versions": versions,
        "viewing": viewing,
        "latest": latest,
        "base": base,
        "base_options": others,
        "show_diff": show_diff,
        "marked_sections": (facts_text.to_sections_marked(base.data, facts) if base else None),
        "marked_uncertain": (facts_text.uncertain_marked(base.data, facts) if base else None),
        "sections": facts_text.to_sections(facts),
        # Editable copies of the version being viewed, so 人工編輯 edits what is
        # on screen rather than always the newest.
        "edit_fields": facts_edit.to_fields(facts),
        "uncertain": facts_text.uncertain_items(facts),
        "latest_uncertain": facts_text.uncertain_items(latest.data if latest else {}),
        # The primary-KOL form always acts on the newest version (see
        # `brief_set_primary_kol`), so it reads from `latest` too, not
        # `facts`/`viewing` — those can be an older version under inspection.
        "latest_primary_kol": (latest.data.get("primary_kol") if latest else "") or "",
        "kol_candidates": ((latest.data.get("kol") if latest else None) or []),
        "guides": _style_guides(),
        "confirm_updates": SiteSettings.load().confirm_fact_updates,
        "llm_backend": SiteSettings.load().llm_backend,
        "runs": runs,
        "runs_active": any(r.in_progress for r in runs),
        "brief_images": brief.images.order_by("source_file_id", "slide_index", "pk"),
    }
    # Switching version or turning the comparison on replaces this one card, not
    # the page: the correction box beside it is often half-written by then.
    if request.GET.get("partial") == "1":
        return render(request, "portal/_facts_pane.html", context)
    if request.GET.get("partial") == "runs":
        return render(request, "portal/_runs_card.html", context)
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
        _ingest_images(request, brief)
        return redirect("portal:brief_detail", pk=pk)

    if request.POST.get("action") == "upload":
        from briefs.services import images as image_service

        files = request.FILES.getlist("images")
        if not files:
            messages.error(request, "請選擇至少一張圖片。")
        else:
            stored, rejected = image_service.save_manual_uploads(brief, files)
            if stored:
                messages.success(request, f"已上傳 {len(stored)} 張圖片，預設為可用素材。")
            if rejected:
                messages.warning(request, f"{len(rejected)} 個檔案無法辨識為圖片，已略過："
                                          + "、".join(rejected))
        return redirect("portal:brief_detail", pk=pk)

    if request.POST.get("action") == "classify":
        from briefs.services import images as image_service

        summary = image_service.classify_checked(brief)
        if summary.get("failed"):
            messages.error(
                request,
                f"辨識 {summary['classified']} 張，其中 {summary['failed']} 張沒有結果"
                f"——本地辨識服務沒有回應。請確認 Ollama 還活著"
                f"（./restart_ollama.sh 可以重啟），再重跑一次。")
        elif summary["classified"]:
            messages.success(request, f"已辨識 {summary['classified']} 張圖片，"
                                      f"其中 {summary['usable']} 張判定可用。")
        else:
            messages.info(request, "沒有需要辨識的圖片——請先勾選要辨識的圖片，已經有圖說的都跳過了。")
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


@login_required
def brief_images_progress(request, pk):
    """How far the running classify pass has got — see the staff twin."""
    from briefs.services import images as image_service

    get_object_or_404(_my_briefs(request), pk=pk)
    return JsonResponse(image_service.read_classify_progress(pk))


@login_required
def brief_rename(request, pk):
    """Rename a project, from wherever its name is shown.

    Saves itself and answers with the stored name, so the caller can put back
    whatever the server actually kept — the value is trimmed and capped at the
    column width here, and a box left showing untrimmed text would disagree
    with every other screen.

    Worth being clear about the scope: this is the *project* name, so it
    changes the project list and every draft made from it. Nothing else is
    touched — a draft's own headline lives in its text.
    """
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)

    title = " ".join((request.POST.get("title") or "").split())[:200]
    if not title:
        if request.headers.get("X-Requested-With") == "fetch":
            return JsonResponse({"error": "專案名稱不能是空的。"}, status=400)
        messages.error(request, "專案名稱不能是空的。")
        return redirect("portal:brief_detail", pk=pk)

    if title != brief.title:
        brief.title = title
        brief.save(update_fields=["title", "updated_at"])

    if request.headers.get("X-Requested-With") == "fetch":
        return JsonResponse({"title": brief.title})
    return redirect("portal:brief_detail", pk=pk)

def _chosen_facts_version(brief, wanted):
    """The fact version a form named, else the newest.

    Same rule as the correction box: editing an older version on purpose is how
    you carry a good early one forward, and the versions in between stay put.
    """
    versions = list(brief.fact_versions.all())
    if not versions:
        return None
    return next((v for v in versions if str(v.version) == str(wanted)), versions[0])


@login_required
def brief_facts_edit(request, pk):
    """Save facts the user retyped directly, field by field.

    Deliberately not routed through `confirm_fact_updates` the way the
    natural-language correction is. That confirmation exists because a model
    merge can change a field nobody mentioned; here the user typed the values
    themselves, so showing them a diff of their own typing asks them to check
    the one thing they cannot have got wrong by surprise.
    """
    from briefs.services import facts_edit

    brief = get_object_or_404(_my_briefs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)

    base = _chosen_facts_version(brief, request.POST.get("base_version"))
    if base is None:
        messages.error(request, "這份簡報還沒有可以編輯的內容。")
        return redirect("portal:brief_detail", pk=pk)

    merged = facts_edit.from_fields(base.data, request.POST)
    if merged == (base.data or {}):
        messages.info(request, "內容沒有變動，未儲存。")
        return redirect("portal:brief_detail", pk=pk)

    version = brief.add_facts_version(merged, source="user_edit", parent=base)
    messages.success(request, f"已依 {base.label} 存成 {version.label}。"
                              f"{base.label} 仍然留著，可以從上方切回去看。")
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
def brief_set_primary_kol(request, pk):
    """Set the primary spokesperson directly — a plain field, not a correction.

    Unlike `brief_facts_update`, this never goes through the model: the value
    is exactly what the operator picked from the candidate list (or typed),
    there is nothing to interpret, and a model call would only add latency
    and a chance of it touching some other field. Still recorded as an
    ordinary fact version via `add_facts_version` so it shows up in the
    version history and feeds the same `primary_kol_directive` as any other
    way of setting this field.

    Saves itself on every change, the same as the picture review's checkboxes
    (`portal:brief_images`) — there is nothing further to confirm, so a
    button asking to confirm it would be asking the operator to confirm a
    decision they already made by typing.
    """
    brief = get_object_or_404(_my_briefs(request), pk=pk)
    is_fetch = request.headers.get("X-Requested-With") == "fetch"
    if request.method != "POST":
        return redirect("portal:brief_detail", pk=pk)

    current = brief.latest_facts()
    if current is None:
        if is_fetch:
            return HttpResponse("這份簡報還沒有抽出內容。", status=409)
        messages.error(request, "這份簡報還沒有抽出內容。")
        return redirect("portal:brief_detail", pk=pk)

    value = (request.POST.get("primary_kol") or "").strip()
    data = dict(current.data or {})
    if data.get("primary_kol", "") != value:
        data["primary_kol"] = value
        brief.add_facts_version(
            data, source="user_update", parent=current,
            user_input=f"設定主打代言人：{value}" if value else "清除主打代言人")

    if is_fetch:
        return HttpResponse(status=204)
    messages.success(request, f"主打代言人已設為「{value}」。" if value
                              else "已清除主打代言人。")
    return redirect("portal:brief_detail", pk=pk)


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
        mode=defaults.default_mode,
        retrieval_strategy=defaults.retrieval_strategy,
        exemplar_count=defaults.exemplar_count,
        max_rewrites=max(0, min(defaults.max_rewrites, 3)),
    )
    # Handed to a worker rather than run here: this response opens in a new tab
    # and the user goes straight back to the brief, where they may well ask for
    # another one before this finishes.
    if not runner.submit(run):
        # The queue is full. Drop the row rather than leave a draft that says
        # 排隊中 with nothing scheduled behind it — it was created seconds ago
        # and nothing refers to it yet.
        run.delete()
        messages.error(request, "目前排隊的稿件已達上限，請等前面的產完再試。")
        return redirect("portal:brief_detail", pk=pk)
    return redirect("portal:draft_detail", pk=run.pk)


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


def _chosen_version(run, wanted):
    """The draft version a form or link named, else the newest."""
    versions = draft_service.versions(run)
    if not versions:
        return draft_service.ensure_first_version(run)
    return next((v for v in versions if str(v.version) == str(wanted)), versions[0])


@login_required
def draft_status(request, pk):
    """Where a run has got to, for the pages waiting on it."""
    run = get_object_or_404(_my_runs(request), pk=pk)
    runner.reap_stale([run])
    return JsonResponse({
        "status": run.status,
        "label": run.get_status_display(),
        "in_progress": run.in_progress,
        "stages": run.stage_labels,
        "queued": run.status == "pending",
    })


@login_required
def draft_detail(request, pk):
    run = get_object_or_404(_my_runs(request).select_related("brief", "outlet", "style_guide"), pk=pk)
    runner.reap_stale([run])
    if not run.in_progress:
        # Runs that finished before drafts were versioned get their v1 here.
        draft_service.ensure_first_version(run)

    versions = draft_service.versions(run)
    latest = versions[0] if versions else None

    viewing = latest
    wanted = request.GET.get("v")
    if wanted:
        viewing = next((v for v in versions if str(v.version) == wanted), latest)

    show_diff = request.GET.get("diff") == "1"
    others = [v for v in versions if viewing and v.pk != viewing.pk]
    base = None
    if show_diff and others:
        wanted_base = request.GET.get("base")
        base = (next((v for v in others if str(v.version) == wanted_base), None)
                or next((v for v in others if v.version == 1), None)
                or others[-1])

    draft_text = viewing.text if viewing else ""
    context = {
        "nav": "drafts",
        "run": run,
        "versions": versions,
        "viewing": viewing,
        "latest": latest,
        "base": base,
        "base_options": others,
        "show_diff": show_diff,
        "marked": (draft_service.mark_paragraphs(base.text, draft_text, run.brief)
                   if base else None),
        "draft_text": draft_text,
        "edit_sections": draft_edit_service.to_fields(draft_text, run.brief),
        "revisions": _human_revisions(run),
    }
    if request.GET.get("partial") == "1":
        return render(request, "portal/_draft_pane.html", context)
    return render(request, "portal/draft_detail.html", context)


@login_required
def draft_edit(request, pk):
    """Save the draft as the user retyped it, paragraph by paragraph."""
    run = get_object_or_404(_my_runs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:draft_detail", pk=pk)

    base = _chosen_version(run, request.POST.get("base_version"))
    if base is None:
        messages.error(request, "這篇稿子還沒有內容可以編輯。")
        return redirect("portal:draft_detail", pk=pk)

    # Captions ride inside the `[[img:N|…]]` tokens, so a caption edit changes
    # the text like any other edit and needs no special case here — it saves a
    # new version and leaves older ones with the captions they were saved with.
    after = draft_edit_service.from_fields(base.text, request.POST, run.brief)
    if after.strip() == base.text.strip():
        messages.info(request, "內容沒有變動，未儲存。")
        return redirect("portal:draft_detail", pk=pk)

    version = draft_service.add_version(run, after, source="manual", parent=base)
    messages.success(request, f"已依 {base.label} 存成 {version.label}。"
                              f"{base.label} 仍然留著，可以從上方切回去看。")
    return redirect("portal:draft_detail", pk=pk)


@login_required
def draft_revise(request, pk):
    """Ask the model to rewrite one version against the user's notes."""
    run = get_object_or_404(_my_runs(request), pk=pk)
    if request.method != "POST":
        return redirect("portal:draft_detail", pk=pk)

    feedback = (request.POST.get("feedback") or "").strip()
    if not feedback:
        messages.error(request, "請寫下要修改的地方。")
        return redirect("portal:draft_detail", pk=pk)

    base = _chosen_version(run, request.POST.get("base_version"))
    if base is None:
        messages.error(request, "這篇稿子還沒有內容可以重寫。")
        return redirect("portal:draft_detail", pk=pk)
    if run.in_progress:
        messages.info(request, "這篇稿子還在處理中，等它完成再試。")
        return redirect("portal:draft_detail", pk=pk)

    # Same queue as generation: it is the same size of model call, and the same
    # reason not to hold a request open for it.
    if not runner.submit_rewrite(run, base, feedback):
        messages.error(request, "目前排隊的稿件已達上限，請等前面的產完再試。")
        return redirect("portal:draft_detail", pk=pk)
    messages.success(request, f"已排入重寫，依據 {base.label}。寫好會出現在這一頁。")
    return redirect("portal:draft_detail", pk=pk)


@login_required
def draft_download(request, pk):
    run = get_object_or_404(_my_runs(request), pk=pk)
    # Whichever version is on screen, not simply the newest: someone comparing
    # two of them downloads the one they are looking at.
    version = _chosen_version(run, request.GET.get("v"))
    name = slugify(run.brief.title) or f"draft-{run.pk}"
    text = version.text if version else run.latest_text
    response = HttpResponse(text, content_type="text/markdown; charset=utf-8")
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
