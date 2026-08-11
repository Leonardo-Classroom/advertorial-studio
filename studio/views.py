import random

from django.contrib import messages
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme

from accounts.decorators import staff_required
from core.pagination import paginate
from briefs.models import Brief
from corpus.models import Article, Author, EmbeddingIndex, Outlet, StyleGuide
from studio.models import (
    EMBED_DEVICES, GENERATION_MODES, LLM_BACKENDS, RETRIEVAL_STRATEGIES,
    SELECTABLE_STRATEGIES, Evaluation, Experiment, GenerationRun, OnlineProvider,
    SiteSettings,
)
from studio.services import evaluate as evaluate_service
from studio.services import generate as generate_service


@staff_required
def home(request):
    outlets = Outlet.objects.annotate(n=Count("articles")).order_by("-n")
    index = EmbeddingIndex.objects.filter(name="articles").first()
    indexed = Article.objects.exclude(vector_row__isnull=True).count()
    total = Article.objects.count()

    return render(request, "studio/home.html", {
        "section": "home",
        "outlets": outlets,
        "total_articles": total,
        "total_authors": Author.objects.count(),
        "index": index,
        "indexed": indexed,
        "index_pct": round(indexed / total * 100) if total else 0,
        "guides": StyleGuide.objects.select_related("outlet", "author").filter(is_active=True)[:6],
        "guide_count": StyleGuide.objects.count(),
        "briefs": Brief.objects.all()[:5],
        "brief_count": Brief.objects.count(),
        "runs": GenerationRun.objects.select_related("brief", "outlet")[:8],
        "run_count": GenerationRun.objects.count(),
    })


def _save_online_providers(request, settings_row) -> None:
    """Create, update, delete and select the online endpoints from the form.

    Two independent lists — one for text, one for pictures — each posting rows
    named by use and index (`provider_text_kind_3`, `provider_vision_key_0`).
    They are separate because the same endpoint rarely serves both: a provider
    that writes well may not accept an image at all.

    A row whose `pk` is not in the POST was removed in the browser, so it is
    deleted here — the remove button drops the `<tr>` rather than submitting a
    flag, which keeps "what you see is what gets saved" true without a second
    round trip. A blank trailing row is normal and ignored unless typed into.

    The key is only written when a value is supplied. The page renders just the
    last four characters, so an empty key field means "leave it alone" — not
    "clear it". Without that rule, saving any other change would wipe every key.
    """
    from studio.models import OnlineProvider

    for use, field in (("text", "online_provider"), ("vision", "vision_online_provider")):
        prefix = f"provider_{use}_"
        indexes = sorted({
            key.rsplit("_", 1)[1] for key in request.POST
            if key.startswith(prefix) and key.rsplit("_", 1)[1].isdigit()
        }, key=int)

        wanted = request.POST.get(f"provider_selected_{use}")
        kept, chosen = [], None
        for i in indexes:
            pk = (request.POST.get(f"{prefix}pk_{i}") or "").strip()
            kind = request.POST.get(f"{prefix}kind_{i}") or "openai"
            base_url = (request.POST.get(f"{prefix}base_url_{i}") or "").strip()
            api_key = (request.POST.get(f"{prefix}key_{i}") or "").strip()
            model = (request.POST.get(f"{prefix}model_{i}") or "").strip()

            if not pk and not (base_url or api_key or model):
                continue                      # the empty trailing row
            row = OnlineProvider.objects.filter(pk=pk, use=use).first() if pk else None
            if pk and row is None:
                continue                      # deleted by someone else meanwhile
            row = row or OnlineProvider(use=use)
            row.kind = kind if kind in dict(OnlineProvider.KINDS) else "openai"
            row.base_url, row.model = base_url, model
            if api_key:
                row.api_key = api_key
            row.save()
            kept.append(row.pk)
            if wanted == str(i):
                chosen = row

        OnlineProvider.objects.filter(use=use).exclude(pk__in=kept).delete()
        # Clear the selection only when the row it named is gone; leave the
        # other list's choice untouched.
        if chosen is not None or getattr(settings_row, f"{field}_id") not in kept:
            setattr(settings_row, field, chosen)


def _duration(seconds: int) -> str:
    """Seconds as 秒/分/時. `timesince` rounds to whole minutes, which reads as
    「0 分鐘」 for most of a run's queue wait and hides the difference between a
    job that just landed and one that has been waiting fifty seconds."""
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        return f"{seconds // 60} 分 {seconds % 60} 秒"
    return f"{seconds // 3600} 小時 {seconds % 3600 // 60} 分"


def _ingest_rows() -> list[dict]:
    """Briefs being parsed, with which file is being read right now.

    "處理中" on its own says nothing about a batch that has been running for
    two minutes. The per-file rows already carry the answer — one of them is
    `processing` — so the page can name the file instead of the state.
    """
    from briefs.models import BriefSourceFile

    briefs = list(Brief.objects.filter(processing=True)
                  .select_related("owner").order_by("updated_at"))
    if not briefs:
        return []
    files = BriefSourceFile.objects.filter(brief__in=briefs).order_by("order", "pk")
    by_brief: dict[int, list] = {}
    for source in files:
        by_brief.setdefault(source.brief_id, []).append(source)

    rows = []
    for brief in briefs:
        owned = by_brief.get(brief.pk, [])
        current = next((f for f in owned if f.status == "processing"), None)
        rows.append({
            "brief": brief,
            "done": sum(1 for f in owned if f.status == "done"),
            "failed": sum(1 for f in owned if f.status == "failed"),
            "total": len(owned),
            "current": current.original_filename if current else "",
            # Files are parsed first, then one fact-extraction call. Nothing
            # left in `processing` while the brief still is means the batch has
            # moved on to that call, which is the slow part and worth naming.
            "phase": "抽取內容" if owned and current is None else "解析檔案",
        })
    return rows


def _classify_rows() -> list[dict]:
    """Briefs with a picture-classification pass running, and how far along."""
    from briefs.services import images as image_service

    pks = image_service.active_classifications()
    if not pks:
        return []
    briefs = {b.pk: b for b in Brief.objects.filter(pk__in=pks).select_related("owner")}
    rows = []
    for pk in pks:
        brief = briefs.get(pk)
        if brief is None:
            continue
        progress = image_service.read_classify_progress(pk) or {}
        done, total = progress.get("done", 0), progress.get("total", 0)
        rows.append({"brief": brief, "done": done, "total": total,
                     "pct": round(done / total * 100) if total else 0})
    return rows


@staff_required
def queue(request):
    """What the machine is working on right now, and who is waiting behind it.

    The generation queue is in memory — a list of run ids inside the worker
    pool, which no page can display and a restart erases. What survives is the
    run rows themselves, and `status` plus `created_at` say the same thing: the
    rows in flight, oldest first, *are* the queue in the order the workers will
    take them (`studio.services.runner` is strictly FIFO). Numbering them here
    from the database rather than reading the queue object keeps this page
    honest across a restart, where the rows persist as `pending` but the
    in-memory queue behind them does not.

    Ingestion runs on its own threads with no queue at all, so it appears as a
    separate list: those jobs are not waiting for a generation worker, but they
    are competing for the same GPU, which is what someone looking at this page
    actually wants to know.
    """
    context = _queue_context()
    if request.GET.get("fragment"):
        # Same context, just the part that changes — see studio/queue.html.
        return render(request, "studio/_queue_body.html", context)
    return render(request, "studio/queue.html", context)


def _queue_context() -> dict:
    """Everything the 隊列 page shows. Shared by the page and its refresh."""
    from core import llm
    from studio.services import runner

    in_flight = list(
        GenerationRun.objects
        .filter(status__in=("pending", "running", "refining"))
        .select_related("owner", "brief", "outlet", "author", "style_guide")
        .order_by("created_at"))
    # Same sweep the draft pages do: without it a run killed by a restart sits
    # here as 排隊中 forever and makes the queue look longer than it is.
    runner.reap_stale(in_flight)

    now = timezone.now()
    rows, place = [], 0
    for run in in_flight:
        if not run.in_progress:
            continue                    # just reaped
        waiting = run.status == "pending"
        if waiting:
            place += 1
        since = run.created_at if waiting else (run.started_at or run.created_at)
        stages = run.stage_labels if run.mode == "staged" else []
        rows.append({
            "run": run,
            "waiting": waiting,
            "place": place if waiting else None,
            "elapsed": _duration(int((now - since).total_seconds())),
            "stage": stages[-1] if stages else "",
        })

    model_active, model_limit = llm.gate_state()
    return {
        "section": "queue",
        "rows": rows,
        "running": [r for r in rows if not r["waiting"]],
        "waiting": [r for r in rows if r["waiting"]],
        "ingesting": _ingest_rows(),
        "classifying": _classify_rows(),
        "parallel": SiteSettings.load().max_parallel_runs,
        "queue_max": runner.QUEUE_MAX,
        "model_active": model_active,
        "model_limit": model_limit,
        "model_local": llm.is_local_backend(),
        "health": llm.backend_health(),
        # Rendered into the fragment so the page can be seen to be live. With
        # nothing running, every refresh returns identical markup and the page
        # looks frozen — indistinguishable from a poll that has stopped.
        "updated_at": timezone.localtime(now).strftime("%H:%M:%S"),
    }


@staff_required
def advanced(request):
    """Generation defaults, in one place, for the people allowed to change them.

    These left the draft forms because nobody writing a draft could choose
    between them usefully. They did not become constants: the measured case for
    switching retrieval to `none` (same quality, ~20% faster, 14% cheaper) is
    strong enough to want to try in production and weak enough to want to undo
    without a deploy.
    """
    from django.conf import settings as django_settings

    settings_row = SiteSettings.load()

    # The model names come from settings rather than from the choice labels, so
    # this page cannot go on naming a model that was swapped out months ago.
    # Only the local option names its model. Which online model is in use is
    # decided by the list right below the dropdown and shown there; naming one
    # here would duplicate it and — worse — the name shown would be the `.env`
    # fallback, which is not what the selected row says. Local has no such
    # list, so this is the only place its model appears.
    def _label(value, label, local_model):
        return f"{label}，使用 {local_model}" if value == "local" else label

    backends = [(v, _label(v, label, django_settings.LOCAL_LLM_MODEL))
                for v, label in LLM_BACKENDS]
    vision_backends = [(v, _label(v, label, django_settings.LOCAL_LLM_VISION_MODEL))
                       for v, label in LLM_BACKENDS]

    if request.method == "POST":
        mode = request.POST.get("default_mode", settings_row.default_mode)
        if mode in dict(GENERATION_MODES):
            settings_row.default_mode = mode
        backend = request.POST.get("llm_backend", settings_row.llm_backend)
        if backend in dict(LLM_BACKENDS):
            settings_row.llm_backend = backend
        vision = request.POST.get("vision_backend", settings_row.vision_backend)
        if vision in dict(LLM_BACKENDS):
            settings_row.vision_backend = vision
        strategy = request.POST.get("retrieval_strategy", settings_row.retrieval_strategy)
        if strategy in dict(SELECTABLE_STRATEGIES):
            settings_row.retrieval_strategy = strategy
        try:
            settings_row.exemplar_count = max(0, min(int(request.POST.get("exemplar_count") or 4), 10))
            settings_row.max_rewrites = max(0, min(int(request.POST.get("max_rewrites") or 1), 3))
            settings_row.max_parallel_runs = max(
                1, min(int(request.POST.get("max_parallel_runs") or 2), 8))
            settings_row.local_model_concurrency = max(
                1, min(int(request.POST.get("local_model_concurrency") or 1), 8))
            # Floors of 30/10/30 seconds: below those the model cannot finish
            # even when everything goes right, so the setting would only
            # manufacture failures. The staleness thresholds follow these
            # automatically (see core.timeouts), so no pairing to keep in sync.
            settings_row.generate_timeout_seconds = max(
                30, min(int(request.POST.get("generate_timeout_seconds") or 600), 3600))
            settings_row.vision_timeout_seconds = max(
                10, min(int(request.POST.get("vision_timeout_seconds") or 120), 1800))
            settings_row.extract_timeout_seconds = max(
                30, min(int(request.POST.get("extract_timeout_seconds") or 240), 3600))
            settings_row.ollama_idle_unload_minutes = max(
                0, min(int(request.POST.get("ollama_idle_unload_minutes") or 3), 120))
            # Floors of 1, not 0: a zero here would read as "no limit" in
            # `source_extract.limits()` and silently fall back to the .env
            # value, which is the opposite of what typing 0 looks like it does.
            settings_row.upload_max_file_mb = max(
                1, min(int(request.POST.get("upload_max_file_mb") or 200), 10240))
            settings_row.upload_max_batch_mb = max(
                1, min(int(request.POST.get("upload_max_batch_mb") or 600), 20480))
            settings_row.upload_max_unpacked_mb = max(
                1, min(int(request.POST.get("upload_max_unpacked_mb") or 2048), 51200))
            settings_row.upload_max_compression_ratio = max(
                2, min(int(request.POST.get("upload_max_compression_ratio") or 200), 100000))
        except ValueError:
            messages.error(request, "數值格式不正確，未儲存。")
            return redirect("studio:advanced")
        # Reset the cached embedder only when the device actually changes, so a
        # save that leaves it alone does not needlessly evict a warm model.
        prev_device = settings_row.embed_device
        device = request.POST.get("embed_device", prev_device)
        if device in dict(EMBED_DEVICES):
            settings_row.embed_device = device
        if settings_row.embed_device != prev_device:
            from core import local_embeddings

            local_embeddings.reset()
        _save_online_providers(request, settings_row)
        settings_row.confirm_fact_updates = request.POST.get("confirm_fact_updates") == "on"
        # Unchecked checkboxes are simply absent from a POST, so these read as
        # False when switched off — no separate hidden field needed.
        settings_row.show_briefs_nav = request.POST.get("show_briefs_nav") == "on"
        settings_row.show_runs_nav = request.POST.get("show_runs_nav") == "on"
        settings_row.save()
        messages.success(request, "已更新產稿預設值。之後建立的稿件會採用新設定，既有紀錄不受影響。")
        return redirect("studio:advanced")

    return render(request, "studio/advanced.html", {
        "section": "advanced",
        "settings": settings_row,
        "strategies": SELECTABLE_STRATEGIES,
        "modes": GENERATION_MODES,
        "backends": backends,
        "vision_backends": vision_backends,
        "embed_devices": EMBED_DEVICES,
        "text_providers": list(OnlineProvider.objects.filter(use="text")),
        "vision_providers": list(OnlineProvider.objects.filter(use="vision")),
        "provider_kinds": OnlineProvider.KINDS,
    })


@staff_required
def runs(request):
    qs = GenerationRun.objects.select_related("brief", "outlet", "author", "experiment")
    return render(request, "studio/runs.html", {
        "section": "runs",
        **paginate(request, qs, 30),
    })


@staff_required
def run_new(request):
    briefs = Brief.objects.all()
    outlets = Outlet.objects.filter(is_target=True) or Outlet.objects.all()

    if request.method == "POST":
        brief = get_object_or_404(Brief, pk=request.POST.get("brief"))
        outlet = get_object_or_404(Outlet, pk=request.POST.get("outlet"))
        author_id = request.POST.get("author") or None
        guide_id = request.POST.get("style_guide") or None
        experiment_id = request.POST.get("experiment") or None

        # The three tuning knobs now come from the site defaults rather than the
        # form. Sweeping them is what `run_experiment` is for; doing it by hand
        # here only ever produced runs nobody could later account for.
        defaults = SiteSettings.load()

        run = GenerationRun.objects.create(
            owner=request.user,
            brief=brief,
            facts_version=brief.latest_facts(),
            outlet=outlet,
            author_id=author_id or None,
            style_guide_id=guide_id or None,
            experiment_id=experiment_id or None,
            mode=request.POST.get("mode", defaults.default_mode),
            retrieval_strategy=defaults.retrieval_strategy,
            exemplar_count=defaults.exemplar_count,
            max_rewrites=max(0, min(defaults.max_rewrites, 3)),
        )
        generate_service.run_generation(run)
        if run.status == "failed":
            messages.error(request, f"生成失敗：{run.error}")
        else:
            messages.success(request, f"生成完成，耗時 {run.elapsed_ms / 1000:.1f} 秒。")
        return redirect("studio:run_detail", pk=run.pk)

    return render(request, "studio/run_new.html", {
        "section": "runs",
        "briefs": briefs,
        "outlets": outlets,
        "authors": Author.objects.select_related("outlet").filter(
            outlet__is_target=True, article_count__gte=100).order_by("-article_count"),
        "guides": StyleGuide.objects.select_related("outlet", "author").filter(is_active=True),
        "experiments": Experiment.objects.all(),
        "modes": GENERATION_MODES,
        "defaults": SiteSettings.load(),
        "preselect_brief": request.GET.get("brief") or "",
    })


@staff_required
def run_detail(request, pk):
    run = get_object_or_404(
        GenerationRun.objects.select_related("brief", "outlet", "author", "style_guide"), pk=pk
    )
    return render(request, "studio/run_detail.html", {
        "section": "runs",
        "run": run,
        "revisions": run.revisions.all(),
        "evaluations": run.evaluations.select_related("revision").all(),
        # Same brief only: comparing drafts written from different briefs would
        # be comparing subject matter, not the thing under test.
        "comparable": GenerationRun.objects.filter(
            brief=run.brief, status="done").exclude(pk=run.pk).order_by("-pk")[:30],
    })


@staff_required
def run_outline(request, pk):
    """Save an edited outline, and optionally write the draft from it."""
    run = get_object_or_404(GenerationRun, pk=pk)
    if request.method != "POST":
        return redirect("studio:run_detail", pk=pk)

    run.outline = request.POST.get("outline", run.outline)
    run.outline_approved = True
    run.save(update_fields=["outline", "outline_approved"])

    if request.POST.get("action") == "generate":
        from studio.services import pipeline

        pipeline.continue_from_outline(run)
        if run.status == "failed":
            messages.error(request, f"生成失敗：{run.error}")
        else:
            messages.success(request, "已依大綱寫出正文。")
    else:
        messages.success(request, "大綱已儲存。")
    return redirect("studio:run_detail", pk=pk)


@staff_required
def run_evaluate(request, pk):
    run = get_object_or_404(GenerationRun, pk=pk)
    if request.method != "POST":
        return redirect("studio:run_detail", pk=pk)

    revision_id = request.POST.get("revision") or None
    revision = run.revisions.filter(pk=revision_id).first() if revision_id else None
    try:
        ev = evaluate_service.evaluate(run, revision=revision,
                                       run_judge=request.POST.get("judge") == "on")
    except Exception as exc:  # noqa: BLE001
        messages.error(request, f"評估失敗：{exc}")
        return redirect("studio:run_detail", pk=pk)

    if ev.overlap_warning:
        messages.warning(
            request,
            f"⚠ 與範例文章的字串重疊率達 {ev.max_overlap:.1%}（來源：{ev.overlap_source}），"
            "有照抄疑慮，請人工檢查後要求改寫。",
        )
    if ev.missing_facts:
        messages.warning(request, f"⚠ 簡報事實有 {len(ev.missing_facts)} 項沒出現在稿件中："
                                  + "、".join(ev.missing_facts[:6]))
    messages.success(request, "評估完成。")
    return redirect("studio:run_detail", pk=pk)


@staff_required
def run_revise(request, pk):
    run = get_object_or_404(GenerationRun, pk=pk)
    if request.method != "POST":
        return redirect("studio:run_detail", pk=pk)

    feedback = (request.POST.get("feedback") or "").strip()
    if not feedback:
        messages.error(request, "請填寫修改意見。")
        return redirect("studio:run_detail", pk=pk)

    revision = generate_service.run_revision(run, feedback)
    messages.success(request, f"已產出第 {revision.round} 稿。")
    return redirect("studio:run_detail", pk=pk)


@staff_required
def run_score(request, pk):
    """Record the human verdict — the third leg of the metric ensemble."""
    run = get_object_or_404(GenerationRun, pk=pk)
    if request.method != "POST":
        return redirect("studio:run_detail", pk=pk)

    ev = run.evaluations.first()
    if ev is None:
        ev = Evaluation.objects.create(run=run)
    ev.human_score = int(request.POST.get("human_score") or 0) or None
    if request.POST.get("human_comment"):
        ev.human_comment = request.POST["human_comment"]
    ev.save(update_fields=["human_score", "human_comment"])
    messages.success(request, f"已記錄 run#{run.pk} 的人工評分：{ev.human_score}/5")

    # Return to wherever the score was entered — scoring happens on the
    # side-by-side page as often as on the run page, and bouncing the user
    # away from a comparison they are half-way through is annoying.
    back = request.META.get("HTTP_REFERER")
    if back and url_has_allowed_host_and_scheme(
        back, allowed_hosts={request.get_host()}, require_https=request.is_secure()
    ):
        return redirect(back)
    return redirect("studio:run_detail", pk=pk)


@staff_required
def compare(request):
    """Side-by-side draft comparison, blind by default.

    Blind because the whole point is to check whether a human can tell the
    arms apart; knowing which is which while judging defeats that. Left/right
    order is randomised deterministically from the pair, so reloading does not
    reshuffle and the same pair always presents the same way.

    Real corpus articles are shown alongside on purpose: the reliable question
    to ask is not "how good is this draft" in the abstract but "which of these
    two reads more like the published pieces next to them".
    """
    a = get_object_or_404(GenerationRun, pk=request.GET.get("a"))
    b = get_object_or_404(GenerationRun, pk=request.GET.get("b"))
    reveal = request.GET.get("reveal") == "1"

    # Deterministic but non-obvious side assignment.
    flip = ((a.pk * 31 + b.pk * 17) % 2) == 1
    left, right = (b, a) if flip else (a, b)

    rng = random.Random(a.pk * 1000 + b.pk)
    ref_ids = list(
        Article.objects.filter(outlet=a.outlet, char_count__gte=800)
        .order_by().values_list("id", flat=True)[:3000]
    )
    references = Article.objects.filter(
        id__in=rng.sample(ref_ids, min(3, len(ref_ids)))
    ).select_related("author") if ref_ids else []

    def label(run):
        # The strategy's display text carries its own "(方案 B 預設)" note, which
        # reads as a contradiction next to a 方案 A mode label. Use the bare code.
        return f"{run.get_mode_display().split('：')[0]} · 檢索 {run.retrieval_strategy}"

    return render(request, "studio/compare.html", {
        "section": "runs",
        "left": left, "right": right, "reveal": reveal,
        "left_label": label(left) if reveal else "稿件 甲",
        "right_label": label(right) if reveal else "稿件 乙",
        "references": references,
        "a": a, "b": b,
    })


@staff_required
def experiments(request):
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        if name:
            Experiment.objects.create(name=name, description=request.POST.get("description", ""))
            messages.success(request, "已建立實驗。接著在「新增生成」時把不同策略的稿件掛到這個實驗底下。")
        return redirect("studio:experiments")

    return render(request, "studio/experiments.html", {
        "section": "experiments",
        "experiments": Experiment.objects.annotate(n=Count("runs")),
    })


@staff_required
def experiment_detail(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    runs_qs = experiment.runs.select_related("outlet", "brief").prefetch_related("evaluations")

    # Aggregate per strategy so the topical-vs-random question gets a number.
    by_strategy = (
        runs_qs.values("retrieval_strategy")
        .annotate(
            n=Count("id"),
            style=Avg("evaluations__style_similarity"),
            overlap=Avg("evaluations__max_overlap"),
            human=Avg("evaluations__human_score"),
        )
        .order_by("retrieval_strategy")
    )

    rows = []
    for row in by_strategy:
        judge_avgs = []
        for run in runs_qs.filter(retrieval_strategy=row["retrieval_strategy"]):
            for ev in run.evaluations.all():
                scores = [v for _, v in ev.judge_dimensions]
                if scores:
                    judge_avgs.append(sum(scores) / len(scores))
        row["judge"] = round(sum(judge_avgs) / len(judge_avgs), 2) if judge_avgs else None
        row["label"] = dict(RETRIEVAL_STRATEGIES).get(row["retrieval_strategy"],
                                                      row["retrieval_strategy"])
        rows.append(row)

    # Pairwise tally. Ties and order-inconsistent verdicts are kept visible
    # rather than folded away: if most pairs are ties, that is the finding.
    comparisons = experiment.comparisons.select_related("run_a", "run_b").all()
    tally: dict[str, int] = {}
    ties = inconsistent = 0
    for c in comparisons:
        if not c.position_consistent or c.winner == "tie":
            ties += 1
            inconsistent += 0 if c.position_consistent else 1
            continue
        winner_run = c.run_a if c.winner == "a" else c.run_b
        tally[winner_run.retrieval_strategy] = tally.get(winner_run.retrieval_strategy, 0) + 1

    return render(request, "studio/experiment_detail.html", {
        "section": "experiments",
        "experiment": experiment,
        "runs": runs_qs,
        "rows": rows,
        "comparisons": comparisons,
        "pairwise_tally": sorted(tally.items(), key=lambda kv: -kv[1]),
        "pairwise_total": len(comparisons),
        "pairwise_ties": ties,
        "pairwise_inconsistent": inconsistent,
    })
