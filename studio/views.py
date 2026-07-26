import random

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.http import url_has_allowed_host_and_scheme

from briefs.models import Brief
from corpus.models import Article, Author, EmbeddingIndex, Outlet, StyleGuide
from studio.models import (
    GENERATION_MODES, RETRIEVAL_STRATEGIES, Evaluation, Experiment, GenerationRun,
)
from studio.services import evaluate as evaluate_service
from studio.services import generate as generate_service


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


def runs(request):
    qs = GenerationRun.objects.select_related("brief", "outlet", "author", "experiment")
    return render(request, "studio/runs.html", {
        "section": "runs",
        "page": Paginator(qs, 30).get_page(request.GET.get("page")),
    })


def run_new(request):
    briefs = Brief.objects.all()
    outlets = Outlet.objects.filter(is_target=True) or Outlet.objects.all()

    if request.method == "POST":
        brief = get_object_or_404(Brief, pk=request.POST.get("brief"))
        outlet = get_object_or_404(Outlet, pk=request.POST.get("outlet"))
        author_id = request.POST.get("author") or None
        guide_id = request.POST.get("style_guide") or None
        experiment_id = request.POST.get("experiment") or None

        mode = request.POST.get("mode", "staged")
        pause = mode == "staged" and request.POST.get("pause_at_outline") == "on"

        run = GenerationRun.objects.create(
            brief=brief,
            outlet=outlet,
            author_id=author_id or None,
            style_guide_id=guide_id or None,
            experiment_id=experiment_id or None,
            mode=mode,
            retrieval_strategy=request.POST.get("retrieval_strategy", "typical"),
            exemplar_count=int(request.POST.get("exemplar_count") or 4),
        )
        generate_service.run_generation(run, stop_after_outline=pause)
        if run.status == "failed":
            messages.error(request, f"生成失敗：{run.error}")
        elif pause:
            messages.success(
                request,
                "大綱已產出，尚未寫正文。請往下確認／修改大綱後再按「依大綱寫出正文」——"
                "在大綱階段改方向，比改完稿便宜得多。",
            )
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
        "strategies": RETRIEVAL_STRATEGIES,
        "modes": GENERATION_MODES,
        "preselect_brief": request.GET.get("brief") or "",
    })


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
