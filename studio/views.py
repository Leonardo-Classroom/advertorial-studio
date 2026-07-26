from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404, redirect, render

from briefs.models import Brief
from corpus.models import Article, Author, EmbeddingIndex, Outlet, StyleGuide
from studio.models import RETRIEVAL_STRATEGIES, Evaluation, Experiment, GenerationRun
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

        run = GenerationRun.objects.create(
            brief=brief,
            outlet=outlet,
            author_id=author_id or None,
            style_guide_id=guide_id or None,
            experiment_id=experiment_id or None,
            retrieval_strategy=request.POST.get("retrieval_strategy", "topical"),
            exemplar_count=int(request.POST.get("exemplar_count") or 4),
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
        "strategies": RETRIEVAL_STRATEGIES,
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
    })


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
    ev.human_comment = request.POST.get("human_comment", "")
    ev.save(update_fields=["human_score", "human_comment"])
    messages.success(request, "已記錄人工評分。")
    return redirect("studio:run_detail", pk=pk)


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

    return render(request, "studio/experiment_detail.html", {
        "section": "experiments",
        "experiment": experiment,
        "runs": runs_qs,
        "rows": rows,
    })
