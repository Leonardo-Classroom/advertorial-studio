"""Run the exemplar-selection A/B test end to end.

    python manage.py run_experiment --brief 1 --outlet COOL-STYLE \
        --guide 3 --strategies topical random --repeats 3

Generates `repeats` drafts per strategy with everything else held constant,
evaluates each, and prints the per-strategy comparison. This is the experiment
the plan commits to: arXiv:2509.14543 reports that topic-similar exemplar
selection *reduced* style fidelity on English informal text, contradicting the
usual RAG assumption, and that has to be settled on this corpus rather than
taken on faith either way.

Repeats matter: one draft per arm tells you nothing, because run-to-run
variance in generation is large compared with the effect being measured.
"""
from __future__ import annotations

import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from django.core.management.base import BaseCommand
from django.db import connections

from briefs.models import Brief
from corpus.models import Author, Outlet, StyleGuide
from studio.models import Experiment, GenerationRun
from studio.services import evaluate as evaluate_service
from studio.services import generate as generate_service


class Command(BaseCommand):
    help = "對同一份簡報跑多種檢索策略並自動評估，輸出對照表"

    def add_arguments(self, parser):
        parser.add_argument("--brief", type=int, required=True, help="Brief 的 ID")
        parser.add_argument("--outlet", required=True, help="目標媒體名稱")
        parser.add_argument("--author", default=None, help="目標作者（可省略）")
        parser.add_argument("--guide", type=int, default=None, help="StyleGuide 的 ID")
        parser.add_argument("--strategies", nargs="+", default=["topical", "random"],
                            help="要比較的策略，預設 topical random")
        parser.add_argument("--rewrites", nargs="+", type=int, default=None,
                            help="改為比較重寫次數（例如 --rewrites 0 1 2 3）。"
                                 "給了這個就以重寫次數為變因。")
        parser.add_argument("--modes", nargs="+", default=None,
                            help="改為比較生成方式（single / staged）。"
                                 "給了這個就以方式為變因，策略固定用 --strategies 的第一項。")
        parser.add_argument("--fixed-mode", choices=["staged", "single"], default="staged",
                            help="比較檢索策略時，生成方式固定用哪一種（預設 staged）")
        parser.add_argument("--fixed-rewrites", type=int, default=0,
                            help="比較檢索策略時，重寫次數固定為幾次（預設 0，"
                                 "把自動潤稿排除在外才能乾淨地只看檢索的影響）")
        parser.add_argument("--repeats", type=int, default=3, help="每種策略跑幾次")
        parser.add_argument("--exemplars", type=int, default=4, help="範例篇數")
        parser.add_argument("--name", default=None, help="實驗名稱")
        parser.add_argument("--no-judge", action="store_true", help="跳過 LLM 評審（省成本）")
        parser.add_argument("--model", default=None,
                            help="這次生成改用哪個模型（覆寫 .env 的 LLM_MODEL）。"
                                 "評審模型不受影響，仍用 LLM_JUDGE_MODEL，"
                                 "否則換模型等於同時換掉出題者與改題者。")
        parser.add_argument("--concurrency", type=int, default=4,
                            help="同時跑幾篇（預設 4）。實測生成端點在 8 並行下"
                                 "每次延遲不變，瓶頸不在 API；設 1 可回到序列執行。")

    def handle(self, *args, **opts):
        if opts["model"]:
            from django.conf import settings

            self.stdout.write(f"生成模型：{opts['model']}（評審固定用 {settings.LLM_JUDGE_MODEL}）")
            settings.LLM_MODEL = opts["model"]

        try:
            brief = Brief.objects.get(pk=opts["brief"])
            outlet = Outlet.objects.get(name=opts["outlet"])
        except (Brief.DoesNotExist, Outlet.DoesNotExist) as exc:
            self.stderr.write(self.style.ERROR(f"找不到指定的簡報或媒體：{exc}"))
            sys.exit(1)

        author = None
        if opts["author"]:
            author = Author.objects.filter(outlet=outlet, name=opts["author"]).first()
            if author is None:
                self.stderr.write(self.style.ERROR(f"找不到作者「{opts['author']}」"))
                sys.exit(1)

        guide = StyleGuide.objects.filter(pk=opts["guide"]).first() if opts["guide"] else None
        if opts["guide"] and guide is None:
            self.stderr.write(self.style.ERROR(f"找不到 StyleGuide #{opts['guide']}"))
            sys.exit(1)

        # An author-scoped guide implies an author-scoped run. Without this the
        # run would carry an author guide while retrieval drew exemplars from
        # the whole outlet — an author-level experiment quietly testing
        # something else. The portal already derives it this way.
        if author is None and guide is not None and guide.author_id:
            author = guide.author
            self.stdout.write(f"（指南 #{guide.pk} 屬於作者 {author.name}，"
                              f"本次生成與檢索都限定該作者）")

        if brief.status != "confirmed":
            self.stdout.write(self.style.WARNING(
                "⚠ 這份簡報的事實尚未確認。生成仍會進行，但事實正確性沒有把關過。"
            ))

        experiment = Experiment.objects.create(
            name=opts["name"] or f"{brief.title} × {outlet.name} 檢索策略比較",
            description=(
                f"控制變因：簡報 #{brief.pk}、媒體 {outlet.name}、"
                f"作者 {author.name if author else '不指定'}、"
                f"指南 #{guide.pk if guide else '無'}、範例 {opts['exemplars']} 篇。"
                f"每種策略各 {opts['repeats']} 次。"
            ),
        )

        # One variable at a time: either the retrieval strategy varies and the
        # mode is fixed, or the mode varies and the strategy is fixed.
        default_n = 1
        if opts["rewrites"]:
            arms = [(opts["strategies"][0], "staged", n) for n in opts["rewrites"]]
        elif opts["modes"]:
            arms = [(opts["strategies"][0], m, default_n) for m in opts["modes"]]
        else:
            # Hold mode and rewrites at explicit values rather than the model
            # defaults: a retrieval comparison that also lets the rewrite loop
            # run is measuring two things at once.
            arms = [(s, opts["fixed_mode"], opts["fixed_rewrites"])
                    for s in opts["strategies"]]

        results: dict[str, list[dict]] = {}
        jobs = [(strategy, mode, n_iter, arm, i)
                for strategy, mode, n_iter in arms
                for arm in [(f"重寫{n_iter}次" if opts["rewrites"]
                             else mode if opts["modes"] else strategy)]
                for i in range(opts["repeats"])]
        for *_, arm, _i in jobs:
            results.setdefault(arm, [])

        lock = threading.Lock()
        done = [0]

        def execute(job):
            strategy, mode, n_iter, arm, i = job
            try:
                run = GenerationRun.objects.create(
                    brief=brief, outlet=outlet, author=author, style_guide=guide,
                    experiment=experiment, retrieval_strategy=strategy, mode=mode,
                    exemplar_count=opts["exemplars"], max_rewrites=n_iter,
                )
                generate_service.run_generation(run)
                if run.status == "failed":
                    with lock:
                        self.stderr.write(self.style.ERROR(f"  {arm} 失敗：{run.error[:120]}"))
                    return

                last_ok = run.revisions.filter(accepted=True).order_by("-round").first()
                ev = (run.evaluations.filter(revision=last_ok).first() if last_ok
                      else run.evaluations.filter(revision__isnull=True).first())
                if ev is None:
                    ev = evaluate_service.evaluate(run, revision=last_ok,
                                                   run_judge=not opts["no_judge"])
                judge_vals = [v for _, v in ev.judge_dimensions]
                row = {
                    "run": run.pk,
                    "elapsed": run.elapsed_ms,
                    "drafts": run.iteration_count,
                    "style": ev.style_similarity,
                    "overlap": ev.max_overlap,
                    "judge": statistics.mean(judge_vals) if judge_vals else None,
                    "coverage": (ev.fact_coverage or {}).get("coverage"),
                }
                with lock:
                    results[arm].append(row)
                    done[0] += 1
                    line = (f"  [{done[0]}/{len(jobs)}] {arm} run#{run.pk}  "
                            f"風格 {ev.style_similarity}  重疊 {ev.max_overlap}")
                    if judge_vals:
                        line += f"  評審 {statistics.mean(judge_vals):.2f}"
                    self.stdout.write(line)
                    self.stdout.flush()
            finally:
                # Each worker thread opens its own connection; leaving them open
                # exhausts SQLite's handles over a long experiment.
                connections.close_all()

        workers = max(1, opts["concurrency"])
        self.stdout.write(f"共 {len(jobs)} 篇，並行度 {workers}…\n")
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(execute, jobs))
        else:
            for job in jobs:
                execute(job)

        self.stdout.write("\n" + self.style.SUCCESS(f"=== 實驗 #{experiment.pk} 結果 ==="))
        arm_label = "重寫次數" if opts["rewrites"] else "方式" if opts["modes"] else "策略"
        self.stdout.write(
            f"{arm_label:<10}{'樣本':>5}{'實際稿數':>10}{'風格相似度':>13}"
            f"{'重疊率':>11}{'評審均分':>11}{'事實覆蓋':>11}{'耗時秒':>9}"
        )
        for arm, rows in results.items():
            if not rows:
                self.stdout.write(f"{arm:<12}{0:>4}   （全部失敗）")
                continue
            self.stdout.write(
                f"{arm:<10}{len(rows):>5}"
                f"{self._avg(rows, 'drafts'):>10}"
                f"{self._avg(rows, 'style'):>13}"
                f"{self._avg(rows, 'overlap'):>11}"
                f"{self._avg(rows, 'judge'):>11}"
                f"{self._avg(rows, 'coverage'):>11}"
                f"{self._avg(rows, 'elapsed', 1000):>9}"
            )

        self.stdout.write(self.style.WARNING(
            f"\n每組 n={opts['repeats']}。樣本這麼少時，數字差異多半仍在雜訊範圍內，"
            "請搭配人工評分一起判讀，不要只憑這張表下結論。"
        ))
        self.stdout.write(f"詳細比較：/experiments/{experiment.pk}/")

    @staticmethod
    def _avg(rows: list[dict], key: str, divide: float = 1) -> str:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return "—"
        mean = statistics.mean(vals) / divide
        return f"{mean:.2f}" if divide != 1 or key == "drafts" else f"{mean:.4f}"
