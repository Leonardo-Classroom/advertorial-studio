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

from django.core.management.base import BaseCommand

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
        parser.add_argument("--repeats", type=int, default=3, help="每種策略跑幾次")
        parser.add_argument("--exemplars", type=int, default=4, help="範例篇數")
        parser.add_argument("--name", default=None, help="實驗名稱")
        parser.add_argument("--no-judge", action="store_true", help="跳過 LLM 評審（省成本）")

    def handle(self, *args, **opts):
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

        results: dict[str, list[dict]] = {}
        for strategy in opts["strategies"]:
            results[strategy] = []
            for i in range(opts["repeats"]):
                self.stdout.write(f"生成中 {strategy} #{i + 1}/{opts['repeats']} …")
                run = GenerationRun.objects.create(
                    brief=brief, outlet=outlet, author=author, style_guide=guide,
                    experiment=experiment, retrieval_strategy=strategy,
                    exemplar_count=opts["exemplars"],
                )
                generate_service.run_generation(run)
                if run.status == "failed":
                    self.stderr.write(self.style.ERROR(f"  失敗：{run.error}"))
                    continue

                ev = evaluate_service.evaluate(run, run_judge=not opts["no_judge"])
                judge_vals = [v for _, v in ev.judge_dimensions]
                results[strategy].append({
                    "run": run.pk,
                    "style": ev.style_similarity,
                    "overlap": ev.max_overlap,
                    "judge": statistics.mean(judge_vals) if judge_vals else None,
                    "coverage": (ev.fact_coverage or {}).get("coverage"),
                })
                line = (f"  run#{run.pk}  風格 {ev.style_similarity}  "
                        f"重疊 {ev.max_overlap}")
                if judge_vals:
                    line += f"  評審 {statistics.mean(judge_vals):.2f}"
                self.stdout.write(line)

        self.stdout.write("\n" + self.style.SUCCESS(f"=== 實驗 #{experiment.pk} 結果 ==="))
        self.stdout.write(f"{'策略':<12}{'n':>4}{'風格相似度':>14}{'重疊率':>12}{'評審均分':>12}{'事實覆蓋':>12}")
        for strategy, rows in results.items():
            if not rows:
                self.stdout.write(f"{strategy:<12}{0:>4}   （全部失敗）")
                continue
            self.stdout.write(
                f"{strategy:<12}{len(rows):>4}"
                f"{self._avg(rows, 'style'):>14}"
                f"{self._avg(rows, 'overlap'):>12}"
                f"{self._avg(rows, 'judge'):>12}"
                f"{self._avg(rows, 'coverage'):>12}"
            )

        self.stdout.write(self.style.WARNING(
            f"\n每組 n={opts['repeats']}。樣本這麼少時，數字差異多半仍在雜訊範圍內，"
            "請搭配人工評分一起判讀，不要只憑這張表下結論。"
        ))
        self.stdout.write(f"詳細比較：/experiments/{experiment.pk}/")

    @staticmethod
    def _avg(rows: list[dict], key: str) -> str:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return f"{statistics.mean(vals):.4f}" if vals else "—"
