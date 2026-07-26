"""Head-to-head judging across an experiment's arms.

    python manage.py compare_drafts --experiment 1
    python manage.py compare_drafts --runs 2 5

Absolute 1–5 scoring could not separate the arms (every draft scored
title_fit=4, tone_fit=3). This runs every cross-arm pair through a comparative
judge instead, twice per pair with the order swapped, and reports win rates.
Pairs where the verdict flips on swap are counted as ties, because that is
position bias rather than a preference.
"""
from __future__ import annotations

import itertools
import sys
from collections import Counter

from django.core.management.base import BaseCommand

from core import llm
from studio.models import Experiment, GenerationRun, PairwiseComparison
from studio.services import evaluate as E


class Command(BaseCommand):
    help = "對實驗中不同策略的稿件做兩兩對比評審"

    def add_arguments(self, parser):
        parser.add_argument("--experiment", type=int, default=None, help="Experiment ID")
        parser.add_argument("--runs", type=int, nargs="+", default=None,
                            help="直接指定要兩兩比較的 run ID")
        parser.add_argument("--cross-arm-only", action="store_true", default=True,
                            help="只比較不同檢索策略之間（預設開啟）")
        parser.add_argument("--all-pairs", action="store_true",
                            help="連同組內也比較")
        parser.add_argument("--use-latest", action="store_true",
                            help="改用最新一稿而非初稿。比較檢索策略時不要開——"
                                 "修訂過的稿件會贏在修訂而不是策略。")

    def handle(self, *args, **opts):
        if opts["runs"]:
            runs = list(GenerationRun.objects.filter(pk__in=opts["runs"], status="done"))
            experiment = runs[0].experiment if runs else None
        elif opts["experiment"]:
            experiment = Experiment.objects.filter(pk=opts["experiment"]).first()
            if experiment is None:
                self.stderr.write(self.style.ERROR(f"找不到實驗 #{opts['experiment']}"))
                sys.exit(1)
            runs = list(experiment.runs.filter(status="done"))
        else:
            self.stderr.write(self.style.ERROR("請指定 --experiment 或 --runs"))
            sys.exit(1)

        if len(runs) < 2:
            self.stderr.write(self.style.ERROR("至少需要兩篇完成的稿件才能比較"))
            sys.exit(1)

        pairs = [
            (a, b) for a, b in itertools.combinations(runs, 2)
            if opts["all_pairs"] or a.retrieval_strategy != b.retrieval_strategy
        ]
        if not pairs:
            self.stderr.write(self.style.WARNING(
                "沒有跨策略的配對可比。若要連同組內一起比，加 --all-pairs。"
            ))
            sys.exit(0)

        revised = [r.pk for r in runs if r.revisions.exists()]
        if revised and opts["use_latest"]:
            self.stdout.write(self.style.WARNING(
                f"⚠ run {revised} 已有修訂稿，而 --use-latest 會拿修訂稿參賽，"
                "結果會反映修訂而非檢索策略。"
            ))

        which = "最新一稿" if opts["use_latest"] else "初稿"
        self.stdout.write(f"共 {len(pairs)} 組配對，比較{which}，每組判斷兩次（正反順序）…\n")
        wins: Counter = Counter()
        inconsistent = 0

        for i, (a, b) in enumerate(pairs, 1):
            guide = a.style_guide.content if a.style_guide else ""
            self.stdout.write(f"[{i}/{len(pairs)}] run#{a.pk}({a.retrieval_strategy}) "
                              f"vs run#{b.pk}({b.retrieval_strategy}) …", ending=" ")
            self.stdout.flush()
            try:
                result = E.pairwise_judge(a, b, guide, use_latest=opts["use_latest"])
            except Exception as exc:  # noqa: BLE001
                self.stdout.write(self.style.ERROR(f"失敗：{exc}"))
                continue

            PairwiseComparison.objects.create(
                experiment=experiment, run_a=a, run_b=b,
                winner=result["winner"],
                position_consistent=result["position_consistent"],
                dimension_winners=result["dimension_winners"],
                reasoning=result["reasoning"],
                model=llm.current_model(),
            )

            if not result["position_consistent"]:
                inconsistent += 1
            if result["winner"] == "a":
                wins[a.retrieval_strategy] += 1
                self.stdout.write(self.style.SUCCESS(f"→ {a.retrieval_strategy}"))
            elif result["winner"] == "b":
                wins[b.retrieval_strategy] += 1
                self.stdout.write(self.style.SUCCESS(f"→ {b.retrieval_strategy}"))
            else:
                wins["tie"] += 1
                self.stdout.write("→ 平手")

        decided = sum(v for k, v in wins.items() if k != "tie")
        self.stdout.write("\n" + self.style.SUCCESS("=== 對比結果 ==="))
        for strategy, n in wins.most_common():
            label = "平手／順序不一致" if strategy == "tie" else strategy
            share = f"{n / len(pairs):.0%}"
            self.stdout.write(f"  {label:<20} {n:>3} 勝  ({share})")

        self.stdout.write(
            f"\n共 {len(pairs)} 組，其中 {inconsistent} 組在調換順序後判斷翻轉"
            f"（{inconsistent / len(pairs):.0%}）——這部分已計為平手。"
        )
        if decided == 0:
            self.stdout.write(self.style.WARNING(
                "所有配對都判平手：對比式評審在這批稿件上同樣沒有鑑別力，"
                "代表兩種策略的產出確實難分高下，而不是評審方法的問題。"
            ))
        elif inconsistent / len(pairs) > 0.5:
            self.stdout.write(self.style.WARNING(
                "超過半數配對受順序影響，這批判斷的可信度低，建議增加樣本再看。"
            ))
