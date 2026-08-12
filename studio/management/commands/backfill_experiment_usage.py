"""Write `TokenUsage` rows for a run that finished before the table existed.

The DS-flash vs luna comparison measured every call's tokens and wrote them to
JSON, but `TokenUsage` did not exist yet — so the cost pages showed those four
projects as blank while the report quoted exact figures for them. This reads
the experiment's own JSON back in.

These are the measured numbers, not reconstructions: the script recorded what
each provider reported at the time. What *cannot* be recovered this way is
anything the script did not save, so a JSON without `judge_usage` backfills
generation and extraction only and leaves judging missing rather than guessing.

Idempotent per (purpose, owner): a purpose that already has rows for a brief or
a run is left alone, so re-running after adding a field does not double the
bill. `created_at` is forced back to when the work actually happened —
`auto_now_add` would otherwise date a year of history to the afternoon of the
backfill.

Expected JSON: a list of objects with `brief`, `run`, `model`, `usage`,
`ingest_usage`, and optionally `judge_usage` keyed by arm name.
"""
from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from core import pricing
from studio.models import TokenUsage

# The experiment names its arms; TokenUsage stores the model that was called.
ARM_MODELS = {
    "luna-none": "gpt-5.6-luna",
    "DS-flash-thinking": "deepseek-v4-flash",
}


def _row(purpose, model, usage, brief_id=None, run_id=None):
    inp = usage.get("input", 0)
    out = usage.get("output", 0)
    cached = usage.get("cached_input", 0)
    return TokenUsage(
        purpose=purpose, model=model, kind="",
        brief_id=brief_id, run_id=run_id,
        input_tokens=inp, output_tokens=out,
        reasoning_tokens=usage.get("reasoning", 0), cached_input_tokens=cached,
        cost_usd=pricing.cost(model, inp, out, cached),
        priced=pricing.is_priced(model),
    )


class Command(BaseCommand):
    help = "把實驗 JSON 裡量到的 token 用量補寫成 TokenUsage 紀錄"

    def add_arguments(self, parser):
        parser.add_argument("--json", required=True, help="實驗結果 JSON 的路徑")
        parser.add_argument("--dry-run", action="store_true", help="只顯示要寫入什麼")

    def handle(self, *args, **opts):
        path = Path(opts["json"])
        if not path.exists():
            raise CommandError(f"找不到 {path}")
        entries = json.loads(path.read_text(encoding="utf8"))

        from briefs.models import Brief
        from studio.models import GenerationRun

        # Which owners already have rows, so a re-run adds nothing twice.
        have_extract = set(TokenUsage.objects.filter(purpose="extract")
                           .values_list("brief_id", flat=True))
        have_generate = set(TokenUsage.objects.filter(purpose="generate")
                            .values_list("run_id", flat=True))
        have_judge = set(TokenUsage.objects.filter(purpose="judge")
                         .values_list("run_id", flat=True))

        rows: list[tuple[TokenUsage, int]] = []   # row, run id it is dated from
        seen_briefs: set[int] = set()

        for e in entries:
            brief_id, run_id = e.get("brief"), e.get("run")
            model = e.get("model") or ARM_MODELS.get(e.get("arm"), "")
            if not e.get("ok", True):
                continue

            # Extraction is per project, and every draft of that project carries
            # a copy of it in the JSON — one row per brief, not per draft.
            ingest = e.get("ingest_usage")
            if (ingest and brief_id and brief_id not in seen_briefs
                    and brief_id not in have_extract):
                seen_briefs.add(brief_id)
                rows.append((_row("extract", model, ingest, brief_id=brief_id), run_id))

            if e.get("usage") and run_id and run_id not in have_generate:
                rows.append((_row("generate", model, e["usage"],
                                  brief_id=brief_id, run_id=run_id), run_id))

            if run_id and run_id not in have_judge:
                for arm, usage in (e.get("judge_usage") or {}).items():
                    judge_model = ARM_MODELS.get(arm, arm)
                    rows.append((_row("judge", judge_model, usage,
                                      brief_id=brief_id, run_id=run_id), run_id))

        if not rows:
            self.stdout.write("沒有要補的紀錄（可能已經補過了）。")
            return

        total = sum(r.cost_usd for r, _ in rows)
        by_purpose: dict[str, list[float]] = {}
        for r, _ in rows:
            by_purpose.setdefault(r.purpose, []).append(r.cost_usd)
        for purpose, costs in by_purpose.items():
            self.stdout.write(f"  {purpose:9s} {len(costs):3d} 筆  ${sum(costs):.5f}")
        self.stdout.write(f"  {'合計':9s} {len(rows):3d} 筆  ${total:.5f}")

        if opts["dry_run"]:
            self.stdout.write(self.style.WARNING("dry-run，未寫入。"))
            return

        TokenUsage.objects.bulk_create([r for r, _ in rows])

        # Date each row from the work it belongs to. Without this every
        # backfilled call looks like it happened during the backfill, and the
        # 逐次呼叫 list on the cost pages reads as one enormous burst.
        when = dict(GenerationRun.objects.filter(
            pk__in={rid for _, rid in rows if rid}).values_list("pk", "created_at"))
        for row, run_id in rows:
            stamp = when.get(run_id)
            if stamp:
                TokenUsage.objects.filter(pk=row.pk).update(created_at=stamp)

        self.stdout.write(self.style.SUCCESS(f"已補寫 {len(rows)} 筆，共 ${total:.5f}。"))
        for b in Brief.objects.filter(pk__in=seen_briefs).order_by("pk"):
            spend = sum(r.cost_usd for r in TokenUsage.objects.filter(brief=b))
            self.stdout.write(f"    專案 {b.pk} {b.title[:36]}  ${spend:.5f}")
