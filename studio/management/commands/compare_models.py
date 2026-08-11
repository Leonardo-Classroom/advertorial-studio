"""Compare online models on speed and judged quality.

    python manage.py compare_models --models deepseek-v4-flash,deepseek-v4-pro \\
        --briefs 198,199,200,201 --judge deepseek-v4-pro --rewrites 1

Exists because the same comparison kept being rewritten as a throwaway script,
and each rewrite made the same two mistakes: it switched
`SiteSettings.online_provider` to change model — process-global, so nothing
could overlap — and it scored one draft at a time even though scoring is
embarrassingly parallel. The second cost twenty minutes of pure waiting on an
eight-draft run.

Two rules the measurements depend on:

* **Generation stays sequential** unless `--concurrent` is passed. Eight drafts
  in flight at one provider queue against each other, and the per-draft time
  then measures contention rather than the model. Sequential is the correct
  method for a speed comparison, not a limitation.
* **Scoring is always concurrent.** The judge is fixed, the calls are
  independent, and nothing about scoring is timed — there is no reason to wait.

The judge never varies with the candidate. A model grading its own draft
produces numbers that cannot be compared across candidates, which is the whole
point of the exercise.
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from django.core.management.base import BaseCommand, CommandError
from django.db import connections

from briefs.models import Brief
from core import llm
from studio.models import GenerationRun, Revision, SiteSettings

DIMS = ("title_fit", "tone_fit", "rhythm_fit", "diction_fit",
        "advertorial_completeness", "reads_as_human")


def _mean(scores: dict) -> float | None:
    picked = [scores[d] for d in DIMS if isinstance(scores.get(d), (int, float))]
    return round(sum(picked) / len(picked), 2) if len(picked) == len(DIMS) else None


class Command(BaseCommand):
    help = "比較多個線上模型的速度與評分（評審固定，不讓模型評自己的稿）"

    def add_arguments(self, parser):
        parser.add_argument("--models", required=True,
                            help="逗號分隔的模型名稱，需同屬目前選定的線上端點")
        parser.add_argument("--briefs", required=True, help="逗號分隔的 brief id")
        parser.add_argument("--judge", required=True, help="評分用的模型，固定不變")
        parser.add_argument("--rewrites", type=int, default=1,
                            help="max_rewrites。0 = 只量初稿；1 = 含評分後重寫（預設）")
        parser.add_argument("--modes", default="single",
                            help="逗號分隔：single,staged。兩個都給就一起比")
        parser.add_argument("--thinking", default="",
                            help="逗號分隔的推理強度：off,low,medium,high。"
                                 "留空表示不指定，用供應商預設。DeepSeek 只分 off 與其他")
        parser.add_argument("--judge-effort", dest="judge_effort", default="",
                            help="評審用的推理強度，固定不變")
        parser.add_argument("--concurrent", action="store_true",
                            help="併發產稿。會讓耗時數字失去意義，只在不比速度時使用")
        parser.add_argument("--workers", type=int, default=6, help="評分併發數")
        parser.add_argument("--guide", type=int, default=0,
                            help="StyleGuide pk。不給就沿用該專案既有的風格指南")
        parser.add_argument("--json", dest="json_path", default="",
                            help="把逐筆結果寫成 JSON")

    def handle(self, *args, **opts):
        models = [m.strip() for m in opts["models"].split(",") if m.strip()]
        briefs = [int(b) for b in opts["briefs"].split(",") if b.strip()]
        judge = opts["judge"].strip()

        row = SiteSettings.load()
        if row.llm_backend != "online":
            raise CommandError("文字後端不是線上模型——這個指令只比較線上端點上的模型。")
        endpoint = llm._online_config("text")
        self._guide = None
        if opts["guide"]:
            from corpus.models import StyleGuide

            self._guide = StyleGuide.objects.filter(pk=opts["guide"]).select_related(
                "outlet", "author").first()
            if self._guide is None:
                raise CommandError(f"找不到 StyleGuide {opts['guide']}。")
        modes = [m.strip() for m in opts["modes"].split(",") if m.strip()]
        efforts = [t.strip() for t in opts["thinking"].split(",") if t.strip()] or [None]
        unknown = [e for e in efforts if e and e not in llm.EFFORTS]
        if unknown:
            raise CommandError(f"不認得的推理強度 {unknown}，可用：{list(llm.EFFORTS)}")

        self.stdout.write(
            f"端點 {endpoint.base_url}｜候選 {models}｜評審 {judge}｜"
            f"專案 {briefs}｜max_rewrites={opts['rewrites']}"
            f"｜方案 {modes}｜推理強度 {efforts}"
            f"｜評審強度 {opts['judge_effort'] or '（預設）'}"
            f"｜風格指南 {self._guide or '（沿用專案原有）'}")

        results = self._generate(models, briefs, modes, efforts, opts)
        self._score(results, judge, opts["judge_effort"] or None, opts["workers"])
        self._report(results, models)
        if opts["json_path"]:
            with open(opts["json_path"], "w", encoding="utf8") as fh:
                json.dump(results, fh, ensure_ascii=False, indent=1)
            self.stdout.write(f"逐筆結果已寫入 {opts['json_path']}")

    # ---- generation ------------------------------------------------------

    def _one_run(self, model: str, brief_pk: int, mode: str, effort, opts) -> dict:
        from studio.services import generate as generate_service

        brief = Brief.objects.get(pk=brief_pk)
        template = GenerationRun.objects.filter(brief=brief).exclude(style_guide=None).first()
        if template is None:
            raise CommandError(f"brief {brief_pk} 沒有可沿用的風格指南。")
        # `--guide` swaps the target publication. Outlet and author come from
        # the guide, not the template: a guide belongs to one publication, and
        # pairing it with another's outlet would label the run with a媒體 it
        # was not written for.
        guide = self._guide or template.style_guide
        run = GenerationRun.objects.create(
            owner=template.owner, brief=brief, facts_version=brief.latest_facts(),
            outlet=guide.outlet, author=guide.author,
            style_guide=guide, mode=mode,
            retrieval_strategy=SiteSettings.load().retrieval_strategy,
            exemplar_count=SiteSettings.load().exemplar_count,
            max_rewrites=opts["rewrites"])

        started = time.time()
        ok = False
        try:
            with llm.model_override(model, effort=effort):
                generate_service.run_generation(run)
            run.refresh_from_db()
            ok = run.status == "done"
        except Exception as exc:  # noqa: BLE001 - one candidate failing is data
            self.stdout.write(self.style.WARNING(
                f"  {model} brief{brief_pk}: {type(exc).__name__}: {str(exc)[:90]}"))
        elapsed = time.time() - started
        refine_ms = next((s.get("elapsed_ms") for s in (run.stages or [])
                          if s.get("name") == "refine"), 0) or 0
        revision = run.revisions.filter(accepted=True, source="auto").order_by("-round").first()
        tag = f"{model}/{mode}/{effort or 'default'}"
        self.stdout.write(
            f"  {tag:44s} brief{brief_pk}  {elapsed:6.1f}s"
            f"（重寫 {refine_ms / 1000:5.1f}s）  {'成功' if ok else '失敗'}"
            f"  {len(run.output or ''):5d} 字  重寫{'採納' if revision else '未採納'}")
        return {"model": model, "mode": mode, "effort": effort, "tag": tag,
                "brief": brief_pk, "run": run.pk, "ok": ok,
                "secs": round(elapsed, 1), "refine_s": round(refine_ms / 1000, 1),
                "revision": revision.pk if revision else None,
                "first_chars": len(run.output or ""),
                "final_chars": len(run.latest_text or "")}

    def _generate(self, models, briefs, modes, efforts, opts) -> list[dict]:
        jobs = [(m, b, mode, effort)
                for m in models for mode in modes for effort in efforts for b in briefs]
        self.stdout.write(
            f"\n[產稿] {len(jobs)} 篇"
            f"（{'併發 ' + str(opts['workers']) if opts['concurrent'] else '序列——耗時才有意義'}）")
        if not opts["concurrent"]:
            return [self._one_run(*job, opts) for job in jobs]

        def worker(job):
            try:
                return self._one_run(*job, opts)
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=min(len(jobs), opts["workers"])) as pool:
            return list(pool.map(worker, jobs))

    # ---- scoring ---------------------------------------------------------

    def _score(self, results: list[dict], judge: str, judge_effort, workers: int) -> None:
        from studio.services import evaluate as evaluate_service

        todo = [r for r in results if r["ok"]]
        self.stdout.write(f"\n[評分] {len(todo)} 篇 × 2（初稿／重寫後），評審 {judge}"
                          f"（強度 {judge_effort or '預設'}），併發 {workers}")

        def score_one(record):
            try:
                run = GenerationRun.objects.get(pk=record["run"])
                with llm.model_override(judge, effort=judge_effort):
                    first = evaluate_service.evaluate(run, run_judge=True)
                    record["first"] = _mean(first.judge_scores or {})
                    record["judge_model"] = first.judge_model
                    if record["revision"]:
                        revision = Revision.objects.get(pk=record["revision"])
                        last = evaluate_service.evaluate(run, revision=revision, run_judge=True)
                        record["final"] = _mean(last.judge_scores or {})
                    else:
                        record["final"] = record["first"]   # nothing was adopted
                if record["first"] is not None and record["final"] is not None:
                    record["delta"] = round(record["final"] - record["first"], 2)
                self.stdout.write(
                    f"  {record['tag']:44s} brief{record['brief']}  "
                    f"初稿 {record.get('first')} → 重寫後 {record.get('final')}"
                    f"  ({record.get('delta', 'n/a')})")
            except Exception as exc:  # noqa: BLE001 - a failed score is not a crash
                record["score_error"] = f"{type(exc).__name__}: {exc}"[:160]
                self.stdout.write(self.style.WARNING(
                    f"  {record['tag']} brief{record['brief']} 評分失敗：{exc}"))
            finally:
                connections.close_all()

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(score_one, todo))

    # ---- report ----------------------------------------------------------

    def _report(self, results: list[dict], models: list[str]) -> None:
        self.stdout.write("\n=== 彙總 ===")
        for model in sorted({r["tag"] for r in results}):
            rows = [r for r in results if r["tag"] == model and r["ok"]
                    and r.get("first") is not None]
            if not rows:
                self.stdout.write(f"  {model:44s} 無有效樣本")
                continue
            deltas = [r["delta"] for r in rows if r.get("delta") is not None]
            line = (f"  {model:44s} n={len(rows)}"
                    f"｜總時中位 {statistics.median(r['secs'] for r in rows):6.1f}s"
                    f"（重寫 {statistics.median(r['refine_s'] for r in rows):5.1f}s）"
                    f"｜初稿 {statistics.median(r['first'] for r in rows):.2f}"
                    f" → 重寫後 {statistics.median(r['final'] for r in rows):.2f}")
            if deltas:
                line += f"｜提升中位 {statistics.median(deltas):+.2f}"
            self.stdout.write(line)
        adopted = sum(1 for r in results if r.get("revision"))
        done = sum(1 for r in results if r["ok"])
        self.stdout.write(f"  重寫被採納：{adopted}/{done} 篇")
