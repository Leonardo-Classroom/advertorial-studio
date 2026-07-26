"""Automatic self-revision: rewrite a draft from the LLM judge's own critique.

Motivated by a measured result rather than a hunch — feeding the judge's ten
`concrete_fixes` back as editor feedback moved advertorial_completeness 2→4 and
reads_as_human 2→4 on run#2. This just automates that loop and parameterises
how many times it runs.

Two guards, both from things that actually went wrong:

  * **Fact loss.** The first manual pass through this loop deleted all six KOL
    names, because the judge advised removing unconfirmed information and the
    model obliged. Coverage fell 1.00 → 0.70. Any iteration that loses ground
    on mandatory facts is recorded but not accepted.
  * **Blind iteration.** More rewrites are not automatically better. The loop
    stops as soon as a rewrite fails to improve, rather than spending the whole
    budget and handing back whatever came out last. Measured: across twelve
    runs a *second* rewrite was accepted zero times, which is why the default
    is one.

Every iteration is stored as a Revision, so the whole chain stays inspectable
and doubles as preference-pair data (plan option D) — tagged `source="auto"`
so machine-generated pairs are never confused with human editorial judgement.
"""
from __future__ import annotations

import statistics

from studio.models import GenerationRun, Revision
from studio.services import evaluate as evaluate_service

MAX_REWRITES = 3


def _judge_mean(evaluation) -> float | None:
    scores = [v for _, v in evaluation.judge_dimensions]
    return statistics.mean(scores) if scores else None


def _coverage(evaluation) -> float | None:
    return (evaluation.fact_coverage or {}).get("coverage")


def refine(run: GenerationRun, progress=None) -> list[dict]:
    """Rewrite `run` up to `run.max_rewrites` times after the first draft.

    Returns one record per draft describing what happened and why the loop
    continued or stopped — the reasoning is as useful as the scores when
    deciding whether more rewrites are worth allowing.
    """
    from studio.services import generate as generate_service

    rewrites = max(0, min(run.max_rewrites or 0, MAX_REWRITES))
    history: list[dict] = []

    current_eval = run.evaluations.filter(revision__isnull=True).first()
    if current_eval is None:
        current_eval = evaluate_service.evaluate(run, run_judge=True)

    history.append({
        "iteration": 1,
        "kind": "初稿",
        "judge_mean": _judge_mean(current_eval),
        "coverage": _coverage(current_eval),
        "accepted": True,
        "note": "",
    })

    for attempt in range(1, rewrites + 1):
        i = attempt + 1  # draft number: the first draft is 1
        fixes = current_eval.judge_fixes
        if not fixes:
            history.append({"iteration": i, "kind": "略過", "judge_mean": None,
                            "coverage": None, "accepted": False,
                            "note": "評審沒有給出可執行的修改建議，停止迭代"})
            break

        if progress:
            progress(attempt, rewrites)

        feedback = ("請依以下編輯意見修訂（這些意見來自文體評審）：\n"
                    + "\n".join(f"{k}. {f}" for k, f in enumerate(fixes, 1)))
        revision = generate_service.run_revision(run, feedback)
        revision.source = "auto"
        revision.save(update_fields=["source"])

        new_eval = evaluate_service.evaluate(run, revision=revision, run_judge=True)
        new_mean, new_cov = _judge_mean(new_eval), _coverage(new_eval)
        old_mean, old_cov = _judge_mean(current_eval), _coverage(current_eval)

        # Guard 1: never trade sanctioned facts for style.
        if new_cov is not None and old_cov is not None and new_cov < old_cov:
            revision.accepted = False
            revision.reject_reason = f"事實覆蓋下降 {old_cov} → {new_cov}"
            revision.save(update_fields=["accepted", "reject_reason"])
            history.append({"iteration": i, "kind": "自動重寫", "judge_mean": new_mean,
                            "coverage": new_cov, "accepted": False,
                            "note": f"未採納：{revision.reject_reason}，停止迭代"})
            break

        # Guard 2: stop when it stops helping.
        if new_mean is not None and old_mean is not None and new_mean <= old_mean:
            revision.accepted = False
            revision.reject_reason = f"評審分數未提升 {old_mean:.2f} → {new_mean:.2f}"
            revision.save(update_fields=["accepted", "reject_reason"])
            history.append({"iteration": i, "kind": "自動重寫", "judge_mean": new_mean,
                            "coverage": new_cov, "accepted": False,
                            "note": f"未採納：{revision.reject_reason}，停止迭代"})
            break

        history.append({"iteration": i, "kind": "自動重寫", "judge_mean": new_mean,
                        "coverage": new_cov, "accepted": True,
                        "note": f"採納（{old_mean:.2f} → {new_mean:.2f}）"
                                if old_mean is not None and new_mean is not None else "採納"})
        current_eval = new_eval

    return history
