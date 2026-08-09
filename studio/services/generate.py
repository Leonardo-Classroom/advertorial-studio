"""Draft generation and the revision loop."""
from __future__ import annotations

import time

from briefs.services.ppt_extract import brief_query_text
from core import llm, timeouts
from studio.models import GenerationRun, Revision
from studio.services import prompts, retrieval


def run_generation(run: GenerationRun, stop_after_outline: bool = False,
                   auto_refine: bool = True) -> GenerationRun:
    """Execute one generation run, recording the exact prompt that produced it.

    The prompt is persisted rather than rebuilt on demand: an A/B experiment is
    only meaningful if you can go back and see precisely what each arm was fed.

    Dispatches on `run.mode` so Plan A and Plan B share one entry point and can
    therefore be compared with every other variable held constant.

    With `run.max_rewrites >= 1` the draft is then rewritten from the judge's
    own critique, that many times at most. Skipped when the run pauses at the
    outline, since there is no draft to refine yet.
    """
    if run.mode == "staged":
        from studio.services import pipeline

        pipeline.run_stages(run, stop_after_outline=stop_after_outline)
        if not stop_after_outline:
            _maybe_refine(run, auto_refine)
        return run

    run.status = "running"
    run.save(update_fields=["status"])
    started = time.time()

    try:
        facts = run.facts
        query = brief_query_text(facts) or run.brief.title

        exemplars = retrieval.retrieve(
            strategy=run.retrieval_strategy,
            outlet_id=run.outlet_id,
            query_text=query,
            count=run.exemplar_count,
            author_id=run.author_id,
            seed=run.pk,  # reproducible random arm
        )

        guide_text = run.style_guide.content if run.style_guide else ""
        instructions = prompts.build_instructions(
            guide_text, run.outlet.name, run.author.name if run.author else None
        )
        user_input = prompts.build_input(facts, exemplars, images=run.brief.usable_images())

        output = llm.complete(
            instructions=instructions, user_input=user_input,
            timeout=timeouts.generate(),
        )

        run.exemplars = [e.as_dict() for e in exemplars]
        run.prompt_instructions = instructions
        run.prompt_input = user_input
        run.output = output
        run.model = llm.current_model()
        run.status = "done"
        run.error = ""
    except Exception as exc:  # noqa: BLE001 - the failure belongs in the record
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        run.elapsed_ms = int((time.time() - started) * 1000)
        run.save()

    _maybe_refine(run, auto_refine)
    return run


def _maybe_refine(run: GenerationRun, auto_refine: bool) -> None:
    if not auto_refine or run.status != "done" or (run.max_rewrites or 0) < 1:
        return
    from studio.services import refine as refine_service

    # elapsed_ms was stamped before this point; rewriting is part of what n
    # costs, so fold it in rather than reporting a time that excludes it.
    refine_started = time.time()
    # Status stays out of "done" until the loop settles: a run that still has
    # rewrites pending has not decided what it delivers yet, and reading it as
    # finished produces numbers that change under you.
    run.status = "refining"
    run.save(update_fields=["status"])

    run.stages = (run.stages or []) + [{
        "name": "refine", "label": "⑥ 依評審意見自動重寫",
        "summary": f"重寫上限 {run.max_rewrites} 次",
        "output": "", "elapsed_ms": 0,
    }]
    history = refine_service.refine(run)
    run.stages[-1]["output"] = "\n".join(
        f"第 {h['iteration']} 輪 {h['kind']}："
        f"評審 {h['judge_mean']}｜覆蓋 {h['coverage']}｜{h['note'] or '採納'}"
        for h in history
    )
    run.stages[-1]["elapsed_ms"] = int((time.time() - refine_started) * 1000)
    run.elapsed_ms += run.stages[-1]["elapsed_ms"]
    run.status = "done"
    run.save(update_fields=["stages", "elapsed_ms", "status"])


def run_revision(run: GenerationRun, feedback: str,
                 previous_text: str | None = None) -> Revision:
    """Apply editor feedback, producing the next 稿次.

    Each round is stored with the text it revised, which is what makes the
    accumulated history usable later as preference pairs (plan option D).

    `previous_text` names the version being revised. The portal passes it
    because the user picks which draft version to work from; without it the
    current best text is used, which is what the staff tooling wants.
    """
    # Revise the current best text, not literally the last one written: a
    # rejected auto-rewrite (one that lost facts or scored worse) is kept for
    # the record but must not become the base for the next round.
    previous_text = previous_text if previous_text is not None else run.latest_text
    last = run.revisions.order_by("-round").first()
    next_round = (last.round + 1) if last else 2

    revision = Revision.objects.create(
        run=run, round=next_round, feedback=feedback, output=""
    )
    try:
        revision.output = llm.complete(
            instructions=prompts.REVISION_ROLE,
            user_input=prompts.build_revision_input(previous_text, feedback, run.facts),
            timeout=timeouts.generate(),
        )
    except Exception as exc:  # noqa: BLE001
        revision.output = f"（修訂失敗）{type(exc).__name__}: {exc}"
    revision.save(update_fields=["output"])
    return revision
