"""Draft generation and the revision loop."""
from __future__ import annotations

import time

from briefs.services.ppt_extract import brief_query_text
from core import llm
from studio.models import GenerationRun, Revision
from studio.services import prompts, retrieval


def run_generation(run: GenerationRun, stop_after_outline: bool = False) -> GenerationRun:
    """Execute one generation run, recording the exact prompt that produced it.

    The prompt is persisted rather than rebuilt on demand: an A/B experiment is
    only meaningful if you can go back and see precisely what each arm was fed.

    Dispatches on `run.mode` so Plan A and Plan B share one entry point and can
    therefore be compared with every other variable held constant.
    """
    if run.mode == "staged":
        from studio.services import pipeline

        return pipeline.run_stages(run, stop_after_outline=stop_after_outline)

    run.status = "running"
    run.save(update_fields=["status"])
    started = time.time()

    try:
        facts = run.brief.facts or {}
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
        user_input = prompts.build_input(facts, exemplars)

        output = llm.complete(
            instructions=instructions, user_input=user_input, timeout=600
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

    return run


def run_revision(run: GenerationRun, feedback: str) -> Revision:
    """Apply editor feedback, producing the next 稿次.

    Each round is stored with the text it revised, which is what makes the
    accumulated history usable later as preference pairs (plan option D).
    """
    last = run.revisions.order_by("-round").first()
    previous_text = last.output if last else run.output
    next_round = (last.round + 1) if last else 2

    revision = Revision.objects.create(
        run=run, round=next_round, feedback=feedback, output=""
    )
    try:
        revision.output = llm.complete(
            instructions=prompts.REVISION_ROLE,
            user_input=prompts.build_revision_input(previous_text, feedback, run.brief.facts or {}),
            timeout=600,
        )
    except Exception as exc:  # noqa: BLE001
        revision.output = f"（修訂失敗）{type(exc).__name__}: {exc}"
    revision.save(update_fields=["output"])
    return revision
