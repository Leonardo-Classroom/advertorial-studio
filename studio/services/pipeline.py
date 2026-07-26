"""Plan B: the multi-stage generation pipeline.

Five stages, following arXiv:2308.07968 (retrieval → ranking → summarization →
synthesis → generation), which reported significant gains over single-shot
generation across several datasets.

Two things this is meant to fix, both observed in Plan A's output rather than
assumed:

  * `advertorial_completeness` scored 2/5 on every one of the eleven Plan A
    first drafts — a systematic structural gap, not variance. Deciding what to
    say and in what order *before* writing is the obvious lever.
  * Plan A drafts leaked proposal-deck language ("引導消費者進店", "推升母檔表現")
    into the copy. The summarize stage translates that vocabulary explicitly,
    ahead of drafting, instead of hoping the writer notices.

Each stage's output is stored on the run, so a staged run can be inspected step
by step — and the outline can be corrected by a human before any prose exists,
which is far cheaper than rewriting a finished draft.
"""
from __future__ import annotations

import time

from briefs.services.ppt_extract import brief_query_text
from core import llm
from studio.models import GenerationRun
from studio.services import prompts, retrieval

STAGE_LABELS = {
    "retrieve": "① 檢索候選範文",
    "rank": "② 依文體代表性重排",
    "summarize": "③ 整理重點筆記",
    "synthesize": "④ 訂出大綱",
    "generate": "⑤ 寫出正文",
}


def _record(run: GenerationRun, name: str, summary: str, output: str, elapsed: float):
    run.stages = (run.stages or []) + [{
        "name": name,
        "label": STAGE_LABELS.get(name, name),
        "summary": summary,
        "output": output,
        "elapsed_ms": int(elapsed * 1000),
    }]


def run_stages(run: GenerationRun, stop_after_outline: bool = False) -> GenerationRun:
    """Execute the staged pipeline.

    With `stop_after_outline` the run halts once the outline exists, leaving it
    for a human to edit and approve before the prose is written. That checkpoint
    is the main practical reason to prefer this pipeline: catching a wrong angle
    at the outline stage costs one edit, catching it in a finished draft costs a
    rewrite.
    """
    run.status = "running"
    run.stages = []
    run.save(update_fields=["status", "stages"])
    started = time.time()

    try:
        facts = run.brief.facts or {}
        guide = run.style_guide
        guide_text = guide.content if guide else ""

        # ① + ② retrieval and re-ranking. The `typical` strategy folds the
        # re-rank in; recording them separately keeps the pipeline legible.
        t = time.time()
        query = brief_query_text(facts) or run.brief.title
        exemplars = retrieval.retrieve(
            strategy=run.retrieval_strategy,
            outlet_id=run.outlet_id,
            query_text=query,
            count=run.exemplar_count,
            author_id=run.author_id,
            seed=run.pk,
        )
        _record(run, "retrieve", f"檢索查詢：{query[:120]}",
                "\n".join(f"- {e.article.title}" for e in exemplars), time.time() - t)
        _record(run, "rank", f"策略 {run.get_retrieval_strategy_display()}",
                "\n".join(f"- {e.reason}｜{e.article.title[:50]}" for e in exemplars), 0)

        # ③ summarize — what to say, in what order, and which deck phrases to translate.
        t = time.time()
        import json
        notes = llm.complete(
            instructions=prompts.SUMMARIZE_ROLE,
            user_input=prompts.SUMMARIZE_TASK.format(
                facts=json.dumps(prompts.writable_facts(facts), ensure_ascii=False, indent=2)),
            timeout=400,
        )
        run.notes = notes
        _record(run, "summarize", "由簡報事實整理", notes, time.time() - t)

        # ④ synthesize — the outline, and the human checkpoint.
        # The outline decides what to leave out, so the mandatory facts have to
        # be named here too. Without it the first staged run dropped a third of
        # them into "刻意不寫" — the same failure mode the revision loop had,
        # just moved one stage earlier.
        from briefs.models import required_fact_values

        t = time.time()
        structure = _structure_hint(guide)
        must_cover = required_fact_values(facts)
        images = list(run.brief.usable_images())
        outline = llm.complete(
            instructions=prompts.SYNTHESIZE_ROLE,
            user_input=prompts.SYNTHESIZE_TASK.format(
                notes=notes,
                structure=structure,
                # The deliverable list is a media-buy artefact; the outline only
                # needs to know it is writing one article.
                deliverables="一篇廣編圖文（含 FB 貼文文案）",
                must_cover="、".join(must_cover) or "（無）",
                images=prompts.build_outline_image_section(images),
            ),
            timeout=400,
        )
        run.outline = outline
        _record(run, "synthesize", "由重點筆記產出", outline, time.time() - t)

        if stop_after_outline:
            run.status = "pending"
            run.model = llm.current_model()
            run.exemplars = [e.as_dict() for e in exemplars]
            run.elapsed_ms = int((time.time() - started) * 1000)
            run.save()
            return run

        run.exemplars = [e.as_dict() for e in exemplars]
        return _generate_from_outline(run, exemplars, guide_text, facts, started)

    except Exception as exc:  # noqa: BLE001 - the failure belongs in the record
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        run.elapsed_ms = int((time.time() - started) * 1000)
        run.save()
        return run


def continue_from_outline(run: GenerationRun) -> GenerationRun:
    """Resume a paused run after a human has edited and approved the outline."""
    started = time.time()
    run.status = "running"
    run.save(update_fields=["status"])
    try:
        facts = run.brief.facts or {}
        guide_text = run.style_guide.content if run.style_guide else ""
        exemplars = _rehydrate_exemplars(run)
        return _generate_from_outline(run, exemplars, guide_text, facts, started)
    except Exception as exc:  # noqa: BLE001
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        run.save()
        return run


def _generate_from_outline(run, exemplars, guide_text, facts, started) -> GenerationRun:
    t = time.time()
    instructions = prompts.build_instructions(
        guide_text, run.outlet.name, run.author.name if run.author else None
    )
    user_input = prompts.build_generate_from_outline(
        facts, run.outline, exemplars, images=run.brief.usable_images())
    output = llm.complete(instructions=instructions, user_input=user_input, timeout=600)

    _record(run, "generate", "依大綱寫出正文", f"（{len(output)} 字）", time.time() - t)
    run.prompt_instructions = instructions
    run.prompt_input = user_input
    run.output = output
    run.model = llm.current_model()
    run.status = "done"
    run.error = ""
    run.elapsed_ms = int((time.time() - started) * 1000)
    run.save()
    return run


def _structure_hint(guide) -> str:
    """Pull just the structural parts of the style guide.

    The outline stage needs to know how the outlet arranges an article; feeding
    it the whole guide would drag tone and diction into a decision that is only
    about shape.
    """
    if guide is None:
        return "（無風格指南，請用一般潮流媒體的常見結構）"
    induced = (guide.sections or {}).get("induced") or {}
    parts = []
    if induced.get("structure"):
        parts.append(f"全文結構：{induced['structure']}")
    for key, label in (("title_formulas", "標題公式"),
                       ("opening_moves", "開場手法"),
                       ("closing_conventions", "結尾慣例")):
        vals = induced.get(key)
        if isinstance(vals, list) and vals:
            parts.append(f"{label}：" + "；".join(str(v) for v in vals[:4]))
    return "\n".join(parts) or guide.content[:2000]


def _rehydrate_exemplars(run) -> list:
    from corpus.models import Article

    ids = [e.get("id") for e in (run.exemplars or []) if e.get("id")]
    if not ids:
        return []
    by_id = {a.id: a for a in Article.objects.filter(id__in=ids).select_related("author")}
    out = []
    for e in run.exemplars or []:
        art = by_id.get(e.get("id"))
        if art:
            out.append(retrieval.Exemplar(art, float(e.get("score") or 0), e.get("reason", "")))
    return out
