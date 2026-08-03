"""Give every finished run a draft v1, and carry its human revisions forward.

The portal now selects copy by version. Runs that finished before versioning
have their text on the run row and their later drafts as `Revision` rows, so
without this the version dropdown would be empty and the page would show
nothing. Nothing is recomputed and no model is called — the existing texts are
copied into versions, in the order they were written.

Auto-rewrites stay out of it: they are the generator finishing its own job, and
they were never separate drafts to the person reading them.
"""
from django.db import migrations


def seed(apps, schema_editor):
    GenerationRun = apps.get_model("studio", "GenerationRun")
    Revision = apps.get_model("studio", "Revision")
    DraftVersion = apps.get_model("studio", "DraftVersion")

    for run in GenerationRun.objects.filter(status="done"):
        if DraftVersion.objects.filter(run=run).exists():
            continue

        last_auto = (Revision.objects.filter(run=run, source="auto", accepted=True)
                     .order_by("-round").first())
        text = last_auto.output if last_auto else run.output
        if not (text or "").strip():
            continue

        parent = DraftVersion.objects.create(
            run=run, version=1, text=text, source="generate")

        later = Revision.objects.filter(
            run=run, accepted=True, source__in=["human", "manual"]).order_by("round")
        for i, revision in enumerate(later, start=2):
            if not (revision.output or "").strip():
                continue
            parent = DraftVersion.objects.create(
                run=run, version=i, text=revision.output,
                source="manual" if revision.source == "manual" else "revise",
                user_input="" if revision.source == "manual" else revision.feedback,
                parent=parent,
            )


def unseed(apps, schema_editor):
    apps.get_model("studio", "DraftVersion").objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [("studio", "0013_draftversion")]
    operations = [migrations.RunPython(seed, unseed)]
