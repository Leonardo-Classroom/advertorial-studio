"""Give every existing brief a BriefSourceFile, so the new aggregate model has
something to aggregate.

Before this, a Brief was exactly one file: `source_file` plus the `raw_text`/
`slide_count` that came out of it. This copies that single file into the new
per-file row rather than reprocessing anything — the text has already been
extracted once, there is no reason to pay for it again. A brief with no file
(four rows predate file storage; they carry `raw_text`/`slide_count` from
before uploads were even saved) gets a fileless row instead of a fabricated
one. A brief whose extraction produced no text (`The_Grey_Manifesto`, the one
row where `raw_text` is empty despite a `source_file` existing) is marked
`failed` rather than `done` with nothing in it — it did fail, historically,
and pretending otherwise would misrepresent what happened.

Every `BriefImage` under a brief is repointed at that brief's one new source
row — unambiguous, since before this migration a brief only ever had one.
"""
from django.db import migrations


def backfill(apps, schema_editor):
    Brief = apps.get_model("briefs", "Brief")
    BriefSourceFile = apps.get_model("briefs", "BriefSourceFile")
    BriefImage = apps.get_model("briefs", "BriefImage")

    for brief in Brief.objects.all():
        has_text = bool(brief.raw_text.strip())
        source_file = BriefSourceFile.objects.create(
            brief=brief,
            file=brief.source_file.name or None,
            format="pptx",
            order=0,
            raw_text=brief.raw_text,
            page_count=brief.slide_count,
            status="done" if has_text else "failed",
            error="" if has_text else "沒有檔案，且沒有已抽取的文字（歷史資料）。",
        )
        BriefImage.objects.filter(brief=brief).update(source_file=source_file)


def unbackfill(apps, schema_editor):
    apps.get_model("briefs", "BriefSourceFile").objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ("briefs", "0011_alter_briefimage_slide_index_briefsourcefile_and_more"),
    ]
    operations = [migrations.RunPython(backfill, unbackfill)]
