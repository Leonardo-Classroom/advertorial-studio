"""Give every already-extracted brief a v1, so the portal has something to show.

The portal now selects facts by version. Briefs uploaded before versioning have
their facts in `Brief.facts` and no version rows at all, which would leave the
version dropdown empty and the generate button pointing at nothing. This copies
what is already there into v1 — no model call, no change to any value.
"""
from django.db import migrations


def seed(apps, schema_editor):
    Brief = apps.get_model("briefs", "Brief")
    BriefFacts = apps.get_model("briefs", "BriefFacts")

    for brief in Brief.objects.exclude(facts={}).exclude(facts=None):
        if BriefFacts.objects.filter(brief=brief).exists():
            continue
        BriefFacts.objects.create(
            brief=brief, version=1, data=brief.facts, source="extract",
        )


def unseed(apps, schema_editor):
    apps.get_model("briefs", "BriefFacts").objects.filter(version=1).delete()


class Migration(migrations.Migration):
    dependencies = [("briefs", "0006_brieffacts")]
    operations = [migrations.RunPython(seed, unseed)]
