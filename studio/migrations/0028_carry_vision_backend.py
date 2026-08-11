"""Give the new vision settings the values the old single switch implied.

Splitting text and vision apart adds `vision_backend` with its own default. On
an existing install that default is simply wrong: one switch used to drive both,
so a deployment running everything locally would silently start sending pictures
to an online endpoint the moment this field appeared — a behaviour change nobody
asked for, and an expensive one if that endpoint bills per image.

Copying the old value across keeps the split invisible until someone chooses to
use it, which is the only honest way to introduce it.

Not reversible in any meaningful sense: the fields it writes to disappear when
the previous migration is unapplied, so reversing is a no-op.
"""
from django.db import migrations


def carry(apps, schema_editor):
    SiteSettings = apps.get_model("studio", "SiteSettings")
    for row in SiteSettings.objects.all():
        row.vision_backend = row.llm_backend
        row.vision_online_provider_id = row.online_provider_id
        row.save(update_fields=["vision_backend", "vision_online_provider"])


class Migration(migrations.Migration):

    dependencies = [
        ("studio", "0027_sitesettings_vision_backend_and_more"),
    ]

    operations = [
        migrations.RunPython(carry, migrations.RunPython.noop),
    ]
