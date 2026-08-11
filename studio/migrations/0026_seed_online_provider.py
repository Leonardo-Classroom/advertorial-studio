"""Carry the existing `.env` online endpoint into the new editable list.

Without this the list starts empty and the page shows nothing, even though the
deployment already has a working online endpoint configured — the operator
would have to retype what the system already knows.

**The key is deliberately not copied.** Two reasons, and the second is the one
that matters: the endpoint and model are configuration worth showing, but a key
copied here would silently create a second place the secret lives, with no
prompt to review it. Leaving it blank makes the row incomplete, which means
`core.llm._online_config()` keeps using `.env` exactly as before — nothing
changes until someone deliberately types a key in.

Reversing drops only rows this migration created and left untouched.
"""
from django.conf import settings
from django.db import migrations


def seed(apps, schema_editor):
    OnlineProvider = apps.get_model("studio", "OnlineProvider")
    if OnlineProvider.objects.exists():
        return
    base_url = getattr(settings, "LLM_BASE_URL", "") or ""
    model = getattr(settings, "LLM_MODEL", "") or ""
    if not (base_url or model):
        return
    OnlineProvider.objects.create(
        kind="openai", base_url=base_url, model=model, api_key="")


def unseed(apps, schema_editor):
    OnlineProvider = apps.get_model("studio", "OnlineProvider")
    OnlineProvider.objects.filter(api_key="").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("studio", "0025_alter_sitesettings_llm_backend"),
    ]

    operations = [
        migrations.RunPython(seed, unseed),
    ]
