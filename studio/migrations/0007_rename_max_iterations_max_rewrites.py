from django.db import migrations, models


def to_rewrites(apps, schema_editor):
    """n (total drafts) -> rewrites (drafts after the first)."""
    Run = apps.get_model("studio", "GenerationRun")
    for run in Run.objects.all():
        Run.objects.filter(pk=run.pk).update(max_rewrites=max((run.max_rewrites or 1) - 1, 0))


def to_iterations(apps, schema_editor):
    Run = apps.get_model("studio", "GenerationRun")
    for run in Run.objects.all():
        Run.objects.filter(pk=run.pk).update(max_rewrites=(run.max_rewrites or 0) + 1)


class Migration(migrations.Migration):
    dependencies = [("studio", "0006_alter_generationrun_status")]

    operations = [
        migrations.RenameField("generationrun", "max_iterations", "max_rewrites"),
        migrations.RunPython(to_rewrites, to_iterations),
        migrations.AlterField(
            model_name="generationrun",
            name="max_rewrites",
            field=models.IntegerField(
                default=1,
                help_text="0 = 只寫初稿不重寫；1 = 初稿評估後自動重寫一次（預設）。上限 3。",
                verbose_name="生成評估後重寫次數",
            ),
        ),
    ]
