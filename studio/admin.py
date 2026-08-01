from django.contrib import admin

from studio.models import (
    Evaluation, Experiment, GenerationRun, PairwiseComparison, Revision,
    SiteSettings,
)


@admin.register(SiteSettings)
class SiteSettingsAdmin(admin.ModelAdmin):
    """Also editable at /manage/advanced/, which is where staff actually go."""

    def has_add_permission(self, request):
        return not SiteSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


class RevisionInline(admin.TabularInline):
    model = Revision
    extra = 0


class EvaluationInline(admin.TabularInline):
    model = Evaluation
    extra = 0
    fk_name = "run"


@admin.register(GenerationRun)
class GenerationRunAdmin(admin.ModelAdmin):
    list_display = ("__str__", "retrieval_strategy", "exemplar_count",
                    "status", "elapsed_ms", "created_at")
    list_filter = ("status", "retrieval_strategy", "outlet", "experiment")
    inlines = [RevisionInline, EvaluationInline]


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("name", "created_at")


@admin.register(PairwiseComparison)
class PairwiseComparisonAdmin(admin.ModelAdmin):
    list_display = ("__str__", "winner", "position_consistent", "created_at")
    list_filter = ("winner", "position_consistent", "experiment")


@admin.register(Evaluation)
class EvaluationAdmin(admin.ModelAdmin):
    list_display = ("run", "style_similarity", "max_overlap", "human_score", "created_at")
    list_filter = ("run__outlet",)
