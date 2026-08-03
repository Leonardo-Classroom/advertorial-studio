from django.contrib import admin

from briefs.models import Brief, BriefFacts


class BriefFactsInline(admin.TabularInline):
    """Read-only on purpose: a version records what happened, so editing one
    after the fact would make the run that points at it describe a state that
    never existed."""

    model = BriefFacts
    extra = 0
    fields = ("version", "source", "user_input", "created_at")
    readonly_fields = fields
    can_delete = False


@admin.register(Brief)
class BriefAdmin(admin.ModelAdmin):
    list_display = ("title", "slide_count", "status", "created_at")
    list_filter = ("status",)
    search_fields = ("title", "raw_text")
    inlines = [BriefFactsInline]
