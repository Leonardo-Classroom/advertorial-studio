from django.contrib import admin

from briefs.models import Brief, BriefFacts, BriefSourceFile


class BriefFactsInline(admin.TabularInline):
    """Read-only on purpose: a version records what happened, so editing one
    after the fact would make the run that points at it describe a state that
    never existed."""

    model = BriefFacts
    extra = 0
    fields = ("version", "source", "user_input", "created_at")
    readonly_fields = fields
    can_delete = False


class BriefSourceFileInline(admin.TabularInline):
    """Read-only — a stuck or failed batch is diagnosed here, not edited here."""

    model = BriefSourceFile
    extra = 0
    fields = ("order", "file", "format", "status", "page_count", "error")
    readonly_fields = fields
    can_delete = False


@admin.register(Brief)
class BriefAdmin(admin.ModelAdmin):
    list_display = ("title", "slide_count", "status", "processing", "created_at")
    list_filter = ("status", "processing")
    search_fields = ("title", "raw_text")
    inlines = [BriefSourceFileInline, BriefFactsInline]
