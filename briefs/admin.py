from django.contrib import admin

from briefs.models import Brief


@admin.register(Brief)
class BriefAdmin(admin.ModelAdmin):
    list_display = ("title", "slide_count", "status", "created_at")
    list_filter = ("status",)
    search_fields = ("title", "raw_text")
