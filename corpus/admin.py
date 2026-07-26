from django.contrib import admin

from corpus.models import Article, Author, EmbeddingIndex, Outlet, StyleGuide


@admin.register(Outlet)
class OutletAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "is_target", "article_total")
    list_filter = ("is_target",)

    @admin.display(description="文章數")
    def article_total(self, obj):
        return obj.articles.count()


@admin.register(Author)
class AuthorAdmin(admin.ModelAdmin):
    list_display = ("name", "outlet", "article_count", "style_confidence")
    list_filter = ("outlet",)
    search_fields = ("name",)


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ("title", "outlet", "author", "published_on", "char_count", "vector_row")
    list_filter = ("outlet", "published_on")
    search_fields = ("title",)
    raw_id_fields = ("author", "outlet")


@admin.register(StyleGuide)
class StyleGuideAdmin(admin.ModelAdmin):
    list_display = ("__str__", "sample_size", "model", "is_active", "updated_at")
    list_filter = ("outlet", "is_active")


@admin.register(EmbeddingIndex)
class EmbeddingIndexAdmin(admin.ModelAdmin):
    list_display = ("name", "model", "dimensions", "vector_count", "built_at")
