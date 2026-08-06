from django.db import models


class Outlet(models.Model):
    """A media outlet whose house style we want to imitate (GQ, COOL-STYLE...)."""

    name = models.CharField("媒體名稱", max_length=64, unique=True)
    slug = models.SlugField("代碼", max_length=64, unique=True)
    is_target = models.BooleanField("列為模仿目標", default=False)
    note = models.TextField("備註", blank=True)

    class Meta:
        verbose_name = verbose_name_plural = "媒體"
        ordering = ["-is_target", "name"]

    def __str__(self):
        return self.name


class Author(models.Model):
    outlet = models.ForeignKey(Outlet, on_delete=models.CASCADE, related_name="authors")
    name = models.CharField("作者", max_length=128)
    article_count = models.IntegerField("文章數", default=0)

    class Meta:
        verbose_name = verbose_name_plural = "作者"
        unique_together = [("outlet", "name")]
        ordering = ["-article_count"]

    def __str__(self):
        return f"{self.outlet.name} / {self.name}"

    @property
    def style_confidence(self) -> str:
        """How much we can trust an author-level style model for this author.

        The plan flags that most GQ bylines have too few articles to model
        individually; this makes that judgement visible in the UI instead of
        silently producing a low-confidence imitation.
        """
        if self.article_count >= 500:
            return "high"
        if self.article_count >= 100:
            return "medium"
        return "low"


class Article(models.Model):
    outlet = models.ForeignKey(Outlet, on_delete=models.CASCADE, related_name="articles")
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name="articles")
    title = models.CharField("標題", max_length=512)
    url = models.TextField("原始網址", blank=True)
    published_on = models.DateField("日期", null=True, blank=True)
    body = models.TextField("內文")
    char_count = models.IntegerField("字數", default=0)
    content_hash = models.CharField(max_length=64, db_index=True)
    source_path = models.TextField()
    # Row offset into the numpy embedding matrix; null means "not indexed yet".
    vector_row = models.IntegerField(null=True, blank=True, db_index=True)
    indexed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = verbose_name_plural = "語料文章"
        ordering = ["-published_on", "-id"]
        indexes = [
            models.Index(fields=["outlet", "published_on"]),
            # Covers retrieval._candidate_rows' filter (outlet, char_count,
            # vector_row not null) so SQLite can answer it from the index
            # alone. Without this it fell back to per-row table lookups —
            # on this project's OneDrive-mounted db.sqlite3, ~200k random
            # lookups for a large outlet cost ~24s on every call.
            models.Index(fields=["outlet", "char_count", "vector_row"]),
        ]

    def __str__(self):
        return self.title[:60]

    @property
    def is_indexed(self) -> bool:
        # Explicit property because row 0 is a valid row and would read as
        # falsy in a template.
        return self.vector_row is not None

    def exemplar_text(self, max_chars: int = 1800) -> str:
        """The form an article takes when injected into a prompt as a style example."""
        body = self.body if len(self.body) <= max_chars else self.body[:max_chars] + "…"
        return f"【標題】{self.title}\n【內文】\n{body}"


class EmbeddingIndex(models.Model):
    """Bookkeeping for one built vector index (one numpy matrix on disk)."""

    name = models.CharField(max_length=64, unique=True)
    model = models.CharField(max_length=64)
    dimensions = models.IntegerField()
    vector_count = models.IntegerField(default=0)
    built_at = models.DateTimeField(auto_now=True)
    note = models.TextField(blank=True)

    class Meta:
        verbose_name = verbose_name_plural = "向量索引"

    def __str__(self):
        return f"{self.name} ({self.vector_count} 筆 / {self.model})"


class StyleGuide(models.Model):
    """An LLM-induced description of a style — component D of the plan.

    Kept editable: the plan (and the GhostWriter paper) both argue the human
    operator must be able to inspect and correct the style description rather
    than trust an opaque extraction.
    """

    outlet = models.ForeignKey(Outlet, on_delete=models.CASCADE, related_name="style_guides")
    author = models.ForeignKey(
        Author, on_delete=models.CASCADE, related_name="style_guides",
        null=True, blank=True,
        help_text="留空代表這是整個媒體的風格指南；填入則為該作者個人風格。",
    )
    content = models.TextField("風格指南（可編輯）")
    sections = models.JSONField("結構化欄位", default=dict, blank=True)
    sample_size = models.IntegerField("取樣文章數", default=0)
    model = models.CharField(max_length=64, blank=True)
    is_active = models.BooleanField("啟用中", default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = verbose_name_plural = "風格指南"
        ordering = ["-created_at"]

    def __str__(self):
        scope = self.author.name if self.author else "全站"
        return f"{self.outlet.name} / {scope} 風格指南"

    @property
    def scope_label(self) -> str:
        return f"{self.outlet.name} · {self.author.name}" if self.author else f"{self.outlet.name} · 全站"
