from django.db import models

FACT_SCHEMA_HINT = {
    "brand": "",
    "campaign": "",
    "product": [],
    "slogan": "",
    "key_message": [],
    "kol": [],
    "target_audience": "",
    "publish_schedule": "",
    "deliverables": [],
    "mandatory_terms": [],
    "forbidden_terms": [],
}


def mandatory_fact_values(facts: dict) -> list[str]:
    """The facts that must appear verbatim in a draft.

    Shared by the generator, the reviser and the evaluator on purpose: when the
    prompt and the metric disagree about what counts as mandatory, a revision
    can quietly drop required facts and still look like it followed orders.
    That is exactly what happened on the first real revision run — the editor
    feedback said "消化不確定資訊" and the model removed every KOL name, which
    the coverage check caught only after the fact.

    Prose fields (campaign_context and friends) are excluded: they are meant to
    be reworded, so absence there is not an error.
    """
    out: list[str] = []
    for key in ("brand", "slogan"):
        value = facts.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    for key in ("product", "kol", "mandatory_terms"):
        value = facts.get(key)
        if isinstance(value, list):
            out.extend(str(v).strip() for v in value if str(v).strip())
    # Preserve order while removing duplicates.
    seen, unique = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


class Brief(models.Model):
    """An uploaded Media Brief / Proposal deck and the facts extracted from it.

    `facts` is the single source of truth for everything factual in a generated
    draft (brand, product codes, KOL names). The generator is instructed never
    to invent facts outside this dict, and the evaluator checks coverage against
    it — this is the guard against the hallucination risk in the plan.
    """

    STATUS = [
        ("uploaded", "已上傳"),
        ("extracted", "已抽取，待確認"),
        ("confirmed", "已確認"),
    ]

    title = models.CharField("名稱", max_length=200)
    source_file = models.FileField("簡報檔", upload_to="briefs/", blank=True, null=True)
    raw_text = models.TextField("簡報純文字", blank=True)
    slide_count = models.IntegerField("投影片數", default=0)
    facts = models.JSONField("結構化事實（唯一事實來源）", default=dict, blank=True)
    status = models.CharField(max_length=16, choices=STATUS, default="uploaded")
    note = models.TextField("備註", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = verbose_name_plural = "簡報"
        ordering = ["-created_at"]

    def __str__(self):
        return self.title

    def fact_values(self) -> list[str]:
        """Flatten `facts` into checkable strings for the fact-coverage test."""
        out: list[str] = []

        def walk(value):
            if isinstance(value, str):
                if value.strip():
                    out.append(value.strip())
            elif isinstance(value, list):
                for v in value:
                    walk(v)
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v)

        walk(self.facts)
        return out
