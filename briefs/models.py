from django.conf import settings
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


def required_fact_values(facts: dict) -> list[str]:
    """Facts a draft must contain to be considered complete.

    Narrower than it used to be. It once included every KOL name, and one Oct
    deck listed 28 of them — so coverage demanded the copy name all 28, which
    is not an article, it is a roster. Talent is now *available* material, not
    a checklist: a piece featuring two of the KOLs well beats one that mentions
    all of them.
    """
    out: list[str] = []
    brand = facts.get("brand")
    if isinstance(brand, str) and brand.strip():
        out.append(brand.strip())
    for key in ("product", "mandatory_terms"):
        value = facts.get(key)
        if isinstance(value, list):
            out.extend(str(v).strip() for v in value if str(v).strip())

    seen, unique = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def preservable_fact_values(facts: dict) -> list[str]:
    """Everything a revision must not silently delete.

    Wider than `required_fact_values`: a rewrite should not be free to drop a
    KOL the draft already named just because the judge asked it to tighten
    things up — that is how the first revision run lost all six names — but nor
    should the first draft be obliged to name every one of them.
    """
    out = list(required_fact_values(facts))
    slogan = facts.get("slogan")
    if isinstance(slogan, str) and slogan.strip():
        out.append(slogan.strip())
    kol = facts.get("kol")
    if isinstance(kol, list):
        out.extend(str(v).strip() for v in kol if str(v).strip())

    seen, unique = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


# Kept as the old name so existing callers keep working.
mandatory_fact_values = required_fact_values


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

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                              null=True, blank=True, related_name="briefs",
                              verbose_name="上傳者")
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
