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

    def usable_images(self):
        """Approved pictures, in slide order — the only ones a draft may place."""
        return self.images.filter(approved=True)


class BriefImage(models.Model):
    """One picture pulled out of the deck, with the evidence needed to judge it.

    Pictures are on the same footing as the extracted facts: they come from the
    brief itself, so there is no borrowed-source problem, but they are equally
    subject to the rule that nothing reaches a draft until a human has approved
    it. `approved` starts at whatever the model guessed and is then confirmed or
    overridden on the review page.
    """

    CATEGORIES = [
        ("usable", "可用素材"),
        ("layout", "表格／排版"),
        ("decoration", "logo／裝飾"),
        ("unknown", "未判斷"),
    ]

    brief = models.ForeignKey(Brief, on_delete=models.CASCADE, related_name="images")
    slide_index = models.IntegerField("投影片頁次", default=0)
    file = models.ImageField("圖片", upload_to="brief_images/")

    # Two hashes, deliberately. md5 catches the byte-identical copy-paste;
    # phash catches the same asset after rescaling or re-encoding, which md5
    # cannot see. Kept on the row so a later cross-deck library can reuse them.
    md5 = models.CharField("精確雜湊", max_length=32, blank=True, db_index=True)
    phash = models.CharField("感知雜湊", max_length=16, blank=True, db_index=True)

    slide_heading = models.CharField("投影片標題", max_length=200, blank=True)
    nearby_text = models.TextField(
        "鄰近文字（依位置推測）", blank=True,
        help_text="同張投影片上距離最近的文字。這是空間推測，不是簡報作者標註的圖說。")

    ai_description = models.TextField("AI 描述", blank=True)
    ai_category = models.CharField("AI 分類", max_length=16, choices=CATEGORIES, default="unknown")
    ai_brand = models.CharField(
        "圖中辨識到的品牌", max_length=80, blank=True,
        help_text="用來擋掉競品照。提案簡報前段常放對手商品做市場分析，"
                  "那些不屬於本篇素材。")
    ai_caption = models.CharField("AI 建議圖說", max_length=200, blank=True)

    approved = models.BooleanField("可用於稿件", default=False)
    caption = models.CharField("圖說", max_length=200, blank=True,
                               help_text="留空就用 AI 建議的圖說。")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "簡報圖片"
        ordering = ["slide_index", "pk"]

    def __str__(self):
        return f"{self.brief.title} 第{self.slide_index}張 #{self.pk}"

    def display_caption(self) -> str:
        return self.caption.strip() or self.ai_caption.strip()
