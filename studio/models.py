from django.db import models

from briefs.models import Brief
from corpus.models import Author, Outlet, StyleGuide

RETRIEVAL_STRATEGIES = [
    ("topical", "主題相似檢索（依簡報內容找同題材範文）"),
    ("random", "隨機抽樣（同媒體/作者內隨機）"),
    ("hybrid", "混合（一半主題相似、一半隨機）"),
    ("typical", "主題檢索後依「文體代表性」重排（預設）"),
    ("none", "不放範例（只用風格指南）"),
]

GENERATION_MODES = [
    # Staged is listed first because it is the default: it scored +0.75 on
    # advertorial_completeness and +0.50 on reads_as_human against single-shot,
    # and the operator's own read of the drafts agreed with that direction.
    ("staged", "方案 B：多階段管線（檢索→重排→摘要→大綱→生成）"),
    ("single", "方案 A：單次生成（較快，結構較鬆）"),
]


class Experiment(models.Model):
    """Groups runs that differ by exactly one variable, for A/B comparison.

    Exists because the plan commits to testing whether topic-similar exemplar
    selection actually helps — arXiv:2509.14543 found it *hurt* style fidelity
    on English informal text, and that assumption has to be checked on this
    Chinese fashion-media corpus rather than assumed either way.
    """

    name = models.CharField("實驗名稱", max_length=200)
    description = models.TextField("說明", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "A/B 實驗"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class GenerationRun(models.Model):
    STATUS = [
        ("pending", "待執行"),
        ("running", "生成中"),
        ("done", "完成"),
        ("failed", "失敗"),
    ]

    brief = models.ForeignKey(Brief, on_delete=models.CASCADE, related_name="runs")
    outlet = models.ForeignKey(Outlet, on_delete=models.PROTECT, related_name="runs")
    author = models.ForeignKey(Author, on_delete=models.SET_NULL, null=True, blank=True,
                               related_name="runs", verbose_name="目標作者（可留空）")
    style_guide = models.ForeignKey(StyleGuide, on_delete=models.SET_NULL,
                                    null=True, blank=True, related_name="runs")
    experiment = models.ForeignKey(Experiment, on_delete=models.SET_NULL,
                                   null=True, blank=True, related_name="runs")

    mode = models.CharField("生成方式", max_length=8, choices=GENERATION_MODES, default="staged")
    retrieval_strategy = models.CharField("檢索策略", max_length=16,
                                          choices=RETRIEVAL_STRATEGIES, default="typical")
    exemplar_count = models.IntegerField("範例篇數", default=4)
    exemplars = models.JSONField("實際使用的範例", default=list, blank=True,
                                 help_text="[{id, title, score, author}]")

    # Plan B only. Each stage is recorded so a run can be inspected — and
    # intervened in — rather than being a single opaque call.
    stages = models.JSONField("各階段紀錄", default=list, blank=True)
    notes = models.TextField("重點筆記（摘要階段產出）", blank=True)
    outline = models.TextField("大綱（可在生成正文前人工修改）", blank=True)
    outline_approved = models.BooleanField("大綱已確認", default=False)

    prompt_instructions = models.TextField(blank=True)
    prompt_input = models.TextField(blank=True)
    output = models.TextField("生成結果", blank=True)

    model = models.CharField(max_length=64, blank=True)
    status = models.CharField(max_length=16, choices=STATUS, default="pending")
    error = models.TextField(blank=True)
    elapsed_ms = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "生成紀錄"
        ordering = ["-created_at"]

    def __str__(self):
        return f"#{self.pk} {self.brief.title} → {self.outlet.name}"

    @property
    def latest_text(self) -> str:
        """The newest text for this run — the last revision, else the draft."""
        last = self.revisions.order_by("-round").first()
        return last.output if last else self.output

    @property
    def is_staged(self) -> bool:
        return self.mode == "staged"

    @property
    def stage_list(self) -> list[dict]:
        return self.stages or []


class Revision(models.Model):
    """One turn of the human feedback loop.

    Mirrors the real 初稿 → Feedback → 二稿 workflow documented in 分析報告.md,
    and doubles as the preference-pair data that plan option D (lightweight
    fine-tuning) would later need — no separate annotation effort required.
    """

    run = models.ForeignKey(GenerationRun, on_delete=models.CASCADE, related_name="revisions")
    round = models.IntegerField("稿次", default=2)
    feedback = models.TextField("修改意見")
    output = models.TextField("修訂後內容", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "修訂稿"
        ordering = ["round"]
        unique_together = [("run", "round")]

    def __str__(self):
        return f"{self.run_id} 第 {self.round} 稿"

    @property
    def previous_text(self) -> str:
        prior = self.run.revisions.filter(round__lt=self.round).order_by("-round").first()
        return prior.output if prior else self.run.output


class PairwiseComparison(models.Model):
    """Head-to-head judgement between two drafts.

    Absolute 1–5 scoring turned out to be useless here: across six drafts the
    judge returned title_fit=4 and tone_fit=3 every single time, so the metric
    could not separate arms it was built to separate. Asking "which of these
    two is more like the outlet" is a much easier question for a model to
    answer consistently than "how like the outlet is this one, on a scale".

    Every pair is judged twice with the drafts swapped. A model that names the
    first draft both times is showing position bias, not a preference, so that
    case is recorded as a tie rather than a win.
    """

    VERDICTS = [("a", "A 較佳"), ("b", "B 較佳"), ("tie", "平手／不一致")]

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE,
                                   null=True, blank=True, related_name="comparisons")
    run_a = models.ForeignKey(GenerationRun, on_delete=models.CASCADE, related_name="comparisons_as_a")
    run_b = models.ForeignKey(GenerationRun, on_delete=models.CASCADE, related_name="comparisons_as_b")
    winner = models.CharField(max_length=4, choices=VERDICTS, default="tie")
    position_consistent = models.BooleanField(
        "兩種順序判斷一致", default=False,
        help_text="False 代表模型只是偏好排在前面的稿件，該結果不可採信。",
    )
    dimension_winners = models.JSONField("各維度勝方", default=dict, blank=True)
    reasoning = models.TextField("評審理由", blank=True)
    model = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "兩兩對比評審"
        ordering = ["-created_at"]

    def __str__(self):
        return f"#{self.run_a_id} vs #{self.run_b_id} → {self.get_winner_display()}"


class Evaluation(models.Model):
    """Metric-ensemble scoring for one draft.

    arXiv:2508.06374 found no single style metric is reliable and that
    ensembles beat any individual metric, so all four signals are stored
    side by side rather than collapsed into one number.
    """

    run = models.ForeignKey(GenerationRun, on_delete=models.CASCADE, related_name="evaluations")
    revision = models.ForeignKey(Revision, on_delete=models.CASCADE, null=True, blank=True,
                                 related_name="evaluations")

    style_similarity = models.FloatField("風格向量相似度", null=True, blank=True,
                                         help_text="與目標媒體語料重心的 cosine，越高越像")
    exemplar_similarity = models.FloatField("與所用範例的相似度", null=True, blank=True)
    max_overlap = models.FloatField("與範例最大字串重疊率", null=True, blank=True,
                                    help_text="抄襲防護：過高代表照抄範文")
    overlap_source = models.CharField(max_length=512, blank=True)
    fact_coverage = models.JSONField("事實覆蓋檢查", default=dict, blank=True)
    judge_scores = models.JSONField("LLM 評審分數", default=dict, blank=True)
    judge_comment = models.TextField("LLM 評審意見", blank=True)
    human_score = models.IntegerField("人工評分 1-5", null=True, blank=True)
    human_comment = models.TextField("人工意見", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "評估結果"
        ordering = ["-created_at"]

    def __str__(self):
        return f"評估 run#{self.run_id}"

    @property
    def has_overlap_score(self) -> bool:
        return self.max_overlap is not None

    @property
    def overlap_warning(self) -> bool:
        return self.max_overlap is not None and self.max_overlap > 0.15

    @property
    def missing_facts(self) -> list[str]:
        return self.fact_coverage.get("missing", []) if self.fact_coverage else []

    # `judge_scores` mixes numeric dimensions, prose fields and the statistical
    # deviation block. Django templates cannot address a key starting with an
    # underscore at all, so the split happens here rather than in the template.
    @property
    def deviation(self) -> dict:
        scores = self.judge_scores or {}
        return scores.get("deviation") or scores.get("_deviation") or {}

    @property
    def judge_dimensions(self) -> list[tuple[str, object]]:
        return [
            (k, v) for k, v in (self.judge_scores or {}).items()
            if isinstance(v, (int, float)) and not k.startswith("_")
        ]

    @property
    def judge_fixes(self) -> list:
        return (self.judge_scores or {}).get("concrete_fixes") or []

    @property
    def judge_weakest(self) -> str:
        return (self.judge_scores or {}).get("weakest_dimension") or ""
