from django.conf import settings
from django.db import models

from briefs.models import Brief
from corpus.models import Author, Outlet, StyleGuide

# Every strategy that has ever run, so historical runs keep rendering a label
# rather than a bare code. `random` and `hybrid` were control arms in the
# topical-vs-random experiment and are still reachable from run_experiment.
RETRIEVAL_STRATEGIES = [
    ("topical", "主題相似檢索（依簡報內容找同題材範文）"),
    ("typical", "主題檢索後依「文體代表性」重排"),
    ("none", "不放範例（只用風格指南）"),
    ("random", "隨機抽樣（同媒體/作者內隨機）"),
    ("hybrid", "混合（一半主題相似、一半隨機）"),
]

# What the forms offer. Narrower than the above on purpose: the other two exist
# to answer experimental questions, not to be picked while writing a draft.
SELECTABLE_STRATEGIES = RETRIEVAL_STRATEGIES[:3]

GENERATION_MODES = [
    # Staged is listed first because it is the default: it scored +0.75 on
    # advertorial_completeness and +0.50 on reads_as_human against single-shot,
    # and the operator's own read of the drafts agreed with that direction.
    ("staged", "方案 B：多階段管線（檢索→重排→摘要→大綱→生成）"),
    ("single", "方案 A：單次生成（較快，結構較鬆）"),
]

# Which provider `core.llm` uses for writing and image classification — never
# for the judge, which always stays online (see core/llm.py's module
# docstring for why that split exists and what went wrong the one time it
# didn't). The local option's cost is not hypothetical: report/本地線上API比較.md
# measured ~20% of local text generations running 5-50x longer than normal,
# and ~30% of local image classifications either failing outright or
# returning a blank description.
LLM_BACKENDS = [
    ("online", "線上 API，使用 gpt-5.4"),
    ("local", "本地模型，文字用 qwen3.6:27b-q4_K_M、圖片用 qwen3-vl:32b-fast"),
]


class SiteSettings(models.Model):
    """Generation defaults, set once by staff instead of asked on every form.

    These three used to be fields on both the portal and the studio form. None
    of them is a decision the person writing a draft can make usefully: the
    81-run comparison could not separate the retrieval strategies at all (judge
    scores spanned 0.02), and a second rewrite was accepted zero times out of
    twelve. Asking anyway spends the user's attention on nothing.

    They stay configurable rather than hard-coded because the experiments still
    vary them — `run_experiment` sweeps strategy and rewrite count — and because
    switching the default to `none` (same measured quality, ~20% faster, 14%
    cheaper) should be a settings change someone can try and undo, not a deploy.

    Models and keys deliberately do *not* live here. They belong to `.env`:
    swapping the judge model mid-corpus makes every prior score incomparable,
    which is not something a web form should make easy. `llm_backend` below
    is the one exception to "not here" — it is a pure on/off switch (which
    provider, not which model or endpoint), same shape as `EMBED_BACKEND` in
    `.env` for the embedding service. The model names/URLs it switches between
    still live in `.env` (`LLM_MODEL` / `LOCAL_LLM_MODEL` etc.), unchanged.
    """

    default_mode = models.CharField(
        "預設生成方式", max_length=8, choices=GENERATION_MODES, default="staged",
        help_text="「產出廣編稿」預設用哪個方案。方案 B 結構完整度較高但慢，"
                  "方案 A 較快，兩者的取捨見系統報告書 §七之四。")
    llm_backend = models.CharField(
        "文字／圖片生成使用的 API", max_length=8, choices=LLM_BACKENDS, default="online",
        help_text="只影響寫稿與圖片辨識，評審永遠走線上——避免本地模型評自己的稿子。")
    retrieval_strategy = models.CharField(
        "檢索策略", max_length=16, choices=RETRIEVAL_STRATEGIES, default="typical",
        help_text="決定拿哪幾篇該媒體的舊文章當語感範例。實測三種策略產出分不出差異。")
    exemplar_count = models.IntegerField("範例篇數", default=4)
    max_rewrites = models.IntegerField(
        "生成評估後重寫次數", default=1,
        help_text="0 = 只寫初稿；1 = 初稿評估後自動重寫一次（預設）。上限 3。")
    # Off by default: the confirmation screen is a stop in the middle of a small
    # edit, and every version is kept anyway, so a bad merge is survivable —
    # the earlier version is still there to generate from, or to correct again
    # from, rather than having to never have made the mistake.
    max_parallel_runs = models.IntegerField(
        "同時產稿上限", default=2,
        help_text="產稿改成背景執行後，使用者可以連按好幾次。超過這個數字的會排隊，"
                  "不會同時打出去。調高會加快多篇產出，也會同時放大 API 用量與"
                  "SQLite 的寫入競爭。")
    confirm_fact_updates = models.BooleanField(
        "更正事實前先確認差異", default=False,
        help_text="開啟後，使用者送出更正會先看到前後對照，確認才存成新版本。"
                  "關閉則直接存成新版本——舊版本都留著，產稿時可以指定用哪一版。")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = verbose_name_plural = "產稿預設值"

    def __str__(self):
        return "產稿預設值"

    def save(self, *args, **kwargs):
        self.pk = 1  # single row, always
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "SiteSettings":
        return cls.objects.get_or_create(pk=1)[0]


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
        # Distinct from "running" so the list does not claim a run is finished
        # while the auto-rewrite loop is still changing what it will hand over.
        ("refining", "自動重寫中"),
        ("done", "完成"),
        ("failed", "失敗"),
    ]

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                              null=True, blank=True, related_name="runs",
                              verbose_name="建立者")
    brief = models.ForeignKey(Brief, on_delete=models.CASCADE, related_name="runs")
    outlet = models.ForeignKey(Outlet, on_delete=models.PROTECT, related_name="runs")
    author = models.ForeignKey(Author, on_delete=models.SET_NULL, null=True, blank=True,
                               related_name="runs", verbose_name="目標作者（可留空）")
    style_guide = models.ForeignKey(StyleGuide, on_delete=models.SET_NULL,
                                    null=True, blank=True, related_name="runs")
    experiment = models.ForeignKey(Experiment, on_delete=models.SET_NULL,
                                   null=True, blank=True, related_name="runs")
    # Which version of the brief's facts this draft was written from. Without it
    # a side-by-side comparison is uninterpretable: two drafts that differ may
    # differ because of the style, or because they were told different facts.
    # Null on runs made before facts were versioned — those read `brief.facts`.
    facts_version = models.ForeignKey("briefs.BriefFacts", on_delete=models.SET_NULL,
                                      null=True, blank=True, related_name="runs",
                                      verbose_name="採用的事實版本")

    mode = models.CharField("生成方式", max_length=8, choices=GENERATION_MODES, default="staged")
    retrieval_strategy = models.CharField("檢索策略", max_length=16,
                                          choices=RETRIEVAL_STRATEGIES, default="typical")
    exemplar_count = models.IntegerField("範例篇數", default=4)
    max_rewrites = models.IntegerField(
        "生成評估後重寫次數", default=1,
        help_text="0 = 只寫初稿不重寫；1 = 初稿評估後自動重寫一次（預設）。上限 3。",
    )
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
    # Set when a worker picks the run up, so a run that dies with the server —
    # threads do not survive a restart — can be told apart from one that is
    # merely slow, and marked failed instead of spinning on screen forever.
    started_at = models.DateTimeField("開始執行時間", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "生成紀錄"
        ordering = ["-created_at"]

    def __str__(self):
        return f"#{self.pk} {self.brief.title} → {self.outlet.name}"

    @property
    def facts(self) -> dict:
        """The facts this run writes from — its own version, else the brief's.

        Every stage reads this rather than `run.brief.facts`, so a later
        correction to the brief cannot retroactively change what an existing
        draft claims to have been written from.
        """
        if self.facts_version_id:
            return self.facts_version.data or {}
        return self.brief.facts or {}

    @property
    def latest_text(self) -> str:
        """The newest *accepted* text — the last kept revision, else the draft.

        Rejected auto-rewrites are excluded on purpose: a rewrite that lost
        mandatory facts is still worth keeping as a record, but it must not
        become what the run hands downstream.
        """
        last = self.revisions.filter(accepted=True).order_by("-round").first()
        return last.output if last else self.output

    @property
    def iteration_count(self) -> int:
        """Drafts that actually count — rejected rewrites are attempts, not drafts.

        Counting rejections here overstated how much work n bought: a run whose
        only rewrite was thrown away looked like it produced two drafts.
        """
        return 1 + self.revisions.filter(source="auto", accepted=True).count()

    @property
    def rewrite_attempts(self) -> int:
        return self.revisions.filter(source="auto").count()

    @property
    def in_progress(self) -> bool:
        return self.status in ("pending", "running", "refining")

    @property
    def stage_labels(self) -> list[str]:
        """Finished stages, for a waiting page that can say where it is."""
        return [s.get("label") or s.get("name") for s in (self.stages or [])]

    @property
    def is_staged(self) -> bool:
        return self.mode == "staged"

    @property
    def stage_list(self) -> list[dict]:
        return self.stages or []


class DraftVersion(models.Model):
    """One version of the copy, as the person waiting for it counts them.

    Deliberately not the same thing as `Revision`. A revision is a record of
    machinery — every auto-rewrite attempt, including the ones thrown away — and
    the experiments count them. A draft version is what someone means when they
    say "the second draft": the first one that arrived, then whatever they asked
    for or typed afterwards. Folding the two together would either bury the
    user's own edits among rejected rewrites, or throw away the record the
    experiments are built on.

    Same shape as `BriefFacts` on purpose: append-only, each one recording what
    it was made from and the sentence that caused it. The reasons are the same
    — nothing is overwritten, and when a rewrite loses something, `user_input`
    is the only evidence of what was actually asked for.
    """

    SOURCES = [
        ("generate", "系統產出"),
        ("manual", "人工直接編輯"),
        ("revise", "依意見重寫"),
    ]

    run = models.ForeignKey(GenerationRun, on_delete=models.CASCADE, related_name="draft_versions")
    version = models.IntegerField("版本", default=1)
    text = models.TextField("稿件內容", blank=True)
    source = models.CharField("來源", max_length=16, choices=SOURCES, default="generate")
    user_input = models.TextField("使用者當時的意見", blank=True)
    parent = models.ForeignKey("self", on_delete=models.SET_NULL, null=True, blank=True,
                               related_name="children", verbose_name="以哪一版為基礎")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "稿件版本"
        ordering = ["-version"]
        unique_together = [("run", "version")]

    def __str__(self):
        return f"run#{self.run_id} 稿件 v{self.version}"

    @property
    def label(self) -> str:
        return f"v{self.version}"

    @property
    def branched(self) -> bool:
        return bool(self.parent_id) and self.parent.version != self.version - 1


class Revision(models.Model):
    """One turn of the human feedback loop.

    Mirrors the real 初稿 → Feedback → 二稿 workflow documented in 分析報告.md,
    and doubles as the preference-pair data that plan option D (lightweight
    fine-tuning) would later need — no separate annotation effort required.
    """

    # `manual` is not a rewrite at all — nobody asked the model for anything.
    # It is the user editing the delivered text directly, kept as a revision so
    # the model's own output survives underneath it: the evaluations point at
    # that text, and overwriting it would quietly change what they measured.
    SOURCES = [("human", "人工意見"), ("auto", "LLM 評審意見（自動重寫）"),
               ("manual", "人工直接編輯")]

    run = models.ForeignKey(GenerationRun, on_delete=models.CASCADE, related_name="revisions")
    round = models.IntegerField("稿次", default=2)
    feedback = models.TextField("修改意見")
    output = models.TextField("修訂後內容", blank=True)
    source = models.CharField("意見來源", max_length=8, choices=SOURCES, default="human")
    # An auto-rewrite can come out worse; when it does we keep it for the record
    # but do not treat it as the current best draft.
    accepted = models.BooleanField("採納為目前最佳稿", default=True)
    reject_reason = models.CharField("未採納原因", max_length=200, blank=True)
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
