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
# Labels without model names. They used to spell them out — "…圖片用
# qwen3-vl:32b-fast" — which duplicated `settings` and promptly drifted: the
# vision model moved to qwen3-vl:8b and this dropdown went on naming the old
# one. Names are read from `settings` at render time instead (see
# `studio.views.advanced`), so there is one source of truth.
#
# They also cannot live here even if the drift were acceptable: `choices` is
# captured in migrations, so a label built from settings would make
# `makemigrations` want a new migration every time a model name changed.
LLM_BACKENDS = [
    ("online", "線上模型"),
    ("local", "本地模型"),
]

# Where the local embedding model (BGE-M3) runs. Unlike the writer/vision
# models — which live in Ollama and are managed by its own VRAM scheduler —
# the embedder is loaded by sentence-transformers *inside this process*, so its
# ~2.3GB of VRAM is held for the process's whole life and Ollama's keep_alive
# cannot reclaim it. On a single 24GB card that 2.3GB is the difference between
# comfortable and marginal: the writer occupies 18.2GB once loaded and the
# Windows host holds ~2.5GB of the same card, leaving 3.8GB — keeping the
# embedder on the GPU too cuts that to 1.5GB, close to where Ollama starts
# spilling layers to CPU. Moving it to CPU costs +44ms per query embed
# (invisible next to a 30-120s generation) and ~11x slower *corpus* indexing,
# which is an occasional batch job.
#
# It does *not* let the writer and vision models co-reside — measured, they
# need 18.2 + 10.2 = 28.4GB and evict each other regardless of the embedder
# (see report/VRAM配置調校.md).
EMBED_DEVICES = [
    ("cuda", "GPU（快，但佔 ~2.3GB VRAM）"),
    ("cpu", "CPU（釋出 VRAM 給生成模型；查詢只慢 ~44ms）"),
]


class OnlineProvider(models.Model):
    """One configured online model endpoint, editable at /manage/advanced/.

    The online backend used to be a single set of `.env` values, which meant
    switching provider needed a deploy and there was nowhere to keep a second
    one ready. Now several can be configured and one selected
    (`SiteSettings.online_provider`); `.env` stays as the fallback for a fresh
    install that has none.

    **The key is stored in the database in clear.** That is a real change from
    keeping it in `.env`: anyone who can read the database, a backup, or a
    stolen `db.sqlite3` gets the key. The page never renders it back — only the
    last four characters — so at least it does not sit in HTML, browser cache,
    or a screenshot. Rotate keys through the provider's console, not by
    trusting this row to stay private.
    """
    # The kind is not cosmetic: it selects which API shape `core.llm` uses.
    # OpenAI and DeepSeek both serve the Responses API; Google's
    # OpenAI-compatible layer answers 404 to it and only does chat completions
    # (measured 2026-08-11). See `core.llm.USES_CHAT_COMPLETIONS`.
    # Thinking is encoded in the kind rather than kept as a separate switch:
    # it changes the model's behaviour as much as the provider does, and one
    # dropdown that says which you get beats a checkbox somewhere else that
    # silently modifies it. Measured on 32 drafts — `deepseek-v4-pro` with
    # thinking off scored worst of every combination tried (1.67) and its
    # rewrite gained nothing, while the same model with thinking on scored
    # best (3.67). That is not a detail to bury in a second control.
    KINDS = [
        ("openai", "OpenAI"),
        ("deepseek", "DeepSeek"),
        ("deepseek-thinking", "DeepSeek-thinking"),
        ("google", "Google Gemini"),
        ("kimi", "Kimi / Moonshot"),
    ]
    # Which list a row belongs to. Text and vision keep separate lists rather
    # than sharing one with two selection columns: the same endpoint rarely
    # serves both well — `deepseek-chat` writes but cannot see a picture — and
    # a shared list made "which of these is valid for pictures?" a question the
    # page could not answer. The cost is retyping a key when one provider
    # genuinely does both.
    USES = [
        ("text", "文字"),
        ("vision", "圖片"),
    ]

    use = models.CharField("用途", max_length=8, choices=USES, default="text")
    kind = models.CharField("API 種類", max_length=32, choices=KINDS, default="openai")
    base_url = models.CharField("端點", max_length=300, blank=True)
    api_key = models.CharField("API Key", max_length=300, blank=True)
    model = models.CharField("模型", max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = verbose_name_plural = "線上模型"
        ordering = ["use", "pk"]

    def __str__(self):
        return f"{self.get_kind_display()} / {self.model or '（未填模型）'}"

    @property
    def thinking(self) -> bool:
        """Whether to let the model reason before answering.

        Derived from the kind, not stored: "DeepSeek" is the non-thinking
        variant and "DeepSeek-thinking" the reasoning one. Anything else gets
        no flag at all — the switch is DeepSeek-specific and sending it
        elsewhere is a 400.
        """
        return self.kind != "deepseek"

    @property
    def key_hint(self) -> str:
        """Last four characters, for confirming *which* key is set without
        showing it. Empty when no key is stored, so the page can say so."""
        return f"…{self.api_key[-4:]}" if len(self.api_key) >= 4 else ("已設定" if self.api_key else "")

    @property
    def is_complete(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


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

    Online endpoints and their keys *do* live here now, as `OnlineProvider`
    rows selected by `online_provider` — keeping them in `.env` meant a deploy
    to switch provider and nowhere to keep a spare configured. `.env` remains
    the fallback when no row is selected. Local model names stay in `.env`
    (`LOCAL_LLM_MODEL`, `LOCAL_LLM_VISION_MODEL`) because they describe what is
    installed on this machine, not a choice someone makes in a form.

    The caution that motivated the old rule still holds: swapping the judge
    model mid-corpus makes every prior score incomparable. That is now handled
    by recording `Evaluation.judge_model` with each result rather than by
    making the change hard.
    """

    default_mode = models.CharField(
        "預設生成方式", max_length=8, choices=GENERATION_MODES, default="staged",
        help_text="「產出廣編稿」預設用哪個方案。方案 B 結構完整度較高但慢，"
                  "方案 A 較快，兩者的取捨見系統報告書 §七之四。")
    # Text and vision are configured apart because they are different models
    # even on one backend: locally they are two Ollama models that cannot both
    # fit this card, and online a provider strong at prose may not do vision at
    # all. Tying them to one switch forced the weaker of the two.
    llm_backend = models.CharField(
        "文字生成使用的模型", max_length=8, choices=LLM_BACKENDS, default="online",
        help_text="寫稿與評審跟著這個設定。選本地時寫稿與評分是同一個模型，"
                  "絕對分數意義有限，相對比較仍可用——每筆評分都記錄了當時的評分模型。")
    vision_backend = models.CharField(
        "圖片辨識使用的模型", max_length=8, choices=LLM_BACKENDS, default="online",
        help_text="只影響圖片辨識。與上面的文字模型各自獨立——本地的文字與視覺是"
                  "兩個模型，這張卡上放不下兩個，分開設定才能一邊走本地、一邊走線上。")
    # Nav visibility. Both default on, and both are only about the *link* —
    # the pages stay reachable by URL, and neither hides anything from anyone
    # who has the address. They exist because these two lists overlap with the
    # portal for whoever only ever looks at their own work; what they must not
    # be mistaken for is a permission. The switch to turn them back on lives on
    # this same page, which is never hidden, so this cannot lock anyone out.
    show_briefs_nav = models.BooleanField(
        "後台顯示「專案」", default=True,
        help_text="關掉只是收起導覽連結。後台看得到所有使用者的專案，前台只看得到自己的。")
    show_runs_nav = models.BooleanField(
        "後台顯示「生成紀錄」", default=True,
        help_text="關掉只是收起導覽連結。失敗的稿件只有這裡看得到，前台一律不顯示。")
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
    # Which configured endpoint the online backend uses. Null falls back to
    # the `.env` values, which is what a fresh install has.
    online_provider = models.ForeignKey(
        "studio.OnlineProvider", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+", verbose_name="文字使用的線上端點")
    vision_online_provider = models.ForeignKey(
        "studio.OnlineProvider", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+", verbose_name="圖片使用的線上端點")
    # Gate size for *every* local model call, not just generation — see the
    # long note in `core.llm`. `max_parallel_runs` above counts drafts in
    # flight; this counts requests at Ollama, which is also where fact
    # extraction and picture recognition land. 任務三 measured 96% of fact
    # extractions timing out for want of it. 1 matches what Ollama actually
    # runs for a 27B on this card; raising it only helps if the server can
    # genuinely serve more at once.
    local_model_concurrency = models.IntegerField(
        "本地模型同時呼叫上限", default=1,
        help_text="同時可以打給本地模型的請求數——產稿、圖片辨識、抽取內容全部算在內。"
                  "Ollama 一次只跑一個推論，排不上的請求會在它那邊等到逾時，"
                  "所以這個閘門的用處是把等待留在我們這邊（等待免費），"
                  "而不是留在 Ollama 那邊（等待會吃掉請求的逾時額度）。"
                  "只在後端為本地模型時生效；線上 API 不受限制。")
    # Three families of model timeout, previously five literals scattered
    # across as many modules (600 / 400 / 240 / 300 / 120). See `core.timeouts`
    # for what belongs to which family, and for why the staleness thresholds
    # are derived from these rather than set beside them.
    generate_timeout_seconds = models.IntegerField(
        "產稿逾時（秒）", default=600,
        help_text="寫一篇稿、重寫一篇稿，以及方案 B 每個階段的單次上限。"
                  "超過就放棄那次呼叫並標記失敗。")
    vision_timeout_seconds = models.IntegerField(
        "圖片辨識逾時（秒）", default=120,
        help_text="辨識單張圖片的上限。本地視覺模型實測單張 13–28 秒。")
    extract_timeout_seconds = models.IntegerField(
        "內容抽取逾時（秒）", default=240,
        help_text="從簡報抽出內容、依自然語言更正、合併新檔案的上限。"
                  "三檔案專案約 2 萬字，實測單次約 45 秒（模型暖機時）。")
    confirm_fact_updates = models.BooleanField(
        "更正事實前先確認差異", default=False,
        help_text="開啟後，使用者送出更正會先看到前後對照，確認才存成新版本。"
                  "關閉則直接存成新版本——舊版本都留著，產稿時可以指定用哪一版。")
    # Only takes effect when the embedding backend is `local` (EMBED_BACKEND in
    # .env). Changing it invalidates the cached SentenceTransformer so the next
    # embed reloads on the new device — see `studio.views.advanced`.
    embed_device = models.CharField(
        "本地嵌入模型執行裝置", max_length=8, choices=EMBED_DEVICES, default="cuda",
        help_text="嵌入模型（BGE-M3）跑在 GPU 還是 CPU。放 CPU 可騰出 ~2.3GB VRAM "
                  "給生成模型多一點餘裕，查詢僅慢約 44 毫秒；只有重建整個語料索引會明顯變慢。")
    # Applied as Ollama's `keep_alive` after every local generation call (via
    # `core.llm._touch_keep_alive` — the OpenAI-compatible endpoint drops the
    # field, so it takes a separate call to the native API): the model
    # unloads this many minutes after the last request. 3 is a compromise —
    # long enough to stay warm through a working session, short enough to free
    # VRAM when idle. 0 unloads immediately (every call pays the cold-start
    # penalty — up to 12x here); a large value keeps it resident. Ignored when
    # the backend is online. Only affects Ollama models, not the embedder above.
    ollama_idle_unload_minutes = models.IntegerField(
        "本地模型閒置卸載（分鐘）", default=3,
        help_text="本地模型在最後一次使用後，閒置這麼多分鐘就從 VRAM 卸載。"
                  "0 = 用完立即卸載（每次都要重新載入，會很慢）；預設 3。上限 120。")

    # Upload ceilings (任務二 §六). Read by `briefs.services.source_extract`
    # before a Brief is created; the matching `.env` values are the fallback
    # for when this row cannot be reached. Stored in MB because that is the
    # unit an operator thinks in — the service converts.
    upload_max_file_mb = models.IntegerField(
        "單檔上限（MB）", default=200,
        help_text="單一來源檔的大小上限。目前見過最大的真實簡報是 124MB，"
                  "所以預設 200 留有餘裕。檔案上傳沒有內建上限，不設就是無上限。")
    upload_max_batch_mb = models.IntegerField(
        "單次上傳合計上限（MB）", default=600,
        help_text="一次上傳的所有檔案加總。擋的是「單檔都合格、但一次丟二十個」。")
    # pptx/docx are zip containers and picture extraction unpacks their media,
    # so these two bound what one upload may expand to rather than what it
    # weighs. A real deck's media are already-compressed JPEG/PNG and barely
    # compress again — the 119MB test deck measures 1.09:1 — so the ratio has
    # a wide margin before it can touch anything legitimate.
    upload_max_unpacked_mb = models.IntegerField(
        "解開後總量上限（MB）", default=2048,
        help_text="pptx/docx 是 zip，抽圖時要解開。這是解開後的總量上限。")
    upload_max_compression_ratio = models.IntegerField(
        "壓縮比上限", default=200,
        help_text="解開後大小 ÷ 壓縮後大小。正常簡報約 1:1（圖片本來就壓過了），"
                  "壓縮炸彈動輒上千比一。")

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
    # What this run actually cost, in tokens. Recorded rather than reconstructed
    # afterwards with a tokenizer: reasoning tokens cannot be reconstructed at
    # all — they never appear in the text — and they are billed as output, so
    # a run with reasoning on can cost several times what its draft length
    # suggests. `usage_calls` counts the API calls the run took, which is how
    # a single "draft" turns out to be three or eight requests.
    input_tokens = models.IntegerField("輸入 token", default=0)
    output_tokens = models.IntegerField("輸出 token", default=0)
    reasoning_tokens = models.IntegerField("推理 token", default=0,
                                           help_text="計入輸出計費，但不出現在文字裡")
    cached_input_tokens = models.IntegerField("命中快取的輸入 token", default=0)
    usage_calls = models.IntegerField("API 呼叫次數", default=0)
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
        return f"run#{self.run_id} 稿件 p{self.version:02d}"

    @property
    def label(self) -> str:
        """`p1`, `p2`… — deliberately not `v`, which names a *facts* version.

        A draft page shows both numbers at once ("以 v3 的內容產出的第 p1 稿"),
        and while both were spelled `v` there was nothing in the label itself
        to say which was which — a run written from v03 of the facts and edited
        twice read as "v3" and "v2" side by side.

        Zero-padded so the labels are the same width down a column and sort as
        text the way they do as numbers: p9 sorts after p10, p09 does not.
        """
        return f"p{self.version:02d}"

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
    # Which model produced `judge_scores`. Recorded rather than inferred: the
    # judge follows the backend toggle now (see `core.llm`), so a score sheet
    # read months later cannot otherwise say whether an independent model or
    # the writer itself graded the text — and on local those are the same
    # model, which is exactly the case a reader must be warned about.
    judge_model = models.CharField("評分模型", max_length=64, blank=True)
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
