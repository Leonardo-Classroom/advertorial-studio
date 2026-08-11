"""Decide which pictures in a deck are usable article material.

A real proposal deck is mostly not photographs. One 25-slide New Balance deck
carried 211 embedded pictures; the same brand logo appeared, byte-for-byte
unchanged, on eight or more slides. Sending all of that to a vision model would
be paying to be told "this is a logo" two hundred times.

So the work is split in two, cheap first:

  `filter_candidates` runs four local rules — exact hash, perceptual hash,
  raster-only, size floor — and costs nothing. Exact hashing catches the
  copy-pasted duplicate; perceptual hashing catches the same asset after it was
  rescaled or re-encoded, which exact hashing misses entirely and which is the
  common case for template furniture.

  `classify` then asks the model, once per surviving picture, what the image is
  and whether it belongs in a consumer-facing article.

Neither step is allowed to be the last word: everything lands in a review page
where a human approves or rejects before a picture can reach a draft. That is
the same gate the extracted facts pass through, and for the same reason.
"""
from __future__ import annotations

import hashlib
import io
import threading

from core import llm

# Hamming distance below which two perceptual hashes are treated as the same
# asset. 5/64 bits tolerates re-encoding and modest rescaling while staying
# well clear of genuinely different photographs; worth re-tuning once there is
# a corpus of decks to measure against.
PHASH_THRESHOLD = 5

# Below this a picture is an icon, a rule, or a bullet decoration.
MIN_BYTES = 20_000

CATEGORY_USABLE = "usable"
CATEGORY_LAYOUT = "layout"
CATEGORY_DECORATION = "decoration"
CATEGORY_UNKNOWN = "unknown"


def md5(blob: bytes) -> str:
    return hashlib.md5(blob).hexdigest()  # noqa: S324 - dedup key, not a credential


def dhash(blob: bytes, size: int = 8) -> str | None:
    """64-bit difference hash, hex encoded.

    Reduces the image to a tiny greyscale grid and records whether each pixel is
    brighter than the one to its right. Two encodings of the same picture agree
    on nearly every one of those comparisons; two different pictures do not.

    Implemented here rather than pulled from `imagehash` because it is ten lines
    over Pillow, which is already installed, and because a dedup rule that
    silently changes behaviour on a dependency bump is worse than one that is
    visible in the repository.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(blob)) as img:
            small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            pixels = list(small.getdata())
    except Exception:  # noqa: BLE001 - an unreadable image simply has no phash
        return None

    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(pixels[base + col] > pixels[base + col + 1])
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    """Bit distance between two hex-encoded hashes."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def is_animated(blob: bytes) -> bool:
    """True for multi-frame image formats (animated GIF/WEBP/APNG).

    A moving image has no single frame that represents it, and vision models
    are inconsistent about handling them — a real deck GIF sent to the local
    Qwen3-VL backend came back "invalid image input" outright rather than
    silently grabbing frame 0. Simplest and most honest: don't send them,
    on either backend — this is a format limitation, not a local-only one.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(blob)) as img:
            return getattr(img, "is_animated", False)
    except Exception:  # noqa: BLE001 - an unreadable image just isn't animated
        return False


def filter_candidates(images: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply the five local rules. Returns (kept, counts-per-rule-rejected).

    Kept images come back largest-first. That ordering is load-bearing: it
    decides which copy survives a near-duplicate collapse (the highest-fidelity
    one) and, when the caller has to cap how many pictures it can afford to
    classify, which ones it keeps. Byte size is a coarse but honest proxy for
    "is this a photograph" — hero shots are large, chart fragments and icons
    are not.
    """
    stats = {"duplicate": 0, "near_duplicate": 0, "vector": 0, "too_small": 0, "animated": 0}
    kept: list[dict] = []
    seen_md5: set[str] = set()
    kept_phashes: list[str] = []

    for image in sorted(images, key=lambda im: len(im["blob"]), reverse=True):
        if not llm.supports_image(image["ext"]):
            stats["vector"] += 1
            continue
        if len(image["blob"]) < MIN_BYTES:
            stats["too_small"] += 1
            continue
        if is_animated(image["blob"]):
            stats["animated"] += 1
            continue

        digest = md5(image["blob"])
        if digest in seen_md5:
            stats["duplicate"] += 1
            continue

        phash = dhash(image["blob"])
        if phash and any(hamming(phash, other) <= PHASH_THRESHOLD for other in kept_phashes):
            stats["near_duplicate"] += 1
            continue

        seen_md5.add(digest)
        if phash:
            kept_phashes.append(phash)
        kept.append({**image, "md5": digest, "phash": phash or ""})

    return kept, stats


def slide_order(images: list[dict]) -> list[dict]:
    """Re-sort into reading order, for display once selection is settled."""
    return sorted(images, key=lambda im: (im["slide_index"], im.get("top") or 0,
                                          im.get("left") or 0))


CLASSIFY_ROLE = """你是行銷素材整理員，負責看過提案簡報裡的圖片，判斷哪些可以放進要給消費者看的廣編稿。
全程使用繁體中文（台灣用語）。只輸出 JSON，不要有任何其他文字或 ``` 標記。"""

CLASSIFY_TASK = """請看這張從簡報中抽出的圖片，輸出這個 JSON：

{{
  "description": "一句話描述圖片實際畫面。這是給編輯核對用的，不是文案",
  "category": "usable / layout / decoration 三選一",
  "brand_seen": "圖中若出現可辨識的品牌（鞋身logo、包裝、店招），寫出品牌名；認不出來就留空字串",
  "caption": "一句圖說，每張都要寫，不因分類而略過。usable 就寫可直接放進文章的版本；
              layout / decoration 則客觀描述畫面內容即可，不要寫成推銷文案——
              這種圖多半不會入稿，圖說只是讓編輯在列表上認出它是什麼"
}}

這篇稿子是要為【{brand}】寫的。分類標準：

- usable：{brand} 的產品照、本次合作的人物／KOL 照、情境或活動照——讀者會想看、
  而且**確實屬於這次檔期**的畫面。
- layout：表格、圖表、時程表、媒體版位示意、聲量分析、社群或網頁截圖——這些是給代理商看的。
  **競品的產品照也一律歸這類**：提案簡報前段常放對手商品做市場分析，
  那些畫面不屬於本篇素材，放進稿子會變成幫競品打廣告。
- decoration：品牌 logo、背景底紋、圖示、色塊等裝飾元素。

判斷時以圖片本身的畫面為準。下面的線索僅供參考，它是依這張圖在原始檔案裡的位置推測的，可能對應到錯的圖：

【所在位置】{location}{heading}
【附近文字（推測，可能不準）】{nearby}"""


def classify(image: dict, brand: str = "", timeout: float | None = None) -> dict:
    """Ask the model what one picture is. Never raises — failures degrade to unknown.

    `brand` is what makes this judgement possible rather than merely descriptive.
    A competitor's shoe is still a product photo, and the first test run duly
    marked a PUMA sneaker from the market-analysis slides as usable material for
    a New Balance piece. Naming the brand turns "is this a photo" into "is this
    *our* photo", which is the question that actually matters.

    One bad picture must not abort the upload, so a failed call is recorded as
    `unknown` with the reason attached and left for the reviewer to judge.

    `timeout` defaults per backend rather than to one flat number: local goes
    to `LOCAL_LLM_VISION_TIMEOUT`, deliberately its own setting rather than the
    300s `LOCAL_LLM_TIMEOUT` the text path uses. See that setting for the
    measurements — the short version is that a stuck local call is far more
    likely than a slow one, so waiting longer buys nothing and costs the whole
    batch.

    `timeout` is enforced twice, deliberately. It is passed down to the HTTP
    client as usual, but that alone is not trustworthy against the local
    backend — a real run sat past both that timeout and an outer 180s wrapper
    with nothing raised (report/本地線上API比較.md §五), because the client's
    timeout measures gaps between bytes, not total call time, and a model that
    keeps trickling tokens (or the connection itself stalling) never triggers
    it. So the call also runs on its own daemon thread with a hard `join`
    deadline around it — an actual wall clock that does not care why the call
    is still running. Deliberately a bare `threading.Thread(daemon=True)`,
    not `ThreadPoolExecutor`: the executor's worker threads are *not* daemon
    threads, so an abandoned one is still tracked by `concurrent.futures`'s
    own atexit machinery and can hold up interpreter shutdown waiting for a
    call that may never return. A daemon thread carries no such promise —
    on the deadline the call is abandoned, not cancelled (Python cannot kill
    a blocked native call), but this function returns immediately either
    way, which is the part that matters: one stuck picture no longer holds
    up the rest of the batch, the request serving it, or the process itself.
    """
    if timeout is None:
        from core import timeouts as timeout_settings

        timeout = timeout_settings.vision()

    heading = f"（標題：{image['slide_heading']}）" if image.get("slide_heading") else ""
    nearby = image.get("nearby_text") or "（附近沒有文字）"
    # Named in the source format's own terms — a Word file has no slide 3, and
    # telling the model it does is a detail it may well try to reconcile.
    location = image.get("location") or f"第 {image['slide_index']} 張投影片"

    outcome: dict = {}

    def _call():
        try:
            outcome["data"] = llm.complete_json_vision(
                instructions=CLASSIFY_ROLE,
                user_input=CLASSIFY_TASK.format(
                    brand=brand or "本次合作品牌（簡報未指明，請以畫面中出現的主要品牌為準）",
                    location=location, heading=heading, nearby=nearby),
                image_bytes=image["blob"],
                image_ext=image["ext"],
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - reported on the calling side
            outcome["error"] = exc

    import threading

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    # A small grace beyond `timeout` itself: if the client-side timeout is
    # ever actually honoured, let it be the one to raise, so the error below
    # still reflects what really happened.
    worker.join(timeout=timeout + 10)

    # `_error` marks the two "the service did not answer" outcomes so a batch
    # can tell them apart from a picture the model genuinely judged. Only these
    # two count toward `classify_all`'s give-up rule — a blank or unparseable
    # answer is the model being unhelpful, which says nothing about whether the
    # next call will work, but a timeout or a connection error usually means
    # the server is gone and every remaining picture will pay the same wait.
    if worker.is_alive():
        return {"description": "（逾時未回應，已略過這張圖）", "category": CATEGORY_UNKNOWN,
                "brand_seen": "", "caption": "", "_error": "timeout"}
    if "error" in outcome:
        exc = outcome["error"]
        return {"description": f"（辨識失敗：{type(exc).__name__}）", "category": CATEGORY_UNKNOWN,
                "brand_seen": "", "caption": "", "_error": type(exc).__name__}
    data = outcome.get("data") or {}

    category = str(data.get("category") or "").strip().lower()
    if category not in (CATEGORY_USABLE, CATEGORY_LAYOUT, CATEGORY_DECORATION):
        category = CATEGORY_UNKNOWN
    return {
        "description": str(data.get("description") or "").strip()[:500],
        "category": category,
        "brand_seen": str(data.get("brand_seen") or "").strip()[:80],
        "caption": str(data.get("caption") or "").strip()[:200],
    }


# Two in a row, not one: a single timeout can be one awkward picture, but two
# consecutive service failures have never yet been anything but a dead server.
GIVE_UP_AFTER = 2
ABANDONED = {"description": "（辨識服務沒有回應，已中止這批的其餘圖片）",
             "category": CATEGORY_UNKNOWN, "brand_seen": "", "caption": "",
             "_error": "abandoned"}


def classify_all(images: list[dict], brand: str = "", workers: int | None = None,
                 on_done=None) -> list[dict]:
    """Classify a batch concurrently, preserving input order.

    Sequential calls would put minutes into an HTTP request — forty pictures at
    a few seconds each. The online endpoint was measured holding per-call
    latency flat at eight concurrent requests, so there the bottleneck is round
    trips, not the service, and concurrency genuinely shortens wall-clock time.

    A single local GPU is the opposite case: one 4090 running one loaded model
    can only actually compute one inference at a time, so several concurrent
    requests just queue up inside Ollama and contend for the same VRAM/context
    rather than running in parallel — plausible extra cause of the unexplained
    tail latency and blank outputs measured in report/本地線上API比較.md §五,
    on top of adding no real speedup. So `workers` defaults to 1 for the local
    backend and 6 for online, unless the caller overrides it explicitly.

    `classify` swallows its own failures (including its own timeout), so one
    bad picture still cannot take the batch down.

    It also gives up. A local Ollama that has wedged (twice now: the kernel
    D-state incident in report/本地線上API比較.md §二, and again mid-batch on
    2026-08-08, where seven pictures came back in 10~21s each and the eighth
    request simply never returned) does not recover on its own, so every
    remaining picture pays the full timeout for nothing — eleven pictures at
    120s is twenty-two minutes of a progress bar that will never finish
    usefully. After `GIVE_UP_AFTER` consecutive service-level failures the
    rest are marked abandoned without being sent. Sequential only: under
    concurrency the calls are already in flight together, so there is nothing
    left to skip.

    `on_done(n)` fires as each picture finishes, with the running count. It is
    what lets the waiting overlay say "第 3／11 張" instead of guessing from a
    clock — the request doing the work is the one being waited on, so this
    callback is the only place that actually knows. It reports *completions*,
    not position: under concurrency the pictures do not finish in order, so a
    count is the only figure that stays true.
    """
    if not images:
        return []
    if workers is None:
        workers = 1 if llm.is_local_backend("vision") else 6
    if workers <= 1:
        results = []
        consecutive = 0
        for image in images:
            if consecutive >= GIVE_UP_AFTER:
                results.append(dict(ABANDONED))
            else:
                verdict = classify(image, brand=brand)
                consecutive = consecutive + 1 if verdict.get("_error") else 0
                results.append(verdict)
            if on_done:
                on_done(len(results))
        return results

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    done = 0

    def _one(image):
        nonlocal done
        verdict = classify(image, brand=brand)
        if on_done:
            # Counting in the worker threads, so the lock is what keeps two
            # simultaneous finishes from reading the same number.
            with lock:
                done += 1
                on_done(done)
        return verdict

    with ThreadPoolExecutor(max_workers=min(workers, len(images))) as pool:
        return list(pool.map(_one, images))


def ingest(brief, run_classify: bool = True, limit: int = 40) -> dict:
    """Extract, filter, classify and store this brief's pictures.

    A brief can now span several source files, so this pulls pictures out of
    every file that finished parsing and runs the filter/classify pipeline
    over the combined list — deliberately combined, not per file, because the
    same logo showing up in both a deck and an attached Word doc should still
    collapse via `filter_candidates`'s md5/phash dedup exactly as two copies
    within one file already do.

    Returns a summary the caller can show the operator, because the interesting
    number is not how many pictures a deck holds but how few survive: the filter
    doing its job is what keeps this affordable.

    `limit` caps the vision calls per brief, applied to the size-ranked list —
    see `filter_candidates` for why the biggest are the ones worth spending on.
    A deck with hundreds of distinct photographs is unusual enough that quietly
    spending hundreds of calls on it would be the wrong default.
    """
    from django.core.files.base import ContentFile

    from briefs.models import BriefImage
    from briefs.services import source_extract

    found = []
    for source_file in brief.source_files.filter(status="done"):
        if not source_file.file:
            continue
        for image in source_extract.extract_images(source_file.file.path, source_file.format):
            found.append({
                **image,
                "_source_file": source_file,
                "location": source_file.location_label(image["slide_index"]),
            })

    kept, stats = filter_candidates(found)

    selected = slide_order(kept[:limit])
    truncated = max(0, len(kept) - limit)

    brand = str((brief.facts or {}).get("brand") or "").strip()
    blank = {"description": "", "category": CATEGORY_UNKNOWN, "brand_seen": "", "caption": ""}
    verdicts = (classify_all(selected, brand=brand) if run_classify
                else [dict(blank) for _ in selected])

    # Re-running replaces, never accumulates — but only the deck-derived rows.
    # Manually uploaded pictures (`is_manual=True`) did not come from this
    # parse and must survive a re-parse untouched.
    brief.images.filter(is_manual=False).delete()

    stored = []
    for image, verdict in zip(selected, verdicts):
        source_file = image["_source_file"]
        record = BriefImage(
            brief=brief,
            source_file=source_file,
            slide_index=image["slide_index"],
            md5=image["md5"],
            phash=image["phash"],
            slide_heading=image["slide_heading"],
            nearby_text=image["nearby_text"],
            ai_description=verdict["description"],
            ai_category=verdict["category"],
            ai_brand=verdict.get("brand_seen", ""),
            ai_caption=verdict["caption"],
            # Never pre-ticked, even for a "usable" verdict — the model's
            # category is a filter (it decides what the operator has to look
            # at), not a decision. Approval is the human's alone to give.
        )
        # `source_file.pk` keeps this collision-free now that `slide_index` is
        # only unique within one file — two files can each have a "slide 3".
        record.file.save(
            f"brief{brief.pk}_sf{source_file.pk}_s{image['slide_index']}_"
            f"{image['md5'][:8]}.{image['ext']}",
            ContentFile(image["blob"]), save=False)
        record.save()
        stored.append(record)

    return {
        "found": len(found),
        "kept": len(kept),
        "stored": len(stored),
        "truncated": truncated,
        "usable": sum(1 for r in stored if r.ai_category == CATEGORY_USABLE),
        **stats,
    }


# Progress for the waiting overlay. Kept in the cache rather than on the Brief
# row because it is worth nothing once the run ends — writing it to the
# database would mean a write per picture for a number no one reads afterwards.
# The TTL is a backstop: a run killed mid-way (server restart) leaves a stale
# entry that would otherwise make the next page load think work is in flight.
#
# Assumes writer and reader share a cache. They do under `runserver`, which is
# one process serving both the classify POST and the progress GET on separate
# threads, and the project has no CACHES setting so that is Django's
# per-process LocMemCache. Moving to a multi-worker server (gunicorn et al.)
# without also configuring a shared cache would not break the classify pass,
# but the poll would land on a worker that never wrote anything and the count
# would simply never appear.
_PROGRESS_TTL = 900


# Which briefs are classifying right now.
#
# The progress cache is keyed per brief, so it can answer "how far along is
# brief 206?" but not "what is running?" — and the 隊列 page needs the second
# question. Scanning every brief's cache key to find out would be a poll over
# rows that are almost all idle, so the passes register themselves instead.
#
# A set in memory, like `runner`'s gate: the progress cache is already
# per-process (LocMemCache), so this adds no assumption that was not there.
_active_lock = threading.Lock()
_active_classifies: set = set()


def active_classifications() -> list:
    """Brief pks with a classify pass in flight, oldest registration first."""
    with _active_lock:
        return sorted(_active_classifies)


def classify_progress_key(brief_pk) -> str:
    return f"classify_progress:{brief_pk}"


def read_classify_progress(brief_pk) -> dict:
    """`{"done": n, "total": m}` while a classify pass runs, `{}` otherwise."""
    from django.core.cache import cache

    return cache.get(classify_progress_key(brief_pk)) or {}


def _write_classify_progress(brief_pk, done: int, total: int) -> None:
    from django.core.cache import cache

    cache.set(classify_progress_key(brief_pk), {"done": done, "total": total},
              _PROGRESS_TTL)


def classify_checked(brief) -> dict:
    """Run classification on checked pictures that still have no caption.

    The checkbox no longer requires a caption to tick on — the operator can
    select pictures on sight, from the thumbnail alone, before AI has looked
    at any of them. This is the other half of that: instead of blindly
    classifying every un-judged picture in the brief, it only spends a vision
    call on the ones the operator actually chose, skipping the rest. A picture
    that already carries a caption, AI-suggested or operator-typed, is left
    untouched: this tops up what is missing, it does not re-judge what someone
    already decided.

    Manually uploaded pictures *are* included, which they did not used to be.
    The old rule reasoned that an operator-chosen picture has already passed
    human judgement so there is nothing for a classifier to add — true of the
    category, but the caption is a separate need, and the reasoning stopped
    holding once the "no caption, no tick" rule went away. A hand-uploaded KOL
    shot would otherwise sit approved and captionless with no way to ask for
    one, and reach the draft with no `<figcaption>` at all.

    What comes back is applied differently for those, though: only the
    description and caption are kept, never the category. The operator picked
    that picture on purpose, and letting the model relabel it `layout` would
    overturn a human decision with a guess — the exact inversion this review
    page exists to prevent.
    """
    targets = [image for image in brief.images.filter(approved=True)
              if not image.display_caption()]
    if not targets:
        return {"classified": 0, "usable": 0, "failed": 0}

    with _active_lock:
        _active_classifies.add(brief.pk)
    try:
        return _classify_targets(brief, targets)
    finally:
        with _active_lock:
            _active_classifies.discard(brief.pk)


def _classify_targets(brief, targets: list) -> dict:
    """The body of `classify_checked`, split out so the in-flight registration
    above has a single, exception-proof place to clear itself."""

    brand = str((brief.facts or {}).get("brand") or "").strip()
    payloads = []
    for image in targets:
        image.file.open("rb")
        try:
            blob = image.file.read()
        finally:
            image.file.close()
        payloads.append({
            "blob": blob, "ext": image.file.name.rsplit(".", 1)[-1],
            "slide_heading": image.slide_heading, "nearby_text": image.nearby_text,
            "slide_index": image.slide_index, "location": image.location_label,
        })

    # Published before the first call so the overlay can say "第 0／11 張"
    # immediately, rather than showing nothing until the first picture lands —
    # with the local backend that first gap is 30 seconds or more.
    total = len(targets)
    _write_classify_progress(brief.pk, 0, total)
    try:
        verdicts = classify_all(
            payloads, brand=brand,
            on_done=lambda n: _write_classify_progress(brief.pk, n, total))
    finally:
        # Cleared even when the batch raises: a leftover entry would leave the
        # next visitor's overlay counting against a run that is already over.
        from django.core.cache import cache

        cache.delete(classify_progress_key(brief.pk))

    usable = 0
    failed = 0
    for image, verdict in zip(targets, verdicts):
        image.ai_description = verdict["description"]
        image.ai_brand = verdict.get("brand_seen", "")
        image.ai_caption = verdict["caption"]
        fields = ["ai_description", "ai_brand", "ai_caption"]
        if not image.is_manual:
            # Deck pictures get the category; hand-uploaded ones keep whatever
            # they had — see the docstring. This is why the field list is built
            # rather than fixed.
            image.ai_category = verdict["category"]
            fields.append("ai_category")
        # Not auto-approved, even when the verdict is "usable" — see ingest().
        image.save(update_fields=fields)
        if verdict["category"] == CATEGORY_USABLE:
            usable += 1
        if verdict.get("_error"):
            failed += 1
    # `failed` separated out because "0 張判定可用" reads as a verdict on the
    # pictures when it is often a verdict on the service — the whole reason a
    # dead Ollama went unnoticed for a whole afternoon.
    return {"classified": len(targets), "usable": usable, "failed": failed}


def save_manual_uploads(brief, files) -> tuple[list, list[str]]:
    """Store operator-supplied pictures directly — no filter, no classification.

    A deck picture passes through `ingest()`'s filter-then-classify pipeline
    because a deck is mostly not usable material — logos, tables, duplicated
    assets. A picture the operator picked and uploaded on purpose (a KOL
    photo, a fresh product shot) has already passed that judgement, so it
    starts approved instead of at whatever a classifier would have guessed.

    `dhash` returning `None` (an unreadable image, per its own docstring) is
    reused here as the upload's validity check rather than adding a separate
    content-type test — a file that fails it is not treated as a picture.
    Returns (stored, rejected_filenames).
    """
    from django.core.files.base import ContentFile

    from briefs.models import BriefImage

    stored: list = []
    rejected: list[str] = []
    for f in files:
        blob = f.read()
        phash = dhash(blob)
        if phash is None:
            rejected.append(f.name)
            continue
        record = BriefImage(
            brief=brief, is_manual=True, approved=True,
            md5=md5(blob), phash=phash,
        )
        record.file.save(f.name, ContentFile(blob), save=False)
        record.save()
        stored.append(record)
    return stored, rejected
