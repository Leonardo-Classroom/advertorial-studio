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


def filter_candidates(images: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply the four local rules. Returns (kept, counts-per-rule-rejected).

    Kept images come back largest-first. That ordering is load-bearing: it
    decides which copy survives a near-duplicate collapse (the highest-fidelity
    one) and, when the caller has to cap how many pictures it can afford to
    classify, which ones it keeps. Byte size is a coarse but honest proxy for
    "is this a photograph" — hero shots are large, chart fragments and icons
    are not.
    """
    stats = {"duplicate": 0, "near_duplicate": 0, "vector": 0, "too_small": 0}
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
  "caption": "若 category 是 usable，寫一句可放在文章裡的圖說；否則留空字串"
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


def classify(image: dict, brand: str = "", timeout: float = 120) -> dict:
    """Ask the model what one picture is. Never raises — failures degrade to unknown.

    `brand` is what makes this judgement possible rather than merely descriptive.
    A competitor's shoe is still a product photo, and the first test run duly
    marked a PUMA sneaker from the market-analysis slides as usable material for
    a New Balance piece. Naming the brand turns "is this a photo" into "is this
    *our* photo", which is the question that actually matters.

    One bad picture must not abort the upload, so a failed call is recorded as
    `unknown` with the reason attached and left for the reviewer to judge.
    """
    heading = f"（標題：{image['slide_heading']}）" if image.get("slide_heading") else ""
    nearby = image.get("nearby_text") or "（附近沒有文字）"
    # Named in the source format's own terms — a Word file has no slide 3, and
    # telling the model it does is a detail it may well try to reconcile.
    location = image.get("location") or f"第 {image['slide_index']} 張投影片"

    try:
        data = llm.complete_json_vision(
            instructions=CLASSIFY_ROLE,
            user_input=CLASSIFY_TASK.format(
                brand=brand or "本次合作品牌（簡報未指明，請以畫面中出現的主要品牌為準）",
                location=location, heading=heading, nearby=nearby),
            image_bytes=image["blob"],
            image_ext=image["ext"],
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - one picture must not fail the upload
        return {"description": f"（辨識失敗：{type(exc).__name__}）", "category": CATEGORY_UNKNOWN,
                "brand_seen": "", "caption": ""}

    category = str(data.get("category") or "").strip().lower()
    if category not in (CATEGORY_USABLE, CATEGORY_LAYOUT, CATEGORY_DECORATION):
        category = CATEGORY_UNKNOWN
    return {
        "description": str(data.get("description") or "").strip()[:500],
        "category": category,
        "brand_seen": str(data.get("brand_seen") or "").strip()[:80],
        "caption": str(data.get("caption") or "").strip()[:200],
    }


def classify_all(images: list[dict], brand: str = "", workers: int = 6) -> list[dict]:
    """Classify a batch concurrently, preserving input order.

    Sequential calls would put minutes into an HTTP request — forty pictures at
    a few seconds each. The generation endpoint was measured holding per-call
    latency flat at eight concurrent requests, so the bottleneck is round trips,
    not the service. `classify` swallows its own failures, so one bad picture
    still cannot take the batch down.
    """
    if not images:
        return []
    if workers <= 1:
        return [classify(image, brand=brand) for image in images]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(images))) as pool:
        return list(pool.map(lambda image: classify(image, brand=brand), images))


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

    brief.images.all().delete()  # re-running replaces, never accumulates

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
            # Pre-set to the model's verdict so review is a check, not data
            # entry — but nothing is used until the operator saves the page.
            approved=verdict["category"] == CATEGORY_USABLE,
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
