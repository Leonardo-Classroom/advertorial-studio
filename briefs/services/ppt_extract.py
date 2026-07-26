"""Turn an uploaded Brief / Proposal deck into structured facts.

Step 1 pulls every scrap of text out of the .pptx (shapes, grouped shapes,
tables, speaker notes) — proposal decks put the campaign's real substance in
tables and notes as often as in title placeholders.

Step 2 asks the model to normalise that into the fact schema. The result is
explicitly *editable* by the operator before generation: it becomes the only
sanctioned source of brand names, product codes and KOL names, so a mistake
here would propagate into every draft.
"""
from __future__ import annotations

from pathlib import Path

from core import llm

INSTRUCTIONS = """你是行銷簡報的資料整理員。你的工作是從媒體簡報（Media Brief / Proposal）中，
抽出後續撰寫廣編稿所需的「事實」。
鐵則：只抽取簡報中真實出現的內容，絕對不要推測、補充或發明任何資訊。
簡報沒寫的欄位就留空字串或空陣列。全程使用繁體中文。
只輸出 JSON，不要有任何其他文字或 ``` 標記。"""

TASK = """請從以下簡報文字中抽取結構化事實，輸出這個 JSON 結構：

{{
  "brand": "品牌名稱",
  "campaign": "檔期／活動名稱",
  "product": ["產品名稱或型號"],
  "slogan": "主標語（簡報中明確寫出的）",
  "key_message": ["溝通重點，逐條，用簡報原文的措辭"],
  "kol": ["合作的KOL／藝人／代言人姓名"],
  "target_audience": "目標受眾描述",
  "publish_schedule": "上線時間或檔期",
  "deliverables": ["交付項目，例如 FB貼文、官網廣編、IG POST"],
  "mandatory_terms": ["必須出現的字詞，例如品牌 hashtag、產品型號"],
  "forbidden_terms": ["明確禁止使用的字詞（沒有就空陣列）"],
  "campaign_context": "用 2-3 句話說明這個 campaign 想達成什麼，供寫手理解背景",
  "uncertain": ["你不確定或簡報中語意模糊的地方，逐條列出供人工確認"]
}}

【簡報文字】
{text}"""


def extract_text(path: str | Path) -> tuple[str, int]:
    """Return (all text, slide count) from a .pptx."""
    from pptx import Presentation

    prs = Presentation(str(path))
    chunks: list[str] = []
    slide_count = 0

    for i, slide in enumerate(prs.slides):
        slide_count += 1
        parts: list[str] = []

        def walk(shapes):
            for shape in shapes:
                if shape.shape_type == 6 and hasattr(shape, "shapes"):  # group
                    walk(shape.shapes)
                    continue
                if shape.has_text_frame:
                    t = shape.text_frame.text.strip()
                    if t:
                        parts.append(t)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        cells = [c.text.strip() for c in row.cells]
                        if any(cells):
                            parts.append(" | ".join(cells))

        walk(slide.shapes)

        if slide.has_notes_slide:
            note = slide.notes_slide.notes_text_frame.text.strip()
            if note:
                parts.append(f"（備註）{note}")

        if parts:
            chunks.append(f"--- 第 {i + 1} 張 ---\n" + "\n".join(parts))

    return "\n\n".join(chunks), slide_count


def extract_facts(raw_text: str, max_chars: int = 40000) -> dict:
    """Ask the model to normalise deck text into the fact schema."""
    if not raw_text.strip():
        return {}
    text = raw_text[:max_chars]
    return llm.complete_json(
        instructions=INSTRUCTIONS,
        user_input=TASK.format(text=text),
        timeout=240,
    )


def brief_query_text(facts: dict) -> str:
    """Condense the facts into one line for topical exemplar retrieval."""
    parts: list[str] = []
    for key in ("brand", "campaign", "slogan", "campaign_context"):
        value = facts.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for key in ("product", "key_message", "kol"):
        value = facts.get(key)
        if isinstance(value, list):
            parts.extend(str(v) for v in value[:5] if str(v).strip())
    return " ".join(parts)[:1500]
