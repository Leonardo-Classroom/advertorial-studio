"""Prompt construction for advertorial generation.

Three sources are kept deliberately separate in the prompt, because they
authorise different things:

  facts      — the only sanctioned source of anything factual;
  style guide— how to write, never what to write;
  exemplars  — tone and rhythm reference only, explicitly *not* a fact source
               and not to be copied at the phrase level.

Blurring these is how a style-transfer system starts inventing product codes
or lifting sentences from a published article, which the plan flags as the two
highest-consequence risks.
"""
from __future__ import annotations

import json

ROLE = """你是一位資深的中文廣編稿（業配圖文）寫手，長期為時尚／潮流媒體撰稿。
你的專長是：拿到品牌的行銷簡報後，寫出讀起來像該媒體編輯自己寫的文章，而不是像廣告稿。
全程使用繁體中文（台灣用語）。"""

HARD_RULES = """【絕對規則】
1. 事實只能來自「簡報事實」區塊。品牌名、產品型號、KOL姓名、活動時間、價格等，
   簡報沒有寫的一律不准出現，也不要用「據悉」「聽說」之類的話含混帶過。
   若某個你覺得該有的資訊簡報沒提供，就不要寫那一段，並在文末的〈待確認〉列出來。
2. 範例文章「只提供語氣、句式、節奏、排版的參考」。
   嚴禁沿用範例中的具體事實、產品、人名，也嚴禁整句或半句照抄範例的文字。
   同樣意思請換句話說。
3. 不要寫出「本文為廣告」以外的自我指涉（例如「作為AI」「根據簡報」）。
4. 不要使用簡體字。"""

OUTPUT_SPEC = """【輸出格式】請完全照下列結構輸出，用 Markdown：

## FB貼文文案
（3-5 行，口語、帶 emoji，結尾放 hashtag）

## 文章標題
（一句，符合該媒體的標題公式）

## 內文
（正文。依該媒體慣例分段並下小標；需要放圖的位置用 `（圖說：…）` 標示。）

## Hashtag
（一行，空格分隔）

## 待確認
（條列：簡報未提供、但寫稿時需要的資訊；沒有就寫「無」）"""


def build_instructions(style_guide_text: str, outlet_name: str, author_name: str | None) -> str:
    scope = f"{outlet_name} 的 {author_name}" if author_name else outlet_name
    guide = style_guide_text.strip() or "（尚未建立風格指南，請依範例文章自行歸納語感）"
    return f"""{ROLE}

這次你要模仿的是【{scope}】的寫作風格。以下是該風格的指南，請嚴格遵守：

{guide}

{HARD_RULES}"""


def build_input(facts: dict, exemplars: list, extra_requirements: str = "") -> str:
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)

    if exemplars:
        blocks = []
        for i, ex in enumerate(exemplars, 1):
            blocks.append(
                f"--- 範例 {i}（{ex.article.author.name}，{ex.article.published_on}）---\n"
                f"{ex.article.exemplar_text()}"
            )
        exemplar_section = (
            "【風格範例文章】\n"
            "（再次提醒：只看語氣與寫法，不要引用其中的任何事實，也不要照抄文字）\n\n"
            + "\n\n".join(blocks)
        )
    else:
        exemplar_section = "【風格範例文章】\n（本次不提供範例，請純粹依風格指南寫作）"

    extra = f"\n\n【額外要求】\n{extra_requirements.strip()}" if extra_requirements.strip() else ""

    return f"""請依據以下簡報事實，寫一篇廣編稿。

【簡報事實（唯一事實來源）】
{facts_json}

{exemplar_section}
{extra}

{OUTPUT_SPEC}"""


REVISION_ROLE = """你是資深廣編稿寫手，正在依據客戶／編輯的修改意見修訂稿件。
全程使用繁體中文（台灣用語）。"""


def build_revision_input(previous_text: str, feedback: str, facts: dict) -> str:
    from briefs.models import mandatory_fact_values

    must_keep = mandatory_fact_values(facts)
    keep_block = "、".join(must_keep) if must_keep else "（無）"

    return f"""以下是上一版稿件，以及編輯的修改意見。請產出修訂後的完整稿件。

【修改意見】
{feedback.strip()}

【簡報事實（唯一事實來源，不得超出此範圍新增事實）】
{json.dumps(facts, ensure_ascii=False, indent=2)}

【上一版稿件】
{previous_text}

【要求】
- 只針對修改意見調整，其餘保持原樣，不要順手改寫沒被點名的段落。
- 維持原本的格式結構（FB貼文文案／文章標題／內文／Hashtag／待確認）。
- **以下項目必須全部保留在稿件中，一個都不能刪**：
  {keep_block}
  即使修改意見要求「刪除不確定資訊」「精簡內容」，也只能調整它們的寫法或措辭，
  不可以整個拿掉。若某項確實不宜寫死，改用較保守的講法（例如「合作陣容規劃中，
  包含 XXX」），但名稱本身仍要出現。
- 輸出完整的修訂稿，不要只寫改動的部分，也不要附上說明。"""
