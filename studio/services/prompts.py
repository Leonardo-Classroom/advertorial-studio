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

# Genre rules, derived from the real advertorials in 0424新增/廣編 (Girl Style,
# Popdaily). Those pieces are consumer-facing throughout: KOL outfits described
# garment by garment, colourway names, how the shoe feels to walk in, dialogue
# from the event. What they never contain is the thing the deck is mostly about
# — media placements, campaign phases, KPIs, budget.
#
# Without these rules the model wrote about the media plan: "Pre-heat 預熱期落在
# 9/25–10/7", "簡報中也提到台北捷運市政府 2 號出口", "簡報尚未提供完整資訊".
# The style guide cannot prevent that; it only governs tone. This is genre.
GENRE_RULES = """【廣編稿是什麼】
這是一篇要直接刊登給**消費者**看的圖文。讀者是想看穿搭、想知道這雙鞋好不好看的人，
不是品牌窗口，也不是廣告代理商。

**要寫的**：
- 產品長什麼樣、什麼配色、什麼材質、穿起來是什麼感覺、怎麼搭
- KOL／藝人穿了什麼（單品逐件寫）、在什麼場景、做了什麼、說了什麼
- 讀者為什麼會想要這雙鞋
- 售價、開賣日、哪裡買、活動怎麼參加——如果簡報有給的話

**絕對不能出現的**（讀者不需要知道，寫出來會立刻露餡）：
- 「campaign」「檔期」「Period」「Pre-heat」「Launch 期」「Sustain」等投放分期用語
- 版位、媒體通路、OOH／戶外廣告點位、捷運燈箱、廣告車、曝光規劃
- 預算、KPI、聲量、心佔、導購、業績、成效、觸及
- 「簡報中提到」「簡報未提供」「根據提案」——**任何指涉簡報本身的句子**
- 品牌內部目標（例如「強化商品心佔」「帶動周慶業績」）

**寫不出來的時候**：如果簡報只給了投放計畫、沒給產品細節，
就把文章聚焦在你手上真正有的產品與人物素材，把篇幅寫短一點，
**不要拿投放計畫充版面**。缺什麼資訊就寫進文末〈待確認〉，正文裡不要提。"""

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
（正文。依該媒體慣例分段並下小標。
需要放圖的位置只寫來源標註，例如 `（Photo from 品牌名）`——
**不要描述那張圖要拍什麼**，那是給攝影師的指示，不是給讀者看的。）

## Hashtag
（一行，空格分隔）

上面每個 `##` 區塊各只能出現一次。內文的小標用 `###`，
**不要在內文裡再寫一次 FB 貼文文案或標題**——那些已經在自己的區塊裡了。"""

# The same spec, for when the deck actually supplied usable pictures. The only
# difference is the 內文 rule: real images exist, so the draft places them by
# number instead of writing a placeholder for someone to fill in later.
OUTPUT_SPEC_WITH_IMAGES = """【輸出格式】請完全照下列結構輸出，用 Markdown：

## FB貼文文案
（3-5 行，口語、帶 emoji，結尾放 hashtag）

## 文章標題
（一句，符合該媒體的標題公式）

## 內文
（正文。依該媒體慣例分段並下小標。
要放圖的地方，**單獨一行**寫圖片編號標記，例如：

[[img:3]]

編號只能用下面〈可用圖片〉清單裡有的。〈可用圖片〉清單裡的每一張都要放進文章，
缺一不可；一張圖最多用一次，放在哪一段、用什麼順序由你判斷最合理的安排。
標記單獨成行，前後空一行，不要寫在句子中間。
不要再另外寫 `（Photo from ...）`，系統會自動加上圖說。）

## Hashtag
（一行，空格分隔）

## 待確認
（條列：簡報未提供、但寫稿時需要的資訊；沒有就寫「無」。
這一節不會跟著稿件刊出，是給編輯看的待辦。）

上面每個 `##` 區塊各只能出現一次。內文的小標用 `###`，
**不要在內文裡再寫一次 FB 貼文文案或標題**——那些已經在自己的區塊裡了。"""


def image_roster(images) -> str:
    """The picture menu handed to the writer.

    Only approved pictures appear. What the model sees is the human-confirmed
    description, never the raw AI guess — the review page exists precisely
    because that guess is sometimes wrong, and a draft placing a competitor's
    shoe because the classifier mislabelled it would defeat the whole gate.
    """
    lines = []
    for i, image in enumerate(images, 1):
        caption = image.display_caption() or image.ai_description
        lines.append(f"[[img:{i}]] （{image.location_label}）{caption}")
    return "\n".join(lines)


def build_image_section(images) -> str:
    if not images:
        return ""
    return ("\n\n【可用圖片（簡報裡已核准的素材，只能用這些編號；每一張都要放進文章）】\n"
            + image_roster(images))


def build_outline_image_section(images) -> str:
    """The outline's version: plan around the pictures that exist.

    Without this the outline stage answers "要不要配圖，配什麼" by inventing a
    photo brief — "這裡放一張 KOL street style 情境照" — for a shoot nobody is
    doing. Handing it the actual roster turns that line from a wish into an
    assignment — and a mandatory one: every approved picture must land
    somewhere in the outline, not just the ones that happen to fit best.
    """
    if not images:
        return ""
    return ("\n【可用圖片（簡報裡已核准的素材）——清單裡每一張都要在大綱裡安排落點，"
            "由你決定放在哪一段、順序怎麼排；不要描述一張不存在的照片】\n"
            + image_roster(images) + "\n")


def build_instructions(style_guide_text: str, outlet_name: str, author_name: str | None) -> str:
    scope = f"{outlet_name} 的 {author_name}" if author_name else outlet_name
    guide = style_guide_text.strip() or "（尚未建立風格指南，請依範例文章自行歸納語感）"
    return f"""{ROLE}

{GENRE_RULES}

這次你要模仿的是【{scope}】的寫作風格。以下是該風格的指南，請嚴格遵守：

{guide}

{HARD_RULES}"""


# Fields the writer must never see. Handing over the media plan and then asking
# the model not to write about it is a losing game; withholding it is not.
INTERNAL_FIELDS = {"internal_only", "campaign", "campaign_context",
                   "publish_schedule", "deliverables", "target_audience", "uncertain"}


def writable_facts(facts: dict) -> dict:
    """The subset of the brief a consumer-facing draft may draw on."""
    return {k: v for k, v in (facts or {}).items() if k not in INTERNAL_FIELDS}


def title_directive(facts: dict) -> str:
    """The headline, when the deck already decided it.

    Proposal decks often set an angle per outlet — that is a decision someone
    made with the client, not a gap for the model to fill. When one is present
    the model is told to use it verbatim rather than to write "in that
    direction", because a headline rewritten "in the direction of" the agreed
    one is a different headline. Empty when the deck said nothing, and then the
    model writes its own as before.
    """
    angle = str((facts or {}).get("angle") or "").strip()
    if not angle:
        return ""
    return ("\n\n【文章標題（簡報已決定，不要自己重擬）】\n"
            f"{angle}\n"
            "請把這句話**逐字**放進〈文章標題〉區塊。若它明顯是一段方向描述而不是一句標題，"
            "就依它的方向下標，並在〈待確認〉說明你為什麼沒有照用。")


def primary_kol_directive(facts: dict) -> str:
    """The lead talent, when the operator has named one.

    Never auto-extracted — a deck can name a dozen KOLs and not one of them is
    "the" one to build the article around, that call belongs to whoever knows
    the campaign. Set only through the correction flow, same path `angle`
    already uses. Empty when unset, and the writer picks its own emphasis as
    before.
    """
    kol = str((facts or {}).get("primary_kol") or "").strip()
    if not kol:
        return ""
    return ("\n\n【主打代言人（操作者指定，不要自行改變比重）】\n"
            f"以「{kol}」為敘事主角，其餘合作人選視為配角，"
            "是否提及、提及多少由篇幅與段落安排自行判斷。")


def build_input(facts: dict, exemplars: list, extra_requirements: str = "") -> str:
    facts_json = json.dumps(writable_facts(facts), ensure_ascii=False, indent=2)

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

    return f"""請依據以下素材，寫一篇廣編稿。

【可用素材（唯一事實來源；這裡沒有的一律不准寫）】
{facts_json}{title_directive(facts)}{primary_kol_directive(facts)}

{exemplar_section}
{extra}

{OUTPUT_SPEC}"""


# --- 方案 B：多階段管線 ---------------------------------------------------
#
# arXiv:2308.07968 的五階段（檢索→排序→摘要→綜合→生成）在多個資料集上顯著
# 優於單次生成。這裡把「摘要」與「綜合」分開，是因為它們解決不同的問題：
# 摘要決定「要講什麼、依什麼順序」，綜合決定「怎麼排版成一篇稿」。
# 單次生成把兩者混在一起做，正是初稿結構鬆散、賣點順序隨機的來源。

SUMMARIZE_ROLE = """你是行銷企劃，擅長把簡報裡零散的資訊整理成寫手可以直接下筆的重點筆記。
你不寫稿，只整理。全程使用繁體中文。"""

SUMMARIZE_TASK = """請把以下簡報事實整理成寫廣編稿用的重點筆記。

【簡報事實】
{facts}

請輸出 Markdown，包含：

## 核心賣點（依重要性排序，最多 5 條）
每條寫成一句話，並註明它為什麼對讀者有意義（不是對品牌有意義）。

## 敘事切角
這篇稿子要用什麼角度切入才不像官方新聞稿？提出 2 個候選，各一句話。

## 素材盤點
- 可寫的：簡報有給、可以寫進稿子的具體材料（人名、型號、場景、時間）
- 缺的：寫廣編稿通常需要、但這份簡報沒給的資訊（例如售價、通路、活動辦法）

## 語言轉譯對照
簡報裡屬於內部提案用語、不能直接寫進稿子的字詞，列出「原句 → 建議改法」。
例如：「引導消費者進店」→ 改成讀者視角的說法。"""

SYNTHESIZE_ROLE = """你是媒體主編，負責在寫手下筆前把文章骨架訂好。
你不寫正文，只出大綱。全程使用繁體中文。"""

SYNTHESIZE_TASK = """依據下方重點筆記與該媒體的風格慣例，訂出這篇廣編稿的大綱。

【重點筆記】
{notes}

【該媒體的結構與標題慣例】
{structure}

【交付項目】
{deliverables}

【必須被寫進稿子的項目——大綱要為它們安排位置，一個都不能列入「刻意不寫」】
{must_cover}
{images}{title}{primary_kol}
請輸出 Markdown 大綱：

## 標題候選
3 個，符合該媒體的標題公式。若上面已經指定了標題，這一節就只寫那一句，不要另外提案。

## 段落架構
逐段列出，每段寫：
- 小標（若該段需要小標）
- 這段要講什麼（一到兩句）
- 這段用到哪些簡報事實
- 配哪張圖（若〈可用圖片〉有清單，每張圖都要在某一段落有落點；這段沒有適合的就寫「無」）

## 結尾與 CTA
依該媒體慣例安排。

## 本篇刻意不寫的東西
列出你決定捨棄的材料與原因——大綱的價值一半在於決定不寫什麼。
但上面「必須被寫進稿子的項目」不在可捨棄之列，不得出現在這一節。

## 必寫項目與配圖落點檢查
逐一列出上面每個必寫項目，標明它被安排在第幾段。若上面有〈可用圖片〉清單，
再逐一列出清單裡的每一張圖，標明它被安排在第幾段。若有任何一項或任何一張圖
沒有落點，回到段落架構補上，不要留下未安排的項目。"""


def build_generate_from_outline(facts: dict, outline: str, exemplars: list,
                                extra_requirements: str = "", images=None) -> str:
    """Stage 5: write the draft against an approved outline."""
    facts_json = json.dumps(writable_facts(facts), ensure_ascii=False, indent=2)

    if exemplars:
        blocks = [
            f"--- 範例 {i}（{ex.article.author.name}）---\n{ex.article.exemplar_text(1200)}"
            for i, ex in enumerate(exemplars, 1)
        ]
        exemplar_section = (
            "【風格範例文章】\n（只看語氣與寫法，不要引用其中的事實，也不要照抄文字）\n\n"
            + "\n\n".join(blocks)
        )
    else:
        exemplar_section = ""

    extra = f"\n\n【額外要求】\n{extra_requirements.strip()}" if extra_requirements.strip() else ""
    images = list(images or [])

    return f"""請依照下面這份**已確認的大綱**寫出完整廣編稿。

【大綱（必須照這個骨架寫，不要自行增刪段落）】
{outline}

【可用素材（唯一事實來源）】
{facts_json}{title_directive(facts)}{primary_kol_directive(facts)}

{exemplar_section}{build_image_section(images)}
{extra}

{OUTPUT_SPEC_WITH_IMAGES if images else OUTPUT_SPEC}

【特別注意】
- 大綱已經決定了段落順序與各段任務，請照著執行，不要重新安排結構。
- 大綱中標為「刻意不寫」的內容，正文不要出現。"""


REVISION_ROLE = """你是資深廣編稿寫手，正在依據客戶／編輯的修改意見修訂稿件。
全程使用繁體中文（台灣用語）。"""


def build_revision_input(previous_text: str, feedback: str, facts: dict) -> str:
    import re as _re

    from briefs.models import preservable_fact_values, required_fact_values

    # Protect what the draft already says, plus what is required regardless.
    # Listing every KOL in the roster here would push the reviser to cram in
    # names the draft never featured.
    flat = _re.sub(r"\s+", "", previous_text).lower()
    present = [v for v in preservable_fact_values(facts)
               if _re.sub(r"\s+", "", v).lower() in flat]
    must_keep = list(dict.fromkeys(required_fact_values(facts) + present))
    keep_block = "、".join(must_keep) if must_keep else "（無）"

    return f"""以下是上一版稿件，以及編輯的修改意見。請產出修訂後的完整稿件。

【修改意見】
{feedback.strip()}

【可用素材（唯一事實來源，不得超出此範圍新增事實）】
{json.dumps(writable_facts(facts), ensure_ascii=False, indent=2)}

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
