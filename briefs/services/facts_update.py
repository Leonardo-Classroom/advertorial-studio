"""Apply a user's plain-language correction to the extracted facts.

The portal asks "要更正什麼？" and gets back a sentence — "合作的是陳○○，
還有品牌名要改成 XX" — or a numbered reply to the questions the extractor left
in `uncertain`. Both are the same job: fold what the person said into the fact
dict without touching anything they did not mention.

The risk is specific and worth naming. These fields are the ones a draft must
reproduce verbatim — brand, product codes, mandatory hashtags — and a model
asked to "update" a structure is perfectly capable of tidying a product code it
was not asked about. So the prompt's main burden is not the edit, it is the
refusal to do anything else.

The version history goes in with it, as a git-style log: which version is being
edited, what each earlier one changed, and the sentence that caused it. Without
that, "改回原本的名字" was unanswerable — the model saw one snapshot and no past
— and the honest failure was to change nothing, which looks to the user like the
system ignoring them. The log carries the old values too, so a revert is a copy
from the log rather than a guess. That is also the new hazard, and the prompt
says so twice: the history is context, never a licence to restore something the
user did not ask for.
"""
from __future__ import annotations

import json

from briefs.services import facts_text
from core import llm

INSTRUCTIONS = """你負責維護一份「簡報事實」的 JSON。使用者會用自然語言告訴你哪裡要更正，
你的工作是把他說的內容套用進 JSON，然後輸出**完整的**更新後 JSON。

鐵則，違反任何一條都是嚴重錯誤：

1. **只改使用者明確提到的部分。** 其他欄位一字不動，原樣複製，
   包含標點、大小寫、空白與陣列順序。不要順手修飾、統一格式或「整理」任何東西。
2. **不要發明。** 使用者沒說、原本 JSON 也沒有的資訊，一律不要加。
   使用者說得不夠具體時，寧可不改，並把疑問留在 uncertain。
3. **保持原本的 JSON 結構與欄位名稱**，不要新增或刪除欄位。
   欄位原本是陣列就維持陣列，是字串就維持字串。
   唯一例外：若原本的 JSON 沒有 `primary_kol` 欄位，而使用者的更正是在
   指定主打代言人，你可以新增這個欄位（字串）。
4. **uncertain 是給人看的待確認清單。** 使用者若回答了其中某一項
   （常見形式是「1. …」「2. …」，數字對應清單順序），
   就把答案套用到對應欄位，並把該項從 uncertain 移除。沒被回答的項目原樣保留。
5. **internal_only 底下是投放計畫、KPI、預算等內部資訊，不會寫進稿子。**
   除非使用者明確要求，否則完全不要動它。
6. **版本紀錄只是背景知識。** 你會看到這份事實過去被改過幾次、每次改了什麼。
   那是為了讓你聽得懂「改回原本的」「上次那個改錯了」「跟 v1 一樣」這類說法。
   使用者**沒有**要求還原時，絕對不要因為看到舊值就把它改回去——
   每一次改動都是有人刻意做的。
7. **還原的界線。** 使用者要求還原某個欄位時，只有在版本紀錄裡看得到**完整**舊值，
   才把它逐字複製回去。若那一行標示「未完整列出」，或紀錄根本沒記到那個欄位，
   就**維持原樣、什麼都不要改**——寧可讓使用者發現沒有生效，
   也絕對不要憑印象補出一份看似完整、其實少了幾項的清單。
8. **使用者要求還原「整個版本」時（例如「改回第一版」「還原到 v2」），
   一律維持原樣、不要動任何欄位。** 那件事由系統的「以這一版為準」按鈕負責，
   它會逐字複製，不需要你重建。
9. 全程使用繁體中文。只輸出 JSON，不要有任何其他文字或 ``` 標記。"""

TASK = """【版本紀錄】
{history}

【目前的簡報事實 JSON（第 {version} 版，你要修改的就是這一份）】
{facts}

【使用者的更正】
{user_input}

請輸出套用更正後的完整 JSON。"""

# Long enough to carry a product list, short enough that ten versions of a
# 30-item deck do not crowd out the facts themselves.
_MAX_LINE = 120
_MAX_VERSIONS = 6


def _summarise(lines: list[str]) -> str:
    """One side of a change, trimmed — and *saying* that it was trimmed.

    Silence here is what broke the first 「改回第一版」: the log showed the first
    24 of 32 KOL names, the model copied exactly what it could see, and nothing
    in the output suggested anything was missing. A value the model cannot see
    in full is a value it must not try to rebuild, so it is labelled as such.
    """
    if not lines:
        return "（空）"
    joined = "、".join(lines)
    if len(joined) <= _MAX_LINE:
        return joined
    return f"{joined[:_MAX_LINE]}…（共 {len(lines)} 項，此處未完整列出）"


def _changes(old: dict, new: dict) -> list[str]:
    """One line per field this version changed, old value → new value."""
    return [f"{row['label']}：{_summarise(row['before'])} → {_summarise(row['after'])}"
            for row in facts_text.diff(old, new)]


def build_history(versions) -> str:
    """A git-style log of the versions, oldest first, newest last.

    `versions` arrives newest-first (the model's own ordering) and is reversed
    here, because a history reads forwards. Only the most recent handful is
    included: the useful references are "上一版" and "原本的", and a deck edited
    twenty times would otherwise spend most of the prompt on its own past.
    """
    rows = sorted(versions, key=lambda v: v.version)[-_MAX_VERSIONS:]
    if len(rows) < 2:
        return "（這是第一版，沒有更早的版本。）"

    lines: list[str] = []
    for i, version in enumerate(rows):
        head = f"v{version.version} · {version.get_source_display()}"
        if i == len(rows) - 1:
            head += "　← 你正在修改的就是這一版"
        lines.append(head)
        if version.user_input.strip():
            lines.append(f"    使用者當時說：「{version.user_input.strip()[:200]}」")
        if i:
            for change in _changes(rows[i - 1].data or {}, version.data or {}):
                lines.append(f"    {change}")
    return "\n".join(lines)


def propose(facts: dict, user_input: str, versions=None, timeout: int = 240) -> dict:
    """Return the merged facts. Raises on model or parse failure."""
    text = (user_input or "").strip()
    if not text:
        raise ValueError("沒有輸入要更正的內容。")

    rows = list(versions or [])
    merged = llm.complete_json(
        instructions=INSTRUCTIONS,
        user_input=TASK.format(
            history=build_history(rows),
            version=max((v.version for v in rows), default=1),
            facts=json.dumps(facts or {}, ensure_ascii=False, indent=2),
            user_input=text,
        ),
        timeout=timeout,
    )
    if not isinstance(merged, dict) or not merged:
        raise ValueError("模型沒有回傳可用的內容。")
    return merged


# A different job from `propose`, and deliberately a different prompt. That one
# is built to refuse: its whole burden is "change only what the user named, and
# invent nothing". Folding in a source file that failed on upload and has now
# been re-parsed needs the opposite permission — the document legitimately
# carries facts the current version has never seen.
#
# What must not change is the other half of that promise. Values already in the
# facts may have been corrected by hand, and a document is not evidence that a
# human decision was wrong. So the rule inverts per field: fill what is empty,
# add to lists, and where the document contradicts something already recorded,
# say so in `uncertain` rather than picking a side.
MERGE_INSTRUCTIONS = """你負責維護一份「簡報事實」的 JSON。

這次的情況是：同一個專案有一份來源文件先前解析失敗，現在補上了。
你的工作是把這份文件裡的資訊補進現有 JSON，然後輸出**完整的**更新後 JSON。

鐵則，違反任何一條都是嚴重錯誤：

1. **只補、不覆蓋。** 現有欄位若已經有值，一律原樣保留——包含標點、大小寫、
   空白與陣列順序。那些值可能是人工修正過的，一份文件不構成推翻人的理由。
   欄位是空字串或空陣列時，才用文件裡的內容填入。
2. **陣列是新增，不是取代。** 文件提到現有陣列沒有的項目就追加在後面，
   既有項目不動、不重排、不去重改寫。
3. **矛盾要講出來，不要自己選一邊。** 文件的說法和現有值不一致時，
   保留現有值，並在 `uncertain` 陣列加一句說明兩者的差異，讓人來判斷。
4. **不要發明。** 只寫文件裡真的有的資訊。文件沒提到的欄位一字不動。
5. **保持原本的 JSON 結構與欄位名稱**，不要新增或刪除欄位。
   欄位原本是陣列就維持陣列，是字串就維持字串。

只輸出 JSON，不要有任何其他文字或 ``` 標記。"""

MERGE_TASK = """目前的簡報事實（第 {version} 版）：

{facts}

以下是補上的來源文件《{filename}》的完整文字內容：

--- 文件開始 ---
{document}
--- 文件結束 ---

請依鐵則把這份文件的資訊補進上面的 JSON，輸出完整的更新後 JSON。"""

# The document goes in whole rather than summarised first: the extractor that
# produced these facts read whole files too, and a summarising pass would be a
# second place for a product code to get quietly rewritten.
MERGE_DOCUMENT_CHARS = 60000


def merge_document(facts: dict, document: str, filename: str = "",
                   version: int = 1, timeout: int = 300) -> dict:
    """Fold a newly-parsed source file into the facts. Raises on failure."""
    text = (document or "").strip()
    if not text:
        raise ValueError("這個檔案沒有解析出任何文字。")

    merged = llm.complete_json(
        instructions=MERGE_INSTRUCTIONS,
        user_input=MERGE_TASK.format(
            version=version,
            facts=json.dumps(facts or {}, ensure_ascii=False, indent=2),
            filename=filename or "（未命名）",
            document=text[:MERGE_DOCUMENT_CHARS],
        ),
        timeout=timeout,
    )
    if not isinstance(merged, dict) or not merged:
        raise ValueError("模型沒有回傳可用的內容。")
    return merged
