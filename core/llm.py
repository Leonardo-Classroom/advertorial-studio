"""Model-abstraction layer for text generation.

Everything that talks to a chat/reasoning model goes through here so the
backing model can be swapped via `.env` alone. Uses the OpenAI-compatible
*Responses* API, which is what the Azure AI Services endpoint exposes.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from django.conf import settings
from openai import OpenAI


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not settings.LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY 未設定。請在專案根目錄的 .env 填入金鑰（可參考 .env.example）。"
        )
    return OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        timeout=settings.LLM_TIMEOUT,
    )


def current_model() -> str:
    return settings.LLM_MODEL


def _extract_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()
    chunks = []
    for item in getattr(response, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(value)
    return "\n".join(chunks).strip() if chunks else str(response)


def complete(
    instructions: str,
    user_input: str,
    model: str | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """Single-shot completion.

    `instructions` is system-level guidance (style guide / role).
    `user_input`   is the task payload (brief facts, exemplars, format spec).
    """
    client = get_client()
    if timeout is not None:
        client = client.with_options(timeout=timeout)
    kwargs = {
        "model": model or settings.LLM_MODEL,
        "instructions": instructions,
        "input": user_input,
    }
    if settings.LLM_SEND_TEMPERATURE:
        kwargs["temperature"] = settings.LLM_TEMPERATURE if temperature is None else temperature
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens
    response = client.responses.create(**kwargs)
    return _extract_text(response)


def _repair_truncated_json(text: str) -> str | None:
    """Close a JSON document that was cut off mid-write.

    A larger style-guide sample makes the model write a longer analysis, and
    when that runs past the output cap the JSON simply stops — no error, just
    an unterminated document. Parsing then fails and the caller silently falls
    back to raw text, which is how a bigger sample quietly produced a *worse*
    guide. Salvaging the complete fields is far better than discarding them.
    """
    text = text.strip()
    if not text.startswith("{"):
        return None

    # Drop whatever trailed after the last complete "key": value pair.
    cut = max(text.rfind('",\n'), text.rfind('"\n'), text.rfind("],\n"),
              text.rfind("]\n"), text.rfind("},\n"), text.rfind("}\n"))
    if cut == -1:
        return None
    head = text[:cut + 1]

    # Balance the brackets that are still open, ignoring those inside strings.
    stack, in_string, escaped = [], False, False
    for ch in head:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]" and stack:
                stack.pop()
    if in_string:
        return None
    return head + "".join("}" if c == "{" else "]" for c in reversed(stack))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def complete_json(
    instructions: str,
    user_input: str,
    model: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Ask for a JSON object and parse it defensively.

    Models sometimes wrap JSON in prose or fences. On an unrecoverable parse
    failure the raw text is returned under `{"_raw": ...}` so callers can show
    something to the user instead of blowing up mid-pipeline.
    """
    raw = complete(instructions=instructions, user_input=user_input,
                   model=model, timeout=timeout)
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = None

    if data is None:
        # Fall back to the outermost {...} span.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                data = None

    if data is None:
        repaired = _repair_truncated_json(cleaned)
        if repaired:
            try:
                data = json.loads(repaired)
                if isinstance(data, dict):
                    data["_truncated"] = True
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict) or not data:
        return {"_raw": raw}
    return data
