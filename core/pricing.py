"""What a model call costs, in US dollars.

Rates are per 1M tokens and are kept here rather than in the database on
purpose: they change when a provider changes them, not when an operator edits
a form, and a wrong rate silently distorts every historical figure. Keeping
them in code means a rate change is a reviewable diff with a date on it.

Reasoning tokens are **not** added separately — every provider seen so far
bills them as output tokens and already counts them inside `output`. Adding
them again would double-count the most expensive part.

An unknown model costs 0 rather than raising. A missing rate should not stop a
draft from being written, and a zero is visible as "unknown" in the reports
(see `is_priced`), which is more useful than an exception nobody can act on.
"""
from __future__ import annotations

# model name -> (input $/1M, output $/1M), cached input priced separately.
# Verified 2026-08-12 against developers.openai.com/api/docs/pricing and each
# provider's own page.
RATES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.20),
    # DeepSeek
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-chat": (0.14, 0.28),      # alias of v4-flash, confirmed by API
    # Local models cost nothing per token — the electricity is not billed here.
    "qwen3.6:27b-q4_K_M": (0.0, 0.0),
    "qwen3-vl:8b": (0.0, 0.0),
}

# Cached input, where a provider offers it. Usually a tenth of the input rate.
CACHED_INPUT: dict[str, float] = {
    "gpt-5.6-sol": 0.50, "gpt-5.6-luna": 0.02, "gpt-5.6-terra": 0.20,
    "gpt-5.5": 0.50, "gpt-5.4": 0.25, "gpt-5.4-mini": 0.075, "gpt-5.4-nano": 0.02,
}


def is_priced(model: str) -> bool:
    return model in RATES


def cost(model: str, input_tokens: int, output_tokens: int,
         cached_input_tokens: int = 0) -> float:
    """Dollars for one call. Cached input is billed at its own lower rate.

    `input_tokens` is the full input as providers report it — the cached part
    is a subset, so it is subtracted before applying the standard rate rather
    than added on top.
    """
    rate_in, rate_out = RATES.get(model, (0.0, 0.0))
    cached = min(cached_input_tokens or 0, input_tokens or 0)
    fresh = (input_tokens or 0) - cached
    total = fresh / 1e6 * rate_in + (output_tokens or 0) / 1e6 * rate_out
    total += cached / 1e6 * CACHED_INPUT.get(model, rate_in)
    return round(total, 6)
