"""Model-abstraction layer for text generation.

Everything that talks to a chat/reasoning model goes through here so the
backing model can be swapped via `.env` alone. Uses the OpenAI-compatible
*Responses* API, which is what the Azure AI Services endpoint exposes — and,
verified across several rounds of local testing (report/本地線上API比較.md),
what Ollama's OpenAI-compatible layer exposes too, text and vision both.

Two clients, not one: `SiteSettings.llm_backend` (a staff-facing toggle,
`/manage/advanced/`) picks online vs. local, and **everything follows it,
including the judge**. `model=` names a model; it does not choose a backend.

The judge used to be pinned online no matter what the toggle said, on the
grounds that a model should not grade its own output. That isolation is gone
because the online subscription expired: every evaluation in 任務四 came back
`401`, so the six 1-5 dimensions were simply never obtained and the A/B
comparison had no quality data at all. A judge that shares the writer's model
is a compromised judge; a judge that returns 401 is no judge.

What that costs, stated plainly so it is not forgotten when reading scores:
with the toggle on local, `qwen3.6:27b` grades text `qwen3.6:27b` wrote.
Absolute scores from that arrangement mean little. *Relative* comparisons —
A vs B, both judged by the same model, with the pairwise prompt also run in
both orderings — remain usable, which is what the A/B work needs. Every
`Evaluation` records `judge_model`, so no score is ever read without knowing
what produced it.

The historical warning that lived here is still worth keeping: an early
benchmark script tried to keep the judge online via `monkeypatch`, and
`complete_json()`'s internal call to `complete(...)` — a bare name, resolved
at call time through the module's current globals — silently picked up the
patched local version, so the "online judge" was quietly grading the local
model against itself *without saying so*. The lesson was never "judges must
be online"; it was that the judge's identity must be explicit and recorded.
"""
from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import NamedTuple

from django.conf import settings
from openai import OpenAI


# Writing and looking at pictures are configured separately all the way
# through: two backends, two selected endpoints. They are different models even
# within one backend — locally they are two Ollama models that cannot both fit
# this card, and an online provider good at prose may not accept an image at
# all — so a single switch could only ever serve the weaker of the two.
TEXT, VISION = "text", "vision"

# Providers known not to serve the Responses API, so the first call skips a
# round trip that would only 404. Google's compatible layer is one (measured
# 2026-08-11). This is a fast path, not the rule — `_prefers_chat` below
# discovers the rest.
USES_CHAT_COMPLETIONS = {"google", "kimi"}

# Endpoints found at runtime to reject Responses, as (base_url, model).
#
# Support is not uniform even within one provider: DeepSeek serves Responses for
# `deepseek-v4-flash` but answers `deepseek-v4-pro` with "Codex integration ...
# will be available starting early August 2026". A hardcoded list would be
# wrong the week that ships, so the decision is learned from the first refusal
# and kept only for this process — a restart tries Responses again, which is
# how the newer shape gets picked up once it works.
_chat_only: set[tuple[str, str]] = set()


def _prefers_chat(endpoint, model: str) -> bool:
    return endpoint.kind in USES_CHAT_COMPLETIONS or (endpoint.base_url, model) in _chat_only


def _responses_unsupported(exc: BaseException) -> bool:
    """Whether this failure means "this endpoint has no Responses API".

    Matched on the message rather than the status code: the two providers seen
    so far disagree on the code (Google 404, DeepSeek 400) while both say
    plainly what is wrong. Anything else — a bad key, a rate limit, a genuinely
    malformed request — must not be mistaken for it and silently retried.
    """
    text = str(exc).lower()
    if "not found for api version" in text or "is not supported for" in text:
        return True
    if "codex integration" in text and "will be available" in text:
        return True
    # Moonshot/Kimi answers 403 "The API you are accessing is not open" — a
    # permission error by status, but it means the same thing: this endpoint
    # has no Responses API. Matched narrowly enough that a genuine permission
    # problem on some other route is not mistaken for it.
    if "api you are accessing is not open" in text:
        return True
    return "responses" in text and ("not supported" in text or "unsupported" in text)


# How to ask a provider not to think first. DeepSeek's v4 models take this
# shape (verified on both flash and pro: `reasoning_tokens` goes to None);
# nothing else here is known to, and sending it to a provider that does not
# understand it is a 400, so it is gated on the kind.
THINKING_OFF = {"deepseek": {"thinking": {"type": "disabled"}}}

# Reasoning effort, as one vocabulary across providers.
#
# The two families disagree on both the shape and the granularity: DeepSeek is
# on/off through `extra_body`, OpenAI's reasoning models take
# `reasoning.effort` at four levels. Comparing them needs one vocabulary, so
# callers say `off/low/medium/high` and each provider translates.
#
# Support is per *model*, not per provider: on one OpenAI endpoint `gpt-5.4`
# accepts effort while `gpt-4o-mini` answers 400 `Unsupported parameter`. So
# there is no list of capable models here — the parameter is sent, and a model
# that rejects it is remembered and retried without (`_no_effort`), the same
# shape as the Responses fallback above.
EFFORTS = ("off", "low", "medium", "high")
_no_effort: set[tuple[str, str]] = set()


def _effort_unsupported(exc: BaseException) -> bool:
    text = str(exc).lower()
    return ("reasoning.effort" in text or "reasoning_effort" in text) and (
        "unsupported" in text or "not supported" in text)


def _reasoning_param(endpoint, model: str, effort: str | None) -> dict:
    """`reasoning=` for the Responses API, or {} when it must not be sent."""
    if not effort or endpoint.kind not in {"openai"}:
        return {}
    if (endpoint.base_url, model) in _no_effort:
        return {}
    return {"reasoning": {"effort": "none" if effort == "off" else effort}}


class OnlineEndpoint(NamedTuple):
    base_url: str
    api_key: str
    model: str
    kind: str
    # True when this came from `.env` rather than a row someone selected. The
    # judge override only applies to the `.env` endpoint — see `judge_model`.
    from_env: bool = False
    # "off" / "low" / "medium" / "high" — see EFFORTS. None means "say nothing",
    # which leaves the provider on its own default.
    effort: str | None = None

    @property
    def thinking(self) -> bool:
        return self.effort != "off"

    def thinking_extra_body(self) -> dict:
        """`extra_body` that disables reasoning, or {} when not applicable.

        DeepSeek is on/off only, so every level above "off" is simply its
        normal thinking mode.
        """
        if self.effort != "off":
            return {}
        # "deepseek" and "deepseek-thinking" are one API with two behaviours,
        # so the payload is keyed on the family rather than the kind.
        return dict(THINKING_OFF.get(self.kind.split("-")[0], {}))


# Per-thread model override, for comparing models without touching settings.
#
# Switching `SiteSettings.online_provider` to change model is process-global,
# so two models can never be in flight at once — which is what made the model
# comparison run one call at a time and take twenty minutes of pure waiting.
# This overrides only the model name, keeping the selected endpoint's URL and
# key, so it is meaningful exactly when the models live behind one endpoint
# (`deepseek-v4-flash` vs `deepseek-v4-pro`) — the case worth parallelising.
_override = threading.local()


@contextmanager
def model_override(model: str | None, effort: str | None = None, provider=None):
    """Use `model` (and optionally this effort, and this endpoint) on this thread.

    `provider` is an `OnlineProvider` row. Overriding the endpoint as well as
    the model is what lets several APIs run at once: the selected endpoint
    lives in `SiteSettings` and is process-global, so without this two
    comparisons against different providers would overwrite each other's
    setting rather than run side by side.
    """
    previous = (getattr(_override, "model", None), getattr(_override, "effort", None),
                getattr(_override, "provider", None))
    _override.model = model
    _override.effort = effort
    _override.provider = provider
    try:
        yield
    finally:
        _override.model, _override.effort, _override.provider = previous


def _online_config(kind: str = TEXT) -> OnlineEndpoint:
    """The online endpoint for `kind` (text or vision).

    From the endpoint selected at /manage/advanced/ if there is one, else the
    `.env` values — which is what a fresh install has, and what every existing
    deployment had before endpoints became editable.
    """
    forced = getattr(_override, "provider", None)
    if forced is not None:
        override_effort = getattr(_override, "effort", None)
        return OnlineEndpoint(
            forced.base_url, forced.api_key,
            getattr(_override, "model", None) or forced.model,
            forced.kind, from_env=False,
            effort=override_effort or (None if forced.thinking else "off"))
    try:
        from studio.models import SiteSettings

        row = SiteSettings.load()
        provider = row.vision_online_provider if kind == VISION else row.online_provider
        if provider is not None and provider.is_complete:
            override_effort = getattr(_override, "effort", None)
            # A provider row carries no level of its own; "DeepSeek" (as
            # opposed to "DeepSeek-thinking") means off, and anything else
            # leaves the provider on its default until a caller asks.
            default = None if provider.thinking else "off"
            return OnlineEndpoint(
                provider.base_url, provider.api_key,
                getattr(_override, "model", None) or provider.model,
                provider.kind, from_env=False,
                effort=override_effort or default)
    except Exception:  # noqa: BLE001 - a broken settings row must not hide .env
        pass
    return OnlineEndpoint(settings.LLM_BASE_URL, settings.LLM_API_KEY,
                          getattr(_override, "model", None) or settings.LLM_MODEL,
                          "openai", from_env=True,
                          effort=getattr(_override, "effort", None))


@lru_cache(maxsize=4)
def _client_for(base_url: str, api_key: str, timeout: float) -> OpenAI:
    """Cached per credential set, so editing an endpoint takes effect at once.

    Keyed on the values rather than cached as a singleton: the old
    `lru_cache(maxsize=1)` would have kept serving the previous endpoint after
    someone switched to a different one.
    """
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _online_client(kind: str = TEXT) -> OpenAI:
    base_url, api_key = _online_config(kind)[:2]
    if not api_key:
        raise RuntimeError(
            "線上模型還沒有可用的金鑰。請到「高級 → 線上模型」新增一組端點，"
            "或在 .env 填入 LLM_API_KEY。"
        )
    return _client_for(base_url, api_key, settings.LLM_TIMEOUT)


@lru_cache(maxsize=1)
def _local_client() -> OpenAI:
    return OpenAI(
        base_url=settings.LOCAL_LLM_BASE_URL,
        api_key=settings.LOCAL_LLM_API_KEY or "ollama",
        timeout=settings.LOCAL_LLM_TIMEOUT,
    )


def _backend(kind: str = TEXT) -> str:
    """"online" or "local" for `kind` — lazy import so `core` never depends on
    `studio` at module load time (only when a call actually needs to know)."""
    from studio.models import SiteSettings

    row = SiteSettings.load()
    return row.vision_backend if kind == VISION else row.llm_backend


def _ollama_keep_alive():
    """Ollama's `keep_alive` for local calls: how long the model stays in VRAM
    after a request, from the staff setting (default 3 minutes).

    Returns None (leave Ollama's own 5-minute default alone) if the setting is
    unreachable, so a missing DB row never breaks generation. Only Ollama
    models honour this; the in-process embedder is unaffected (see
    `studio.models.SiteSettings`)."""
    try:
        from studio.models import SiteSettings

        minutes = int(SiteSettings.load().ollama_idle_unload_minutes)
    except Exception:
        return None
    return f"{max(0, minutes)}m"


def _touch_keep_alive(model: str) -> None:
    """Reset `model`'s VRAM idle timer to the configured span.

    Ollama honours `keep_alive` only on its **native** `/api/*` endpoints —
    measured 2026-08-09, the OpenAI-compatible `/v1/responses` and
    `/v1/chat/completions` paths silently drop it and leave the server default
    of 5 minutes in place, so passing it in `extra_body` did nothing
    (`report/VRAM配置調校.md`). A prompt-less POST to `/api/generate` sets the
    timer alone: it returns `done_reason: "load"` in ~0.2s without generating
    or reloading anything, which is noise beside a 30-120s completion.

    Called after a successful local call, so the countdown restarts on each
    use: warm through an active session, unloaded once idle that long. Best
    effort — a failure here costs VRAM residency, never the caller's result.
    """
    keep_alive = _ollama_keep_alive()
    if keep_alive is None:
        return
    import httpx

    # settings.LOCAL_LLM_BASE_URL is the OpenAI-compatible ".../v1"; the native
    # API sits beside it at the host root.
    root = settings.LOCAL_LLM_BASE_URL.rstrip("/").removesuffix("/v1")
    try:
        httpx.post(f"{root}/api/generate", timeout=10.0,
                   json={"model": model, "keep_alive": keep_alive})
    except Exception:
        pass


def is_local_backend(kind: str = TEXT) -> bool:
    """Whether `kind`'s backend is currently local, per `SiteSettings`.

    Public wrapper around `_backend()` for callers outside this module that
    need to make a decision based on which backend is active — e.g. how many
    picture-classification calls to run at once, which depends on the *vision*
    backend and not the writer's (see `briefs/services/images.py`).
    """
    return _backend(kind) == "local"


def get_client(kind: str = TEXT) -> OpenAI:
    """The client for `kind`'s currently-configured backend.

    Each backend's client is cached separately (see `_online_client` /
    `_local_client`) precisely so flipping the toggle doesn't get stuck
    serving whichever one a single shared `lru_cache` happened to build first.
    """
    return _local_client() if _backend(kind) == "local" else _online_client(kind)


def current_model() -> str:
    if _backend() == "local":
        return settings.LOCAL_LLM_MODEL
    return _online_config()[2]


def judge_model() -> str:
    """The model used for evaluation, on whichever backend is selected.

    Online has a model held apart from the writer (`LLM_JUDGE_MODEL`); local
    does not — there is one local model that fits this card, so it both writes
    and grades. See the module docstring for what that means when reading
    scores, and `Evaluation.judge_model`, which records this value with every
    result so the answer is never inferred.
    """
    if _backend() == "local":
        return settings.LOCAL_LLM_MODEL
    endpoint = _online_config(TEXT)
    # A model name only means anything at the endpoint that serves it. When an
    # endpoint was chosen in 高級設定, its own model is the only valid answer;
    # `LLM_JUDGE_MODEL` belongs to the `.env` endpoint and applies only there.
    # Ignoring that sent "gpt-5.4" to Google and got a 404 on every evaluation.
    override = getattr(_override, "model", None)
    if override:
        return override
    if endpoint.from_env:
        return getattr(settings, "LLM_JUDGE_MODEL", None) or endpoint.model
    return endpoint.model


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


def _chat_completion(client, model: str, instructions: str, content,
                     max_output_tokens: int | None, temperature=None,
                     extra_body: dict | None = None) -> str:
    """The pre-Responses call shape, for providers that only serve that.

    Same two inputs as `complete`, mapped onto the older wire format:
    `instructions` becomes the system message and `content` the user one —
    a string for text, or OpenAI's content-parts list when a picture is
    attached. Kept apart from the Responses path rather than abstracted over
    it, because the two disagree on more than parameter names and a wrapper
    pretending otherwise would hide which one actually ran.
    """
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": instructions},
                     {"role": "user", "content": content}],
    }
    if max_output_tokens:
        kwargs["max_tokens"] = max_output_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = client.chat.completions.create(**kwargs)
    choices = getattr(response, "choices", None) or []
    text = (choices[0].message.content if choices else "") or ""
    return text.strip()


class ModelTimeout(RuntimeError):
    """A call that ran past its wall-clock deadline and was abandoned."""


class ModelBusy(RuntimeError):
    """Gave up waiting for a free slot on the local model."""


class ModelUnavailable(RuntimeError):
    """The local model server looks wedged; failing fast instead of waiting."""


# A circuit breaker over the local backend.
#
# Ollama's model runner can wedge while the server itself stays responsive:
# 2026-08-10, `/api/version` answered in 80µs throughout while every
# `/v1/responses` hung until its 600s client timeout and came back 500 with
# `Post ".../tokenize": context canceled`. So a ping is not a health check —
# the only reliable signal is that calls stop coming back.
#
# Left alone, that state costs 15 minutes per attempt: 10 minutes for the call
# to time out, then 900s of gate wait for whoever is behind it, because an
# abandoned call keeps its slot (correctly — the server is still busy with it).
# An A/B run lost two data points and an hour to exactly this.
#
# Two consecutive timeouts trip it; a success anywhere resets it. While open,
# calls fail immediately with something that names the likely fix. After the
# cooldown one call is let through to test the water — recovery needs no
# intervention, and a still-wedged server just re-opens the breaker.
_BREAKER_THRESHOLD = 2
_BREAKER_COOLDOWN = 120.0

_breaker_lock = threading.Lock()
_breaker_failures = 0
_breaker_opened_at = 0.0


def _breaker_check() -> None:
    """Raise if the backend is known-bad and the cooldown has not elapsed."""
    with _breaker_lock:
        if _breaker_failures < _BREAKER_THRESHOLD:
            return
        waited = time.monotonic() - _breaker_opened_at
        if waited < _BREAKER_COOLDOWN:
            raise ModelUnavailable(
                f"本地模型連續 {_breaker_failures} 次沒有回應，暫時停止呼叫"
                f"（{round(_BREAKER_COOLDOWN - waited)} 秒後自動再試一次）。"
                "常見原因是 Ollama 的 runner 卡住——伺服器還會回應 /api/version，"
                "但什麼都算不了；此時要重啟 Ollama，若無效則重啟 WSL。")
    # Cooldown elapsed: fall through and let this one call probe the server.


def _breaker_record(ok: bool) -> None:
    global _breaker_failures, _breaker_opened_at

    with _breaker_lock:
        if ok:
            _breaker_failures = 0
            return
        _breaker_failures += 1
        if _breaker_failures >= _BREAKER_THRESHOLD:
            _breaker_opened_at = time.monotonic()


# "The server did not answer", as opposed to "the model answered something
# unusable". Matched by class name so this module does not have to import
# httpx's and openai's error hierarchies just to classify a failure.
_TRANSPORT_ERRORS = {
    "APITimeoutError", "APIConnectionError", "InternalServerError",
    "ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError",
    "ReadError", "PoolTimeout",
}


def _is_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in _TRANSPORT_ERRORS


def backend_health() -> dict:
    """For the 隊列 page: is the local backend answering?"""
    with _breaker_lock:
        open_ = _breaker_failures >= _BREAKER_THRESHOLD
        return {
            "ok": not open_,
            "consecutive_failures": _breaker_failures,
            "retry_in": max(0, round(_BREAKER_COOLDOWN - (time.monotonic() - _breaker_opened_at)))
            if open_ else 0,
        }


# One gate for *every* local model call — writing, vision, fact extraction.
#
# 任務三 measured what its absence costs: five users uploading at once put five
# fact extractions into Ollama at the same time, Ollama runs one at a time, and
# the fifth sat in Ollama's own queue until its 240s client timeout expired.
# 96% of extractions failed that way, and the system never reached generation.
#
# Generation already had a gate (`SiteSettings.max_parallel_runs`) and behaved:
# 10 concurrent submissions held at 2, queue depth 1, and after warm-up nearly
# all succeeded. The lesson is not that generation is special — it is that the
# queue in front of Ollama has to be *ours*, where waiting is free, rather than
# Ollama's, where waiting is charged against a request timeout.
#
# So the gate belongs here, at the one place every model call passes through,
# and not one gate per kind of work: three gates of two would still be six
# requests at a server that runs one, which is the original failure with extra
# steps. `OLLAMA_NUM_PARALLEL` is unset here, and Ollama sizes itself from VRAM
# — a 27B on a 24GB card means one — so the default is 1.
#
# Online calls are not gated: a hosted endpoint has its own capacity and its
# own queue, and serialising against it would only make batches slower.
_gate = threading.Condition()
_gate_active = 0
_gate_limit_cache: tuple[float, int] = (0.0, 1)
_GATE_LIMIT_TTL = 5.0

# A hang guard, not a policy: below `ingest_runner`'s 20-minute staleness sweep
# so a wait that has gone wrong reports itself instead of being swept up as a
# mystery. Real waits are far shorter — even a five-deep queue of 85s fact
# extractions clears in about seven minutes.
GATE_MAX_WAIT = 900.0


def _concurrency() -> int:
    """How many local model calls may run at once, from 高級設定.

    Cached for a few seconds because this is read inside the gate's wait loop,
    once per wakeup: reading `SiteSettings` there would put a database query
    inside a threading lock, which is the exact mistake 任務一 發現 4 found in
    the old generation gate.
    """
    global _gate_limit_cache

    now = time.monotonic()
    expires, value = _gate_limit_cache
    if now < expires:
        return value
    try:
        from studio.models import SiteSettings

        value = max(1, int(SiteSettings.load().local_model_concurrency))
    except Exception:  # noqa: BLE001 - a broken settings row must not block work
        value = 1
    _gate_limit_cache = (now + _GATE_LIMIT_TTL, value)
    return value


def _gate_acquire() -> None:
    global _gate_active

    deadline = time.monotonic() + GATE_MAX_WAIT
    with _gate:
        while _gate_active >= _concurrency():
            if not _gate.wait(timeout=max(0.0, deadline - time.monotonic())):
                if _gate_active >= _concurrency():
                    raise ModelBusy(
                        f"等了 {round(GATE_MAX_WAIT)} 秒仍排不到本地模型，已放棄這次呼叫。"
                        "請稍後再試，或在高級設定調高「本地模型同時呼叫上限」。")
        _gate_active += 1


def _gate_release() -> None:
    global _gate_active

    with _gate:
        _gate_active -= 1
        _gate.notify()


def gate_state() -> tuple[int, int]:
    """(running, limit) — for the 隊列 page and for tests."""
    with _gate:
        return _gate_active, _concurrency()


def _with_deadline(call, seconds: float, gated: bool = False):
    """Run `call()` under a real wall clock, not the client's own timeout.

    The HTTP client's `timeout` measures the gap *between bytes*, so a model
    that keeps trickling tokens never trips it. That is not hypothetical: a
    fact extraction with `timeout=240` ran past eight minutes here, generating
    12,868 tokens and still going, because Qwen3.6's thinking mode was not
    suppressed despite `reasoning={"effort": "none"}` — the request only ended
    when Ollama was restarted by hand. Meanwhile `processing` stayed True and
    the brief was stuck.

    So the call also runs on its own daemon thread with a hard `join` deadline.
    On expiry the call is *abandoned*, not cancelled — Python cannot kill a
    blocked native call — but this returns immediately either way, which is the
    part that matters: the caller fails cleanly, its `finally` clears the flags,
    and the user can retry without restarting anything.

    Deliberately a bare `threading.Thread(daemon=True)` rather than
    `ThreadPoolExecutor`: the executor's workers are not daemon threads, so an
    abandoned one is still tracked by `concurrent.futures`'s atexit machinery
    and can hold up interpreter shutdown waiting for a call that may never
    return. This mirrors `briefs.services.images.classify`, which has guarded
    the vision path this way since before the text path needed it.

    `gated=True` puts the call through the local-model gate. Two details of
    where the gate sits are load-bearing:

    * The slot is taken **before** the deadline starts, so time spent waiting
      for a free slot is not charged against the time allowed to answer.
      Waiting inside the deadline would move the 任務三 failure rather than fix
      it — the request would still expire in a queue, just ours instead of
      Ollama's.
    * The slot is released **on the worker thread**, when the call really
      finishes — not when this function returns. An abandoned call is still
      running inside Ollama and still occupying it; releasing on abandonment
      would hand the next caller a slot the server does not actually have.
    """
    outcome: dict = {}

    if gated:
        # Before the slot, not after: a wedged server should not be queued for.
        _breaker_check()
        _gate_acquire()

    def _run():
        try:
            outcome["value"] = call()
        except Exception as exc:  # noqa: BLE001 - re-raised on the calling side
            outcome["error"] = exc
        finally:
            if gated:
                _gate_release()

    worker = threading.Thread(target=_run, daemon=True, name="llm-call")
    worker.start()
    # A small grace beyond the client's own timeout: if that one is ever
    # actually honoured, let it be the one to raise, so the error still
    # reflects what really happened.
    worker.join(timeout=seconds + 10)

    if worker.is_alive():
        if gated:
            _breaker_record(ok=False)
        raise ModelTimeout(
            f"模型超過 {round(seconds)} 秒沒有回應完畢，已放棄這次呼叫。"
            "本地模型偶爾會停不下來（見 report/本地線上API比較.md §4.2），"
            "請再試一次；連續失敗時重啟 Ollama。")
    if "error" in outcome:
        # A refusal or an unparseable answer says nothing about the server's
        # health, so only transport-level failures count toward the breaker.
        if gated and _is_transport_error(outcome["error"]):
            _breaker_record(ok=False)
        raise outcome["error"]
    if gated:
        _breaker_record(ok=True)
    return outcome["value"]

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

    `model=` overrides which model is asked; it does **not** choose a backend.
    Routing follows `SiteSettings.llm_backend` for every call, judge included
    — see the module docstring for why that isolation was dropped, and what it
    costs. The only caller that passes a model is `evaluate.py`, via
    `judge_model()`, which already returns the right name for the backend.
    """
    local = _backend() == "local"
    endpoint = None if local else _online_config(TEXT)

    client = get_client()
    if timeout is not None:
        client = client.with_options(timeout=timeout)
    elif local:
        client = client.with_options(timeout=settings.LOCAL_LLM_TIMEOUT)

    kwargs = {
        "model": model or (settings.LOCAL_LLM_MODEL if local else endpoint.model),
        "instructions": instructions,
        "input": user_input,
    }
    if local:
        # Qwen3.6 defaults to an extended-thinking mode that turns a
        # sub-second reply into 15-70x the latency (本地API效能評估.md §2.2)
        # unless told not to. The online model needs no equivalent — it
        # already runs with reasoning_tokens == 0 by default (API價錢評估.md).
        kwargs["reasoning"] = {"effort": "none"}
    elif settings.LLM_SEND_TEMPERATURE:
        kwargs["temperature"] = settings.LLM_TEMPERATURE if temperature is None else temperature
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens

    deadline = timeout if timeout is not None else (
        settings.LOCAL_LLM_TIMEOUT if local else settings.LLM_TIMEOUT)
    if not local:
        extra = endpoint.thinking_extra_body()
        if extra:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra}
        kwargs.update(_reasoning_param(endpoint, kwargs["model"], endpoint.effort))
        chat = lambda: _chat_completion(  # noqa: E731
            client, kwargs["model"], instructions, user_input, max_output_tokens,
            kwargs.get("temperature"), extra)
        if _prefers_chat(endpoint, kwargs["model"]):
            return _with_deadline(chat, deadline)
        try:
            response = _with_deadline(lambda: client.responses.create(**kwargs), deadline)
        except Exception as exc:  # noqa: BLE001 - re-raised unless a known shape
            if _effort_unsupported(exc):
                # This model does not take an effort level; remember and retry
                # without one rather than lose the call to a parameter that was
                # only ever a preference.
                _no_effort.add((endpoint.base_url, kwargs["model"]))
                kwargs.pop("reasoning", None)
                response = _with_deadline(lambda: client.responses.create(**kwargs), deadline)
                return _extract_text(response)
            if not _responses_unsupported(exc):
                raise
            _chat_only.add((endpoint.base_url, kwargs["model"]))
            return _with_deadline(chat, deadline)
        return _extract_text(response)

    response = _with_deadline(lambda: client.responses.create(**kwargs), deadline,
                              gated=local)
    _touch_keep_alive(kwargs["model"])
    return _extract_text(response)


MIME_BY_EXT = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "gif": "image/gif", "webp": "image/webp"}


def supports_image(ext: str) -> bool:
    """Whether an image format can be sent to the model at all.

    Decks are full of .wmf/.emf vector shapes — logos and PowerPoint clip art.
    The API takes raster formats only, so those are filtered out before any
    call is made rather than discovered as a 400 per picture.
    """
    return (ext or "").lower().lstrip(".") in MIME_BY_EXT


def complete_vision(
    instructions: str,
    user_input: str,
    image_bytes: bytes,
    image_ext: str,
    model: str | None = None,
    timeout: float | None = None,
) -> str:
    """Single-shot completion over one image plus text.

    Online: same endpoint, same key, same model as `complete` — the deployed
    model is multimodal, so looking at a picture needs no separate vision
    service. Local: a *different* model than `complete`'s text path
    (`LOCAL_LLM_VISION_MODEL`) — Ollama has no single model spanning both at
    a size this GPU can hold (本地API效能評估.md §一), so the writer and the
    classifier are two separate local models, unlike the online setup.

    Same routing rule as `complete`: `model=` names a model, the backend
    toggle picks the client.
    """
    import base64

    ext = (image_ext or "").lower().lstrip(".")
    mime = MIME_BY_EXT.get(ext)
    if mime is None:
        raise ValueError(f"不支援的圖片格式：{image_ext}")

    local = _backend(VISION) == "local"
    endpoint = None if local else _online_config(VISION)

    client = get_client(VISION)
    if timeout is not None:
        client = client.with_options(timeout=timeout)
    elif local:
        client = client.with_options(timeout=settings.LOCAL_LLM_TIMEOUT)

    encoded = base64.b64encode(image_bytes).decode()
    kwargs = {
        "model": model or (settings.LOCAL_LLM_VISION_MODEL if local else endpoint.model),
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_input},
                {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"},
            ],
        }],
    }
    if local:
        kwargs["reasoning"] = {"effort": "none"}

    # Same guard as `complete`. `images.classify` also wraps its call, so a
    # picture is covered twice — harmless, the inner deadline simply fires
    # first, and it means any *other* caller of this function is covered too.
    deadline = timeout if timeout is not None else (
        settings.LOCAL_LLM_VISION_TIMEOUT if local else settings.LLM_TIMEOUT)
    if not local:
        parts = [{"type": "text", "text": user_input},
                 {"type": "image_url",
                  "image_url": {"url": f"data:{mime};base64,{encoded}"}}]
        extra = endpoint.thinking_extra_body()
        if extra:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), **extra}
        chat = lambda: _chat_completion(  # noqa: E731
            client, kwargs["model"], instructions, parts, None, None, extra)
        if _prefers_chat(endpoint, kwargs["model"]):
            return _with_deadline(chat, deadline)
        try:
            response = _with_deadline(lambda: client.responses.create(**kwargs), deadline)
        except Exception as exc:  # noqa: BLE001 - see `complete`
            if not _responses_unsupported(exc):
                raise
            _chat_only.add((endpoint.base_url, kwargs["model"]))
            return _with_deadline(chat, deadline)
        return _extract_text(response)

    response = _with_deadline(lambda: client.responses.create(**kwargs), deadline,
                              gated=local)
    _touch_keep_alive(kwargs["model"])
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
    return parse_json(raw)


def complete_json_vision(
    instructions: str,
    user_input: str,
    image_bytes: bytes,
    image_ext: str,
    model: str | None = None,
    timeout: float | None = None,
) -> dict:
    """`complete_vision`, parsed with the same defences as `complete_json`."""
    raw = complete_vision(instructions=instructions, user_input=user_input,
                          image_bytes=image_bytes, image_ext=image_ext,
                          model=model, timeout=timeout)
    return parse_json(raw)


def parse_json(raw: str) -> dict:
    """Recover a JSON object from model output that may be fenced or truncated."""
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
