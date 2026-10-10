"""Groq chat client with tool calling. One job: return {"content", "tool_calls"} or raise LLMError.

Rate limits (HTTP 429) are the normal condition of a free/low tier, not a failure, so they are handled explicitly:
  * a 429 is never retried on the same model: that model goes on COOLDOWN (Retry-After header, else the "try again in 7.2s"
    hint in the message, else a default; a daily quota gets a long cooldown) and the next configured model, which has its
    own bucket, is tried immediately;
  * if every model is cooling and the shortest wait fits LLM_MAX_WAIT we pause ONCE (with jitter); otherwise RateLimited is
    raised straight away so the caller can answer from data it already has instead of hammering Groq;
  * cooldowns persist across customer turns, so the next message does not spend a request on a model known to be limited;
  * a 404/"model not found" marks the model dead for the life of the process (and the configured list is checked once
    against GET /models), so a retired model costs at most one request, once.
Other quirks: gpt-oss spends max_tokens on hidden reasoning (empty content), `tool_use_failed` 400s, <think> blocks."""
from __future__ import annotations

import logging
import random
import re
import threading
import time

import config

log = logging.getLogger("tix.llm")


class LLMError(Exception):
    pass


class RateLimited(LLMError):
    def __init__(self, retry_after: float, daily: bool = False):
        super().__init__(f"rate limited, retry in ~{retry_after:.0f}s")
        self.retry_after, self.daily = retry_after, daily


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_content(text: str | None) -> str:
    return _THINK.sub("", text or "").strip()


def parse_duration(text: str) -> float | None:
    """'7.2s', '1m26.4s', '350ms', '2' -> seconds."""
    m = re.fullmatch(r"\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?\s*(?:(\d+(?:\.\d+)?)ms)?\s*", text or "")
    if m and any(m.groups()):
        h, mi, sec, ms = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + sec + ms / 1000
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def retry_after_of(e: Exception) -> tuple[float | None, bool]:
    """(seconds to wait or None, is_daily_quota) from a 429's headers / message."""
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    get = (lambda k: headers.get(k) or headers.get(k.title())) if hasattr(headers, "get") else (lambda k: None)
    text = str(e)
    daily = bool(re.search(r"per day|\bTPD\b|\bRPD\b|daily", text, re.IGNORECASE))
    for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        v = get(key)
        sec = parse_duration(str(v)) if v is not None else None
        if sec is not None:
            return sec, daily
    m = re.search(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", text, re.IGNORECASE)
    if m:
        sec = parse_duration(m.group(1))
        if sec is not None:
            return sec, daily
    return None, daily


class _Retryable(Exception):
    pass


class _NextModel(Exception):
    pass


class _Dead(Exception):
    pass


class _Limited(Exception):
    def __init__(self, wait: float, daily: bool):
        super().__init__(f"429 wait {wait:.1f}s")
        self.wait, self.daily = wait, daily


class GroqLLM:
    def __init__(self, api_key: str | None = None, models: list[str] | None = None, client=None,
                 sleep=time.sleep, clock=time.monotonic):
        self.models = list(models or config.GROQ_MODELS)
        self._sleep, self._now = sleep, clock
        self._cool: dict[str, float] = {}   # model -> monotonic time it may be used again
        self._dead: set[str] = set()
        self._checked = False
        self._lock = threading.Lock()
        if client is not None:
            self._client = client
            return
        key = api_key if api_key is not None else config.get_groq_key()
        if not key:
            raise LLMError("GROQ_API_KEY is not set")
        from openai import OpenAI  # lazy: tests and the no-key UI do not need the SDK
        self._client = OpenAI(api_key=key, base_url=config.GROQ_BASE_URL, timeout=config.LLM_TIMEOUT, max_retries=0)

    # ------------------------------------------------------------------ state
    def cooling(self) -> float:
        """Seconds until some usable model is available again (0 if one is available now)."""
        alive = [m for m in self.models if m not in self._dead]
        if not alive:
            return float("inf")
        return max(0.0, min(self._cool.get(m, 0.0) for m in alive) - self._now())

    def _check_models_once(self) -> None:
        """Compare the configured models with GET /models once; models Groq does not list are dead."""
        if self._checked:
            return
        self._checked = True
        try:
            available = {m.id for m in self._client.models.list().data}
        except Exception as e:  # noqa: BLE001 - purely advisory
            log.info("could not list Groq models (%s); will rely on per-call 404 handling", type(e).__name__)
            return
        if not available:
            return
        missing = [m for m in self.models if m not in available]
        for m in missing:
            self._dead.add(m)
            log.error("configured Groq model %r is not available for this key: it will not be used", m)
        if len(missing) == len(self.models):
            log.error("NONE of the configured models exist. Set GROQ_MODEL to one of: %s", ", ".join(sorted(available)))

    # ------------------------------------------------------------------ public
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self._check_models_once()
        last: Exception | None = None
        failed: set[str] = set()  # models that failed for a non-rate-limit reason during THIS call: never re-tried in it
        waited = False
        while True:
            now = self._now()
            cand = [m for m in self.models if m not in self._dead and m not in failed]
            avail = [m for m in cand if self._cool.get(m, 0.0) <= now]
            if not avail:
                if not cand:
                    raise LLMError(f"no usable model ({last})")
                wait = min(self._cool[m] for m in cand) - now
                if waited or wait > config.LLM_MAX_WAIT:
                    raise RateLimited(max(wait, 0.0), daily=wait > config.LLM_MAX_WAIT * 10)
                waited = True
                self._sleep(wait + random.uniform(0.1, 0.6))  # bounded, jittered, once per customer turn
                continue
            if waited:  # after the single pause, spend ONE request on the model that recovered first
                avail = sorted(avail, key=lambda m: self._cool.get(m, 0.0))[:1]
            for model in avail:
                try:
                    return self._call_with_retries(model, messages, tools)
                except _Limited as e:
                    self._cool[model] = self._now() + e.wait
                    last = e
                    log.warning("Groq 429 on %s: cooling it down for %.0fs%s", model, e.wait, " (daily quota)" if e.daily else "")
                except _Dead as e:
                    self._dead.add(model)
                    last = e
                    log.error("Groq model %s is unavailable (%s): it will not be used again", model, e)
                except (_Retryable, _NextModel) as e:
                    failed.add(model)
                    last = e
                    log.warning("Groq %s failed: %s", model, e)
            if waited:  # the single post-pause attempt failed too: stop here, never loop on a limited provider
                if isinstance(last, _Limited):
                    raise RateLimited(max(last.wait, 0.0), last.daily)
                raise LLMError(f"all models failed: {last}")

    # ------------------------------------------------------------------
    def _call_with_retries(self, model, messages, tools) -> dict:
        err: Exception | None = None
        for attempt in range(config.LLM_RETRIES + 1):
            try:
                return self._call(model, messages, tools)
            except _Retryable as e:  # 5xx / network / empty / malformed tool call: worth one more try on this model
                err = e
                if attempt < config.LLM_RETRIES:
                    self._sleep(0.4 * (attempt + 1) + random.uniform(0, 0.3))
        raise _NextModel(str(err))

    def _call(self, model: str, messages: list[dict], tools: list[dict] | None) -> dict:
        kw: dict = dict(model=model, messages=messages, temperature=config.LLM_TEMPERATURE, max_tokens=config.LLM_MAX_TOKENS)
        if tools:
            kw.update(tools=tools, tool_choice="auto")
        if "gpt-oss" in model and config.REASONING_EFFORT:
            kw["reasoning_effort"] = config.REASONING_EFFORT
        try:
            resp = self._client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001 - classify by what the SDK exposes
            raise self._classify(e) from e
        try:
            m = resp.choices[0].message
        except (AttributeError, IndexError, TypeError) as e:
            raise _Retryable("empty response") from e
        calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                 for tc in getattr(m, "tool_calls", None) or []]
        content = clean_content(getattr(m, "content", None))
        if not calls and not content:
            raise _Retryable("model returned neither text nor a tool call")
        return {"content": content, "tool_calls": calls}

    @staticmethod
    def _classify(e: Exception) -> Exception:
        status = getattr(e, "status_code", None)
        text = str(e).lower()
        if status == 429:
            wait, daily = retry_after_of(e)
            if wait is None:
                wait = config.LLM_DAILY_COOLDOWN if daily else config.LLM_DEFAULT_COOLDOWN
            elif daily and wait < 60:
                wait = config.LLM_DAILY_COOLDOWN
            return _Limited(wait + 0.25, daily)
        if status == 404 or "model_not_found" in text or "decommissioned" in text or "does not exist" in text:
            return _Dead(f"HTTP {status}")
        if status == 400 and "tool_use_failed" in text:
            return _Retryable("malformed tool call")
        if status in (401, 403):
            raise LLMError(f"authentication failed ({status}): check GROQ_API_KEY")
        if status in (408, 409, 500, 502, 503, 504) or status is None:
            return _Retryable(f"transient error ({status}): {type(e).__name__}")
        if status == 400 and "reasoning_effort" in text:
            return _NextModel("reasoning_effort rejected")
        return _NextModel(f"HTTP {status}: {str(e)[:120]}")
