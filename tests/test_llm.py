"""The Groq layer against a fake OpenAI-SDK client that reproduces the failure modes seen in real testing
(429 with Retry-After, 404 for a retired model, empty reasoning-only replies, malformed tool calls).
A fake clock replaces time so rate-limit waits are asserted, not slept."""
import types

import pytest

import config
from agent.llm import GroqLLM, LLMError, RateLimited, clean_content, parse_duration, retry_after_of


class HTTPErr(Exception):
    def __init__(self, status, text="", headers=None):
        super().__init__(text)
        self.status_code = status
        self.response = types.SimpleNamespace(headers=headers or {})


def resp(content=None, calls=None):
    tcs = [types.SimpleNamespace(id=f"t{i}", function=types.SimpleNamespace(name=n, arguments=a)) for i, (n, a) in enumerate(calls or [])]
    msg = types.SimpleNamespace(content=content, tool_calls=tcs or None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeSDK:
    """Outcomes are consumed in order; an outcome may be a dict {model: outcome} to depend on the requested model."""

    def __init__(self, *outcomes, available=None):
        self.outcomes, self.kwargs = list(outcomes), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
        self.available = available
        self.models = types.SimpleNamespace(list=self._list)
        self.list_calls = 0

    def _list(self):
        self.list_calls += 1
        if self.available is None:
            raise HTTPErr(500)
        return types.SimpleNamespace(data=[types.SimpleNamespace(id=m) for m in self.available])

    def _create(self, **kw):
        self.kwargs.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, dict):
            o = o[kw["model"]]
        if isinstance(o, Exception):
            raise o
        return o


class Clock:
    def __init__(self):
        self.t, self.slept = 1000.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


M1, M2 = "openai/gpt-oss-120b", "openai/gpt-oss-20b"


def llm(*outcomes, models=(M1, M2), available=None):
    sdk, clk = FakeSDK(*outcomes, available=available), Clock()
    return GroqLLM(models=list(models), client=sdk, sleep=clk.sleep, clock=clk.now), sdk, clk


def used(sdk):
    return [k["model"] for k in sdk.kwargs]


# ------------------------------------------------------------------------------------------------ basics
def test_plain_answer_and_tool_call_are_parsed():
    g, *_ = llm(resp("أهلاً"))
    assert g.chat([{"role": "user", "content": "x"}], []) == {"content": "أهلاً", "tool_calls": []}
    g, *_ = llm(resp(None, [("search_products", '{"queries":["حليب"]}')]))
    assert g.chat([], [])["tool_calls"] == [{"id": "t0", "name": "search_products", "arguments": '{"queries":["حليب"]}'}]


def test_empty_reasoning_only_answer_is_retried_not_returned_blank():
    g, sdk, _ = llm(resp(""), resp("تمام"))
    assert g.chat([], [])["content"] == "تمام" and used(sdk) == [M1, M1]


def test_gpt_oss_gets_low_reasoning_effort_and_other_models_do_not():
    g, sdk, _ = llm(resp("a"), models=(M1,))
    g.chat([], [{"type": "function"}])
    assert sdk.kwargs[0]["reasoning_effort"] == config.REASONING_EFFORT and sdk.kwargs[0]["max_tokens"] >= 1000
    g, sdk, _ = llm(resp("b"), models=("qwen/qwen3-32b",))
    g.chat([], [{"type": "function"}])
    assert "reasoning_effort" not in sdk.kwargs[0]


def test_malformed_tool_call_400_is_retried_once_then_falls_to_next_model():
    bad = HTTPErr(400, "Error code: 400 - {'error': {'code': 'tool_use_failed'}}")
    g, sdk, _ = llm(bad, bad, resp("ok"))
    assert g.chat([], [])["content"] == "ok" and used(sdk) == [M1, M1, M2]


def test_bad_key_fails_fast_with_a_clear_message():
    g, sdk, _ = llm(HTTPErr(401, "invalid api key"))
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        g.chat([], [])
    assert len(sdk.kwargs) == 1


def test_think_blocks_are_stripped_and_no_key_raises_at_construction(monkeypatch):
    assert clean_content("<think>hidden</think>\nمرحبا") == "مرحبا"
    monkeypatch.setattr(config, "get_groq_key", lambda: "")
    with pytest.raises(LLMError):
        GroqLLM()


# ------------------------------------------------------------------------------------------------ 429
def test_429_is_not_retried_on_the_same_model_the_other_model_answers_immediately():
    g, sdk, clk = llm(HTTPErr(429, "rate limit", {"retry-after": "7"}), resp("ok"))
    assert g.chat([], [])["content"] == "ok"
    assert used(sdk) == [M1, M2] and clk.slept == []          # exactly two requests, no pause, no hammering


def test_a_rate_limited_model_is_skipped_on_later_turns_until_its_cooldown_ends():
    g, sdk, clk = llm(HTTPErr(429, "x", {"retry-after": "30"}), resp("a"), resp("b"), resp("c"))
    g.chat([], [])
    g.chat([], [])
    assert used(sdk) == [M1, M2, M2]                          # turn 2 did not spend a request on the limited model
    clk.t += 31
    g.chat([], [])
    assert used(sdk)[-1] == M1                                # cooldown over: preferred model again


def test_retry_after_header_and_message_hint_are_both_understood():
    assert retry_after_of(HTTPErr(429, "", {"retry-after": "12"})) == (12.0, False)
    assert retry_after_of(HTTPErr(429, "Rate limit reached ... Please try again in 7.66s. Need more tokens?")) == (7.66, False)
    assert retry_after_of(HTTPErr(429, "Please try again in 1m26.4s")) == (86.4, False)
    assert retry_after_of(HTTPErr(429, "Limit 200000, tokens per day (TPD). Please try again in 41m3s"))[1] is True
    assert retry_after_of(HTTPErr(429, "slow down")) == (None, False)
    assert parse_duration("350ms") == pytest.approx(0.35)


def test_all_models_limited_with_a_short_wait_pauses_once_with_jitter_then_succeeds():
    g, sdk, clk = llm(HTTPErr(429, "x", {"retry-after": "2"}), HTTPErr(429, "x", {"retry-after": "3"}), resp("ok"))
    assert g.chat([], [])["content"] == "ok"
    assert used(sdk) == [M1, M2, M1]  # after the pause only the model that recovered first is tried and len(clk.slept) == 1 and 2.0 < clk.slept[0] <= config.LLM_MAX_WAIT


def test_all_models_limited_with_a_long_wait_raises_ratelimited_without_sleeping_or_more_requests():
    g, sdk, clk = llm(HTTPErr(429, "x", {"retry-after": "40"}), HTTPErr(429, "x", {"retry-after": "55"}))
    with pytest.raises(RateLimited) as e:
        g.chat([], [])
    assert len(sdk.kwargs) == 2 and clk.slept == [] and 39 < e.value.retry_after < 41
    with pytest.raises(RateLimited):                          # the NEXT customer message does not hit Groq at all while both cool down
        g.chat([], [])
    assert len(sdk.kwargs) == 2


def test_a_daily_quota_gets_a_long_cooldown_not_a_pointless_short_retry():
    g, sdk, clk = llm(HTTPErr(429, "Limit 100000, tokens per day (TPD). Please try again in 7s"), resp("ok"))
    g.chat([], [])
    assert g._cool[M1] - clk.t >= config.LLM_DAILY_COOLDOWN - 1


def test_429_without_any_hint_uses_the_bounded_default():
    g, _, clk = llm(HTTPErr(429, "rate limited"), resp("ok"))
    g.chat([], [])
    assert config.LLM_DEFAULT_COOLDOWN - 1 < g._cool[M1] - clk.t < config.LLM_DEFAULT_COOLDOWN + 1


def test_worst_case_request_count_for_one_turn_is_bounded():
    g, sdk, _ = llm(*[HTTPErr(429, "x", {"retry-after": "1"})] * 2, *[HTTPErr(429, "x", {"retry-after": "100"})] * 2)
    with pytest.raises((RateLimited, LLMError)):
        g.chat([], [])
    assert len(sdk.kwargs) <= 3       # was 4 per call (2 models x 2 attempts) in v6.0, each also repeated next turn


# ------------------------------------------------------------------------------------------------ 404 / retired models
def test_retired_model_404_is_remembered_and_never_requested_again():
    g, sdk, _ = llm(HTTPErr(404, "model_not_found"), resp("ok"), resp("again"))
    assert g.chat([], [])["content"] == "ok"
    assert g.chat([], [])["content"] == "again"
    assert used(sdk) == [M1, M2, M2]


def test_configured_models_missing_from_the_models_list_are_never_called():
    g, sdk, _ = llm(resp("ok"), available=[M2, "whisper-large-v3"], models=("llama-3.3-70b-versatile", M2))
    assert g.chat([], [])["content"] == "ok" and used(sdk) == [M2] and sdk.list_calls == 1
    g.chat([], []) if sdk.outcomes else None
    assert sdk.list_calls == 1                                # checked once only


def test_models_list_failure_is_harmless():
    g, sdk, _ = llm(resp("ok"))                              # FakeSDK.list raises HTTP 500
    assert g.chat([], [])["content"] == "ok"


def test_no_configured_model_exists_gives_llmerror_and_zero_chat_requests():
    g, sdk, _ = llm(available=["some-other-model"], models=("a", "b"))
    with pytest.raises(LLMError):
        g.chat([], [])
    assert sdk.kwargs == []


def test_transient_5xx_is_retried_once_on_the_same_model():
    g, sdk, _ = llm(HTTPErr(503), resp("ok"))
    assert g.chat([], [])["content"] == "ok" and used(sdk) == [M1, M1]


def test_everything_failing_raises_llmerror_with_bounded_requests():
    g, sdk, _ = llm(*[HTTPErr(503)] * 4)
    with pytest.raises(LLMError):
        g.chat([], [])
    assert len(sdk.kwargs) == 4
