"""SIMULATED provider, REAL agent + REAL GroqLLM + REAL tools: the Groq SDK is replaced by a fake that reproduces what the
terminal showed (200, then 429, 404). Proves the pieces work TOGETHER and counts requests; it is not a live Groq test."""
import types

from agent import guards
from agent.agent import SalesAgent
from agent.llm import GroqLLM
from agent.state import Session
from agent.tools import TixTools
from tests.fakes import FakeClient
from tests.test_llm import HTTPErr, M1, M2, Clock, FakeSDK, resp


def build(*outcomes, models=(M1, M2)):
    sdk, clk = FakeSDK(*outcomes), Clock()
    llm = GroqLLM(models=list(models), client=sdk, sleep=clk.sleep, clock=clk.now)
    return SalesAgent(TixTools(FakeClient()), llm), sdk, clk, Session()


SEARCH = resp(None, [("search_products", '{"queries":["تيشيرت","t-shirt"],"category":"تيشرتات"}')])
FINAL = resp("لقيتلك تيشيرت أسود 👌\n[[show: 40]]")


def test_429_on_the_final_call_is_absorbed_by_the_second_model_and_cards_show():
    agent, sdk, clk, sess = build(SEARCH, HTTPErr(429, "x", {"retry-after": "20"}), FINAL)
    agent.run("وريني تيشيرتات", sess)
    assert [c["id"] for c in sess.messages[-1]["products"]] == [40] and "[[" not in sess.messages[-1]["content"]
    assert [k["model"] for k in sdk.kwargs] == [M1, M1, M2] and clk.slept == []


def test_both_models_limited_after_a_good_search_cards_still_show_and_the_next_message_costs_no_request():
    agent, sdk, clk, sess = build(SEARCH, HTTPErr(429, "x", {"retry-after": "40"}), HTTPErr(429, "x", {"retry-after": "45"}))
    agent.run("وريني تيشيرتات", sess)
    assert sess.messages[-1]["content"] == "لقيتلك دول 👇" and [c["id"] for c in sess.messages[-1]["products"]] == [40]
    n = len(sdk.kwargs)
    out = agent.run("شكرًا", sess)                       # still inside the cooldown: answered locally, Groq not touched
    assert len(sdk.kwargs) == n and guards.reply_problem(out) is None and "زحمة" in out
    assert clk.slept == []


def test_after_the_cooldown_the_conversation_simply_continues():
    agent, sdk, clk, sess = build(SEARCH, HTTPErr(429, "x", {"retry-after": "40"}), HTTPErr(429, "x", {"retry-after": "45"}),
                                  resp("العفو 😄 لو محتاج حاجة تانية قولّي."))
    agent.run("وريني تيشيرتات", sess)
    clk.t += 60
    assert agent.run("شكرًا", sess).startswith("العفو")


def test_retired_fallback_model_is_not_retried_every_turn():
    agent, sdk, _, sess = build(HTTPErr(429, "x", {"retry-after": "30"}), HTTPErr(404, "model_not_found"), resp("أهلاً 👋"), resp("تمام"))
    agent.run("أهلاً", sess)                              # M1 limited, M2 retired -> everything unavailable right now
    assert [k["model"] for k in sdk.kwargs] == [M1, M2]
    agent.run("إزيك؟", sess)                              # M1 still cooling and M2 dead: zero requests
    assert len(sdk.kwargs) == 2
