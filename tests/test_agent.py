"""End-to-end conversations through the real agent loop, real tools and a fake TIX API; the model is scripted
(see tests/fakes.py): these verify orchestration, state, guards, call counts and failure handling, not language understanding."""
import json

import pytest

import config
from agent import guards
from agent.agent import SalesAgent, UnavailableLLM
from agent.llm import LLMError, RateLimited
from agent.state import Session
from agent.tools import TixTools
from tests.fakes import CATALOG, FakeClient, ScriptedLLM, call, prod, say, show, tool_results, PHONES, ELEC


def make(*replies, client=None, fail=False, llm=None):
    client = client or FakeClient()
    llm = llm or ScriptedLLM(*replies, fail=fail)
    return SalesAgent(TixTools(client), llm), llm, client, Session()


def ids(sess):
    return [c["id"] for c in sess.messages[-1].get("products", [])]


def reply(sess):
    return sess.messages[-1]["content"]


# ===================================================================== conversation
def test_small_talk_costs_one_llm_call_and_no_api_traffic():
    agent, llm, client, sess = make(say("تمام الحمد لله 😄 وإنت عامل إيه؟ محتاج حاجة؟"))
    agent.run("عامل ايه؟", sess)
    assert llm.calls == 1 and client.calls == [] and "products" not in sess.messages[-1]


def test_show_me_xiaomi_is_two_llm_calls_search_then_reply_with_cards():
    agent, llm, _, sess = make(call("search_products", queries=["xiaomi", "شاومي"], category="موبايلات"),
                               show("دي موبايلات شاومي المتاحة 👌", 24))
    agent.run("وريني الشاومي", sess)
    assert llm.calls == 2 and ids(sess) == [24] and "[[" not in reply(sess) and "show" not in reply(sess)
    assert llm.tools_offered[0] and llm.tools_offered[1]  # tools stay available, the model simply did not need more


def test_general_question_is_answered_in_words_from_real_data_without_cards():
    agent, llm, _, sess = make(call("search_products", queries=["موبايلات", "phones"], category="موبايلات"),
                               say("عندنا آيفون وسامسونج وشاومي، من 6000 لـ 35000 جنيه 👌 تحب أوريهملك؟"))
    agent.run("إيه الموبايلات اللي عندكو؟", sess)
    res = tool_results(llm.requests[1])[0]
    assert res["brands"] == {"Apple": 3, "Samsung": 2, "Xiaomi": 1} and res["price_range"] == [6000, 35000]
    assert "products" not in sess.messages[-1] and sess.shown == [] and llm.calls == 2   # allowed prices came from the tool, no correction


def test_show_request_then_cards_for_the_same_topic():
    agent, llm, _, sess = make(call("search_products", queries=["موبايلات"], category="موبايلات", limit=5),
                               show("دي الموبايلات المتاحة 👇", 24, 23, 21))
    agent.run("وريني الموبايلات", sess)
    assert ids(sess) == [24, 23, 21]


def test_brand_that_does_not_exist_is_reported_honestly_and_nothing_is_invented():
    client = FakeClient([c for c in CATALOG if c["brand"] is None or c["brand"]["name"] != "Samsung"])
    agent, llm, _, sess = make(call("search_products", queries=["samsung", "سامسونج"], category="موبايلات", brand="samsung"),
                               say("مش لاقي موبايلات Samsung عندنا حاليًا 🙏 عندنا Apple و Xiaomi لو تحب أشوفهملك."), client=client)
    agent.run("طب سامسونج؟", sess)
    res = tool_results(llm.requests[1])[0]
    assert res["total"] == 0 and set(res["in_category"]["brands"]) == {"Apple", "Xiaomi"} and res["in_category"]["products"] == 4
    assert "products" not in sess.messages[-1]


def test_unavailable_product_type_has_no_cards():
    agent, llm, _, sess = make(call("search_products", queries=["غسالة", "washing machine"]),
                               say("للأسف مفيش غسالات عندنا دلوقتي 🙏 تحب أشوفلك حاجة تانية؟"))
    agent.run("عندكم غسالات؟", sess)
    assert tool_results(llm.requests[1])[0]["total"] == 0 and "products" not in sess.messages[-1]


def test_recommendation_then_yes_shows_the_products_it_just_named_without_searching_again():
    agent, llm, _, sess = make(call("search_products", queries=["موبايلات"], category="موبايلات", sort="price_high", limit=6),
                               say("عندي اختيارين كويسين: Samsung Galaxy S24 بـ 28000 جنيه للأداء، و Xiaomi Redmi 13 بـ 6000 جنيه لو عايز حاجة أوفر. تحب أوريهملك؟"),
                               show("اتفضل 👇", 25, 24))
    agent.run("بص أنا عايز موبايل كويس", sess)
    assert "products" not in sess.messages[-1] and llm.calls == 2
    agent.run("أيوه", sess)
    assert ids(sess) == [25, 24] and llm.calls == 3 and len(tool_results(llm.requests[-1])) == 0


def test_phone_conversation_keeps_context_and_cheaper_uses_what_is_on_screen():
    agent, llm, _, sess = make(
        say("تمام 👌 ميزانيتك كام تقريبًا؟"),
        say("حلو، تحب نوع معين زي آيفون ولا سامسونج؟"),
        call("search_products", queries=["iphone", "آيفون"], max_price=15000, category="موبايلات"),
        show("دول أقرب اختيارين آيفون في ميزانيتك 👌", 22, 21),
        call("compare_products", product_ids=[22, 21]),
        show("آيفون 12 أحدث وعليه خصم، فهو الأنسب.", 22),
        call("search_products", queries=["iphone"], category="موبايلات", max_price=14899, sort="price_low"),
        show("أرخص واحد ده 👌", 21))
    for text in ["عايز موبايل", "15000", "ويكون آيفون", "وريني الأحسن"]:
        agent.run(text, sess)
    assert ids(sess) == [22]
    texts = [m["content"] for m in llm.requests[2] if m["role"] in ("user", "assistant")]
    assert texts[:3] == ["عايز موبايل", "تمام 👌 ميزانيتك كام تقريبًا؟", "15000"]
    agent.run("طب في أرخص؟", sess)
    system = llm.requests[-2][0]["content"]
    assert "id=22" in system and "14900 EGP" in system and "max_price=15000" in system
    assert ids(sess) == [21]


def test_the_second_one_is_resolved_from_the_products_on_screen():
    agent, llm, _, sess = make(call("search_products", queries=["iphone"], category="موبايلات", sort="price_low"),
                               show("دول الآيفونات", 21, 22, 20),
                               call("get_product_details", product_id=22),
                               say("آيفون 12 كويس، وعليه خصم 10%."))
    agent.run("وريني آيفون", sess)
    agent.run("التاني ده كويس؟", sess)
    system = llm.requests[2][0]["content"]
    assert "1. id=21" in system and "2. id=22" in system and "3. id=20" in system
    assert tool_results(llm.requests[3])[0]["name"].startswith("iPhone 12")


def test_changing_category_midway_then_a_refinement_inside_the_new_category():
    agent, llm, _, sess = make(
        call("search_products", queries=["موبايلات"], category="موبايلات"), show("موبايلات 👇", 24),
        call("search_products", queries=["تيشيرت", "t-shirt"], category="تيشيرتات"), show("تيشيرتات 👇", 40),
        call("search_products", queries=["اسود", "black"], category="تيشيرتات"), show("لقيت أسود 👌", 40))
    agent.run("وريني موبايلات", sess)
    agent.run("خلاص سيبك من الموبايلات وريني تيشيرتات", sess)
    assert ids(sess) == [40] and sess.last_search["category"] == "تيشيرتات"
    agent.run("فيه أسود؟", sess)
    assert "category=تيشيرتات" in llm.requests[-1][0]["content"] and ids(sess) == [40]
    assert [c["id"] for c in sess.messages[1]["products"]] == [24]  # the earlier phone cards stay in the transcript


def test_vague_gift_request_asks_first_then_searches_with_the_answer():
    agent, llm, client, sess = make(
        say("أكيد 👌 عنده كام سنة وميزانيتك حوالي كام؟"),
        call("list_categories"), call("search_products", category="ساعات", max_price=1000), show("ساعة كلاسيك تنفع هدية لشاب 👌", 30))
    agent.run("عايز هدية لأخويا", sess)
    assert client.calls == []
    agent.run("20 سنة و1000 جنيه", sess)
    assert ids(sess) == [30]


def test_video_question_answers_from_details_and_the_card_carries_the_video_link():
    agent, _, _, sess = make(call("search_products", queries=["iphone 15"]), call("get_product_details", product_id=20),
                             show("أيوه، المنتج ده عليه فيديو وتقدر تفتحه من الكارت 🎥", 20))
    agent.run("آيفون 15 فيه فيديو؟", sess)
    assert sess.messages[-1]["products"][0]["video"] == "https://cdn.tix/i15.mp4"


def test_english_and_mixed_language_flow_through_unchanged():
    agent, _, _, sess = make(call("search_products", queries=["samsung", "سامسونج"], brand="samsung"),
                             show("Two Samsung options 👌 the A15 is the budget one.", 23, 25))
    assert agent.run("عايز Samsung phone", sess).startswith("Two Samsung") and ids(sess) == [23, 25]


# ===================================================================== ambiguous / corrected Arabic
def test_a_correction_keeps_the_whole_history_so_the_model_can_replace_the_old_request():
    agent, llm, _, sess = make(say("تقصد ألبان للشرب ولا بنطلون؟ 😄"),
                               call("search_products", queries=["لبن", "حليب", "milk"], category="حليب"),
                               show("آه 😂 فهمتك، تقصد لبن للشرب. دول المتاح:", 10, 12, 11))
    agent.run("طيب في البان ايه", sess)
    agent.run("لا اقصد عايز لبن", sess)
    texts = [m["content"] for m in llm.requests[1] if m["role"] in ("user", "assistant")]
    assert texts == ["طيب في البان ايه", "تقصد ألبان للشرب ولا بنطلون؟ 😄", "لا اقصد عايز لبن"]
    assert ids(sess) == [10, 12, 11]


# ===================================================================== guards
def test_model_cannot_show_a_product_no_tool_returned_and_the_marker_never_leaks():
    agent, llm, _, sess = make(show("دي اقتراحاتي 👇", 20, 999))
    out = agent.run("وريني حاجة", sess)
    assert sess.messages[-1].get("products") is None and "[[" not in out and "999" not in out


def test_only_ids_from_tools_become_cards_even_if_the_model_mixes_real_and_fake_ids():
    agent, _, _, sess = make(call("search_products", queries=["حليب"], category="حليب"), show("دول 👇", 10, 20, 999, 12))
    agent.run("لبن", sess)
    assert ids(sess) == [10, 12]  # 20 exists in the store but no tool returned it in this conversation


def test_leftover_or_malformed_markers_are_stripped():
    assert guards.extract_show("تمام 👌 [[show: 1,2]] تحب؟ [[sho")[1] == "تمام 👌  تحب؟"
    assert guards.extract_show("x [[ SHOW 7 ]]")[0] == [7]


def test_invented_price_is_sent_back_for_correction_once():
    agent, llm, _, sess = make(call("search_products", queries=["iphone"]), say("آيفون 11 بـ 9999 جنيه بس!"),
                               say("آيفون 11 بـ 14500 جنيه 👌"))
    assert agent.run("آيفون", sess) == "آيفون 11 بـ 14500 جنيه 👌" and "9999" in llm.requests[-1][-1]["content"]


def test_persistent_invented_price_never_reaches_the_customer():
    agent, _, _, sess = make(say("بـ 123 جنيه"), say("بـ 123 جنيه"))
    assert "123" not in agent.run("بكام آيفون؟", sess)


def test_real_prices_budget_echo_and_discount_difference_are_allowed():
    agent, llm, _, sess = make(call("search_products", queries=["iphone"]),
                               say("ميزانيتك 15 ألف، آيفون 12 بـ 14,900 جنيه وموفر 1600 جنيه عن سعره الأصلي"))
    assert "1600" in agent.run("آيفون بحدود 15 ألف", sess) and llm.calls == 2


# ===================================================================== corrupted / empty output
CORRUPT = ["ده كُ... به ت... أعتون... Oops! Sorry ...", "Oops! Sorry, something went wrong.", '{"error": "rate_limit"}',
           "<think>الزبون عايز</think>", "تمام تمام تمام تمام تمام", "لقيت Error 429 Too Many Requests", "ده ك به ت أع و ي ف م ن ب ل ع", "[system] retry", "   "]


@pytest.mark.parametrize("bad", CORRUPT)
def test_corrupted_output_is_rejected_and_one_clean_retry_is_used(bad):
    agent, llm, _, sess = make(say(bad), say("آه 😂 فهمتك، تقصد لبن للشرب. خليني أشوف المتاح."))
    assert agent.run("لا اقصد عايز لبن", sess) == "آه 😂 فهمتك، تقصد لبن للشرب. خليني أشوف المتاح."
    assert llm.calls == 2 and "could not be shown" in llm.requests[1][-1]["content"]
    assert bad.strip() not in json.dumps(llm.requests[1], ensure_ascii=False) or not bad.strip()  # the garbage is not fed back


@pytest.mark.parametrize("bad", CORRUPT)
def test_persistently_corrupted_output_never_reaches_the_customer(bad):
    agent, _, _, sess = make(say(bad), say(bad))
    out = agent.run("لا اقصد عايز لبن", sess)
    assert guards.reply_problem(out) is None and out.strip() and "Oops" not in out and "Sorry" not in out


def test_corrupted_final_reply_after_an_unambiguous_search_still_shows_the_products():
    bad = "ده كُ... به ت... أعتون... Oops! Sorry ..."
    agent, _, _, sess = make(call("search_products", queries=["تيشيرت"], category="تيشيرتات"), say(bad), say(bad))
    assert agent.run("وريني تيشيرتات", sess) == "لقيتلك دول 👇" and ids(sess) == [40]


def test_corrupted_reply_after_an_ambiguous_search_asks_instead_of_guessing():
    bad = "ده كُ... به ت... أعتون..."
    agent, _, _, sess = make(call("search_products", queries=["milk", "حليب"]), say(bad), say(bad))
    out = agent.run("لبن", sess)
    assert "products" not in sess.messages[-1] and "توضحلي" in out


def test_no_result_claim_after_a_successful_budget_search_is_rejected_and_cards_survive():
    contradictory = "مع الأسف مفيش موبايل شاومي بسعر أقل من 5000 جنيه في المتجر دلوقتي."
    client = FakeClient(CATALOG + [prod(41, "Xiaomi Redmi A5", 4500, "Xiaomi", PHONES, ELEC)])
    agent, _, _, sess = make(
        call("search_products", queries=["موبايل", "phone"], category="موبايلات", brand="Xiaomi", max_price=5000),
        say(contradictory), say(contradictory), client=client,
    )
    out = agent.run("زوّد الميزانية لـ5000 وخليك في شاومي", sess)
    assert out == "لقيتلك دول 👇"
    assert ids(sess) == [41]


def test_no_correction_attempt_while_the_provider_is_rate_limited():
    class Cooling(ScriptedLLM):
        def cooling(self):
            return 5.0
    llm = Cooling(say("Oops!"))
    agent, _, _, sess = make(llm=llm)
    agent.run("اهلا", sess)
    assert llm.calls == 1


# ===================================================================== LLM failures keep valid commerce results
def test_llm_fails_after_a_successful_unambiguous_search_cards_survive():
    """The scenario from real testing: T-shirts found, then the next Groq call fails."""
    class Dies(ScriptedLLM):
        def chat(self, messages, tools=None):
            if len(self.requests) >= 1:
                raise LLMError("boom")
            return super().chat(messages, tools)
    llm = Dies(call("search_products", queries=["تيشيرت"], category="تيشيرتات"))
    agent, _, _, sess = make(llm=llm)
    out = agent.run("وريني تيشيرتات", sess)
    assert out == "لقيتلك دول 👇" and ids(sess) == [40] and sess.shown[0].id == 40


def test_rate_limit_after_a_successful_search_keeps_the_cards_too():
    class Limited(ScriptedLLM):
        def chat(self, messages, tools=None):
            if len(self.requests) >= 1:
                raise RateLimited(8)
            return super().chat(messages, tools)
    agent, _, _, sess = make(llm=Limited(call("search_products", queries=["iphone"], category="موبايلات", brand="apple")))
    assert agent.run("وريني آيفون", sess) == "لقيتلك دول 👇" and ids(sess) == [20, 21, 22]


def test_rate_limit_with_ambiguous_results_says_busy_not_broken_and_shows_nothing_random():
    class Limited(ScriptedLLM):
        def chat(self, messages, tools=None):
            if len(self.requests) >= 1:
                raise RateLimited(8)
            return super().chat(messages, tools)
    agent, _, _, sess = make(llm=Limited(call("search_products", queries=["milk", "حليب"])))
    out = agent.run("لبن", sess)
    assert "زحمة" in out and "products" not in sess.messages[-1]


def test_llm_down_before_anything_gives_a_polite_message_without_technical_words():
    agent, _, client, sess = make(fail=True)
    out = agent.run("أهلاً", sess)
    assert guards.reply_problem(out) is None and client.calls == []


def test_missing_api_key_is_handled_like_an_llm_failure():
    out = SalesAgent(TixTools(FakeClient()), UnavailableLLM()).run("عايز موبايل", Session())
    assert guards.reply_problem(out) is None


def test_store_api_down_is_explained_by_the_model_without_inventing_products():
    agent, llm, _, sess = make(call("search_products", queries=["حليب"]), say("مش قادر أوصل للمتجر دلوقتي 🙏 جرّب كمان شوية."),
                               client=FakeClient(fail=True))
    agent.run("عايز حليب", sess)
    assert tool_results(llm.requests[1])[0]["error"] == "store_unavailable" and "products" not in sess.messages[-1]


# ===================================================================== loop discipline
def test_endless_tool_calling_is_cut_at_the_step_budget_and_the_last_call_offers_no_tools():
    agent, llm, _, sess = make(*([call("list_categories")] * config.MAX_STEPS), say("خلصت 👌 قولّي عايز إيه بالظبط؟"))
    assert agent.run("وريني كل حاجة", sess).startswith("خلصت") and llm.calls == config.MAX_STEPS + 1
    assert llm.tools_offered[-1] is None and all(t for t in llm.tools_offered[:-1])


def test_malformed_tool_arguments_are_returned_to_the_model_as_an_error():
    bad = {"content": "", "tool_calls": [{"id": "x", "name": "search_products", "arguments": "{not json"}]}
    agent, llm, _, sess = make(bad, say("ممكن توضح عايز إيه؟"))
    agent.run("حاجة", sess)
    assert tool_results(llm.requests[1])[0]["error"] == "invalid_arguments"


def test_parallel_tool_calls_run_in_one_round():
    two = {"content": "", "tool_calls": [
        {"id": "a", "name": "search_products", "arguments": json.dumps({"queries": ["xiaomi"], "category": "موبايلات"})},
        {"id": "b", "name": "search_products", "arguments": json.dumps({"queries": ["samsung"], "category": "موبايلات"})}]}
    agent, llm, _, sess = make(two, show("دول الاتنين 👇", 24, 23))
    agent.run("شاومي وسامسونج", sess)
    assert llm.calls == 2 and len(tool_results(llm.requests[1])) == 2 and ids(sess) == [24, 23]


def test_context_sent_to_the_model_stays_small_after_a_long_chat():
    replies = []
    for _ in range(12):
        replies += [call("search_products", queries=["iphone"], category="موبايلات"), show("تمام 👌", 20, 21, 22)]
    agent, llm, _, sess = make(*replies)
    for _ in range(12):
        agent.run("وريني آيفون", sess)
    assert max(len(json.dumps(r, ensure_ascii=False)) for r in llm.requests) < 12000
    last = llm.requests[-1]
    assert sum(1 for m in last if m["role"] in ("user", "assistant") and not m.get("tool_calls")) == 15  # 14 history + current


def test_new_conversation_resets_everything():
    agent, _, _, sess = make(call("search_products", queries=["iphone"]), show("ok", 21))
    agent.run("آيفون", sess)
    sess.reset()
    assert sess.messages == [] and sess.shown == [] and sess.seen == {} and sess.last_search == {} and sess.found is None
