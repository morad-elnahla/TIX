"""UI structure and the chip -> agent flow with a recording Streamlit double (no browser). CSS is asserted textually:
how it LOOKS still has to be checked in a real browser."""
import pytest

pytest.importorskip("streamlit")

from agent.agent import SalesAgent  # noqa: E402
from agent.state import Session  # noqa: E402
from agent.tools import TixTools  # noqa: E402
from tests.fake_streamlit import FakeSt  # noqa: E402
from tests.fakes import FakeClient, ScriptedLLM, say  # noqa: E402
from ui import chat  # noqa: E402


def chips(st):
    return [e for e in st.of("button") if e[1].startswith("chip_")]


def setup(monkeypatch, typed=None, session_state=None, llm=None):
    st = FakeSt(typed)
    st.session_state.update(session_state or {})
    monkeypatch.setattr(chat, "st", st)
    monkeypatch.setattr(chat.time, "sleep", lambda *_: None)
    return st, SalesAgent(TixTools(FakeClient()), llm or ScriptedLLM(say("تمام 👋")))


def test_bubbles_are_single_line_and_escape_user_text():
    h = chat._bubble("user", "<script>alert(1)</script>\n\nسطر")
    assert "\n" not in h and "<script>" not in h and "&lt;script&gt;" in h


def test_cards_show_image_discount_link_and_video_and_escape_everything():
    cards = [{"id": 1, "name": "A <b>x</b>", "brand": "Brand", "price": 40.0, "price_before": 50.0, "discount_pct": 20,
              "url": "https://tix-eg.com/product/1", "image": "https://cdn/x.jpg", "video": "https://cdn/x.mp4"},
             {"id": 2, "name": "B", "brand": "", "price": 100.0, "price_before": None, "discount_pct": None, "url": None,
              "image": "javascript:alert(1)", "video": None}]
    h = chat._cards(cards, "ar")
    assert "\n" not in h and "&lt;b&gt;" in h and "-20%" in h and "جنيه" in h and h.count('class="card"') == 2
    assert h.count('href="https://tix-eg.com/product/1"') == 1 and 'src="https://cdn/x.jpg"' in h and "x.mp4" in h
    assert "javascript:" not in h and h.count('class="cd"') == 1


def test_message_html_cards_only_for_assistant_messages_that_have_products_and_never_while_streaming():
    m = {"role": "assistant", "content": "لقيتلك 👌", "products": [{"id": 1, "name": "N", "brand": "", "price": 5.0}]}
    assert 'class="cards"' in chat._message_html(m)
    assert 'class="cards"' not in chat._message_html({"role": "assistant", "content": "hello"})
    assert 'class="cards"' not in chat._message_html(m, text="stream…")


def test_chips_render_under_the_welcome_message_inside_the_conversation_area(monkeypatch):
    st, agent = setup(monkeypatch)
    chat.render_chat(agent)
    assert [e[1] for e in chips(st)] == [f"chip_{i}" for i, _ in chat.CHIPS]
    greeting = next(i for i, e in enumerate(st.events) if e[0] == "markdown" and "أهلاً وسهلاً" in e[1])
    first_chip = next(i for i, e in enumerate(st.events) if e[0] == "button" and e[1].startswith("chip_"))
    assert greeting < first_chip and all("chat_msgs" in e[2] for e in chips(st))


def test_chips_disappear_once_the_conversation_has_started(monkeypatch):
    st, agent = setup(monkeypatch)
    sess = Session()
    sess.messages += [{"role": "user", "content": "اهلا"}, {"role": "assistant", "content": "x"}]
    st.session_state["tix_session"] = sess
    chat.render_chat(agent)
    assert chips(st) == [] and "new_chat" in st.buttons


def test_clicking_a_chip_goes_through_the_normal_agent_flow(monkeypatch):
    st, agent = setup(monkeypatch)
    chat.render_chat(agent)
    label, on_click, args = st.buttons["chip_deal"]
    on_click(*args)
    assert st.session_state["queued_prompt"] == label
    st.events.clear()
    chat.render_chat(agent)
    sess = st.session_state["tix_session"]
    assert sess.messages[0]["content"] == label and sess.messages[1]["content"] == "تمام 👋" and chips(st) == []


def test_typed_message_with_cards_renders_text_then_cards(monkeypatch):
    from tests.fakes import call, show
    llm = ScriptedLLM(call("search_products", queries=["حليب"], category="حليب"), show("دي أقرب حاجة 👌", 10))
    st, agent = setup(monkeypatch, typed="عايز حليب", llm=llm)
    chat.render_chat(agent)
    html_out = [e[1] for e in st.of("markdown")]
    assert any('class="cards"' in h and "جهينة" in h for h in html_out)


def test_unexpected_agent_crash_shows_a_polite_message_not_a_stack_trace(monkeypatch):
    st, agent = setup(monkeypatch, typed="أهلاً")
    agent.run = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    chat.render_chat(agent)
    assert any("مشكلة بسيطة" in e[1] for e in st.of("markdown")) and not any("Traceback" in e[1] for e in st.of("markdown"))


# ------------------------------------------------------------------------------------------- the floating-widget design
def test_assistant_is_a_floating_widget_over_the_store_with_a_launcher_and_no_attachment_control():
    from ui import styles
    css = styles.CSS + styles.header_html() + styles.WELCOME_HTML + styles.FAB_HTML
    assert 'id="tix-fab"' in styles.FAB_HTML and ".st-key-chat_panel {{" not in css
    assert "position:fixed !important; bottom:90px; right:22px" in css and "body.tix-open .st-key-chat_panel" in css
    assert not any(w in css.lower() for w in ("attach", "paperclip", "file_uploader", "upload"))
    assert "store-hero" in styles.STORE_HTML and "TIX" in styles.STORE_HTML       # a simple backdrop only


def test_widget_side_is_a_single_setting():
    from ui import styles
    right, left = styles.build_css("right"), styles.build_css("left")
    assert "bottom:90px; right:22px" in right and "transform-origin:bottom right" in right and "#tix-fab {" in right
    assert "bottom:90px; left:22px" in left and "transform-origin:bottom left" in left and "bottom:22px; left:22px" in left
    assert styles.build_css("anything-else") == right


def test_header_is_symmetric_with_close_and_new_chat_at_equal_widths():
    from ui import styles
    h = styles.header_html()
    assert 'id="tix-close"' in h and "TIX Assistant" in h and "مساعد التسوق الذكي" in h and "tix-dot" in h and "href" not in h
    assert "grid-template-columns:56px 1fr 56px" in styles.CSS                      # equal side columns around the centred identity


def test_widget_js_binds_once_opens_by_default_and_scrolls_to_the_newest_message():
    from ui import styles
    js = styles.WIDGET_JS
    assert "__tixBound" in js and "#tix-fab" in js and "#tix-close" in js and "tix-open" in js and "scrollTop" in js


def test_welcome_has_the_three_part_greeting_and_chips_form_a_two_by_two_grid():
    from ui import styles
    assert styles.WELCOME_HTML.count('class="w') == 3 and "أهلاً وسهلاً" in styles.WELCOME_HTML
    assert [k for k, _ in chat.CHIPS] == ["price", "cart", "new", "deal"] and len(chat.CHIPS) == 4
    assert "grid-template-columns:1fr 1fr" in styles.CSS and all(f"st-key-chip_{k} button::before" in styles.CSS for k, _ in chat.CHIPS)


def test_the_real_app_script_runs_end_to_end_without_exceptions(monkeypatch):
    """Runs app.py itself with Streamlit's headless AppTest: no key, no network -> the page must still render and a chip must answer politely."""
    at_mod = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("config.get_groq_key", lambda: "")
    at = at_mod.AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert all(t in labels for _, t in chat.CHIPS)
    next(b for b in at.button if b.label == "عروض وخصومات").click().run()
    assert not at.exception
    bodies = " ".join(m.value for m in at.markdown)
    assert "مشكلة بسيطة" in bodies and "Traceback" not in bodies and "GROQ" not in bodies.replace("GROQ_API_KEY غير", "")