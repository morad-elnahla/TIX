"""Chat panel rendering (presentation only). State lives in st.session_state (one Session per browser session);
all conversation and retrieval logic lives in the agent."""
from __future__ import annotations

import html
import logging
import re
import time

import streamlit as st

from agent.agent import UnavailableLLM
from agent.render import detect_lang, fmt_price, msg
from agent.state import Session
from ui import styles

log = logging.getLogger("tix.ui")
# Starter actions as (icon key, message), in DOM order of the RTL 2x2 grid: first = top-right. The icon is drawn by CSS; the message is
# an ordinary user message: clicking a chip sends it through the SAME agent flow as typing it. Only capabilities TIX really supports
# (browse, budget filter, discounted products, sort=newest); there is no verified best-seller capability.
CHIPS = [("price", "منتجات تحت 300 جنيه"), ("cart", "وريني المنتجات"), ("new", "وصل حديثًا"), ("deal", "عروض وخصومات")]
AV_A, AV_U = '<div class="av a"></div>', '<div class="av u"></div>'


def get_session() -> Session:
    if "tix_session" not in st.session_state:
        st.session_state["tix_session"] = Session()
    return st.session_state["tix_session"]


def _esc(text: str) -> str:
    """Single-line HTML (blank lines would end the markdown HTML block): newlines -> <br>, indents kept."""
    lines = []
    for line in html.escape(text).split("\n"):
        indent = len(line) - len(line.lstrip(" "))
        lines.append("&nbsp;" * indent + line.lstrip(" "))
    return "<br>".join(lines)


# Reply-to: the quoted message travels WITH the user's text so TIX knows what it refers to. Stored form: REPLY_MARK + quote + "\n" + text.
REPLY_MARK = "↩ رد على: "
_REPLY_RE = re.compile(r"^↩ رد على: «(.*?)»\n(.*)$", re.S)


def _short(text: str, n: int = 110) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n].rstrip() + "…"


def _with_reply(text: str, quote: str | None) -> str:
    return f"{REPLY_MARK}«{_short(quote)}»\n{text}" if quote else text


def _bubble(role: str, text: str) -> str:
    quote = ""
    hit = _REPLY_RE.match(text)
    if hit:
        quote, text = f'<div class="rq">{_esc(hit.group(1))}</div>', hit.group(2)
    if role == "user":
        return f'<div class="row u">{AV_U}<div class="bubble u">{quote}{_esc(text)}</div></div>'
    return f'<div class="row a"><div class="bubble a">{quote}{_esc(text)}</div>{AV_A}</div>'


def _safe_url(u) -> str:
    return html.escape(u, quote=True) if isinstance(u, str) and u.startswith(("http://", "https://")) else ""


def _cards(products: list[dict], lang: str) -> str:
    cur = msg(lang, "currency")
    out = []
    for i, c in enumerate(products, 1):
        old = f'<span class="cs">{fmt_price(c["price_before"])}</span>' if c.get("price_before") else ""
        badge = f'<span class="cd">-{c["discount_pct"]:g}%</span>' if c.get("discount_pct") else ""
        img = _safe_url(c.get("image"))
        pic = f'<img class="ci" src="{img}" alt="" loading="lazy" referrerpolicy="no-referrer">' if img else ""
        inner = (f'<div class="cn">{i}</div>{pic}<div class="ct">{html.escape(c["name"])}</div>'
                 f'<div class="cb">{html.escape(c.get("brand") or "")}</div>'
                 f'<div class="cp">{fmt_price(c["price"])} {cur}{old}</div>{badge}')
        url = _safe_url(c.get("url"))
        body = (f'<a class="cl" href="{url}" target="_blank" rel="noopener noreferrer">{inner}</a>' if url
                else f'<div class="cl">{inner}</div>')
        vid = _safe_url(c.get("video"))
        video = (f'<a class="cv" href="{vid}" target="_blank" rel="noopener noreferrer">▶ {"فيديو" if lang == "ar" else "Video"}</a>'
                 if vid else "")
        out.append(f'<div class="card">{body}{video}</div>')
    return f'<div class="cards">{"".join(out)}</div>'


def _message_html(m: dict, text: str | None = None) -> str:
    if m["role"] == "user":
        return _bubble("user", m["content"])
    if text is not None:  # streaming: text only, cards appear when the answer is complete
        return _bubble("assistant", text)
    cards = _cards(m["products"], detect_lang(m["content"])) if m.get("products") else ""
    return _bubble("assistant", m["content"]) + cards


def _chunks(text: str):
    words = re.split(r"(\s+)", text)  # keep whitespace so the layout is identical while streaming
    for i in range(0, len(words), 6):
        yield "".join(words[i:i + 6])


def _new_conversation() -> None:
    """Clears chat history AND shopping state of THIS session only."""
    get_session().reset()


def _queue_prompt(text: str) -> None:
    st.session_state["queued_prompt"] = text


def _set_reply(text: str) -> None:
    st.session_state["reply_to"] = text


def _clear_reply() -> None:
    st.session_state.pop("reply_to", None)


def render_chat(agent) -> None:
    sess = get_session()
    queued = st.session_state.pop("queued_prompt", None)
    with st.container(key="chat_panel"):
        st.markdown(styles.header_html(), unsafe_allow_html=True)
        st.button(":material/edit_square:", key="new_chat", help="محادثة جديدة", on_click=_new_conversation)
        msgs = st.container(key="chat_msgs", border=False)
        if isinstance(agent.llm, UnavailableLLM):
            st.caption("⚠️ GROQ_API_KEY غير مضبوط — الشات شغّال بالوضع المحدود.")
        with st.container(key="chat_composer"):
            reply_to = st.session_state.get("reply_to")
            if reply_to:
                with st.container(key="chat_replybar"):
                    st.markdown(f'<div class="rbar"><b>رد على</b><span>{_esc(_short(reply_to, 80))}</span></div>',
                                unsafe_allow_html=True)
                    st.button("✕", key="reply_cancel", help="إلغاء الرد", on_click=_clear_reply)
            typed = st.chat_input("اسأل عن أي منتج، عروض، أو تفاصيل...", key="chat_in")
        if typed and reply_to:
            typed = _with_reply(typed, reply_to)
            st.session_state.pop("reply_to", None)
        prompt = typed or queued

    with msgs:
        if not sess.messages and not prompt:  # welcome state: greeting, quick actions beneath it
            with st.container(key="chat_welcome"):
                st.markdown(styles.WELCOME_HTML, unsafe_allow_html=True)
                with st.container(key="chat_chips"):
                    for icon, text in CHIPS:
                        st.button(text, key=f"chip_{icon}", on_click=_queue_prompt, args=(text,))
        for i, m in enumerate(sess.messages):
            st.markdown(_message_html(m), unsafe_allow_html=True)
            if not prompt:  # small "reply" action under each bubble, on the bubble's side
                hit = _REPLY_RE.match(m["content"])
                side = "u" if m["role"] == "user" else "a"
                st.button("↩ رد", key=f"reply_{side}_{i}", on_click=_set_reply,
                          args=(hit.group(2) if hit else m["content"],))
        if prompt:
            st.markdown(_bubble("user", prompt), unsafe_allow_html=True)
            slot = st.empty()
            slot.markdown(styles.TYPING_HTML, unsafe_allow_html=True)  # loading state while retrieval runs
            try:
                answer = agent.run(prompt, sess)
            except Exception:  # noqa: BLE001 - never show a stack trace to a customer
                log.exception("agent failure")
                answer = msg(detect_lang(prompt), "llm_down")
                sess.messages += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
            last = sess.messages[-1]
            to_stream = last["content"]
            shown = ""
            for piece in _chunks(to_stream):
                shown += piece
                slot.markdown(_bubble("assistant", shown + "▍"), unsafe_allow_html=True)
                time.sleep(0.02)
            slot.markdown(_message_html(last), unsafe_allow_html=True)