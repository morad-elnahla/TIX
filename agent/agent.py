"""The sales agent: one plain tool-calling loop.

    customer message -> LLM (sees history + what is on screen) -> [tool calls -> real TIX data -> LLM]* -> reply (+ cards)

No router, no judge, no separate "understand" step: the same LLM that talks also decides when to call tools,
which phrasings to search, and which of the returned real products actually fit. Code only enforces what must not be
trusted to a model: budget/brand/test-product filters, ids that may be shown, and price statements."""
from __future__ import annotations

import json
import logging
import re

import config
from agent import guards, render
from agent.llm import GroqLLM, LLMError, RateLimited
from agent.prompts import SYSTEM, context_block
from agent.state import Session
from agent.tools import SCHEMAS, TixTools

log = logging.getLogger("tix.agent")


class UnavailableLLM:
    """Used when no GROQ_API_KEY is configured: every call fails cleanly so the UI can say so."""

    def chat(self, messages, tools=None):
        raise LLMError("GROQ_API_KEY is not set")


def default_llm():
    try:
        return GroqLLM()
    except LLMError:
        return UnavailableLLM()


class SalesAgent:
    def __init__(self, tools: TixTools | None = None, llm=None):
        self.tools = tools or TixTools()
        self.llm = llm or default_llm()

    def warmup(self, block: bool = False) -> None:
        self.tools.warmup(block)

    # ------------------------------------------------------------------ public
    def run(self, user_text: str, session: Session) -> str:
        """Answer one customer message. Appends the user + assistant turn (with product cards) to session.messages."""
        user_text = (user_text or "").strip()
        lang = render.detect_lang(user_text)
        session.user_text, session.pending_cards, session.found = user_text, [], None
        try:
            text = self._converse(user_text, session)
        except RateLimited as e:
            log.warning("LLM rate limited (retry in ~%.0fs): answering from what was already found", e.retry_after)
            text = self._fallback(session, lang, "busy")
        except LLMError as e:
            log.warning("LLM failure: %s", e)
            text = self._fallback(session, lang, "down")
        except Exception:  # noqa: BLE001 - never leak a stack trace to a customer
            log.exception("agent failure")
            text = self._fallback(session, lang, "down")
        text = text or render.msg(lang, "llm_down")
        cards = list(session.pending_cards)
        session.messages += [{"role": "user", "content": user_text},
                             {"role": "assistant", "content": text, **({"products": cards} if cards else {})}]
        return text

    # ------------------------------------------------------------------ the loop
    def _converse(self, user_text: str, session: Session) -> str:
        lang = render.detect_lang(user_text)
        messages = [{"role": "system", "content": SYSTEM + "\n\n" + context_block(session)},
                    *session.history(), {"role": "user", "content": user_text}]
        corrected = False
        for step in range(config.MAX_STEPS + 1):  # the extra pass offers no tools: it forces a final answer
            tools = SCHEMAS if step < config.MAX_STEPS else None
            reply = self.llm.chat(messages, tools)
            calls = reply["tool_calls"] if tools else []
            if calls:
                messages.append({"role": "assistant", "content": reply["content"] or None, "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
                for c in calls:
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": self._execute(c, session)})
                continue
            ids, text = guards.extract_show(reply["content"])
            bad_prices = guards.unverified_prices(text, self._allowed_prices(session, user_text))
            problem = guards.reply_problem(text) or (f"price {bad_prices}" if bad_prices else None)
            if problem:
                log.warning("reply rejected (%s): %.80r", problem, reply["content"])
                if corrected or getattr(self.llm, "cooling", lambda: 0)() > 0:
                    return self._fallback(session, lang, "bad")  # one correction at most, and never while rate-limited
                corrected = True
                note = ("[system] Your last reply could not be shown to the customer. Answer again in one or two clear, natural sentences"
                        + (f", using only real prices from the tool results (not {', '.join(f'{b:g}' for b in bad_prices)})."
                           if bad_prices else "."))
                messages.append({"role": "user", "content": note})
                continue
            self._show(ids, session)
            return guards.plain_text(text)
        return self._fallback(session, lang, "down")

    def _show(self, ids: list[int], session: Session) -> None:
        """Cards for the ids the model asked to display: only products a tool returned in this conversation."""
        by_id = {p.id: p for p in session.seen.values() if p.id is not None}
        chosen = [by_id[i] for i in ids if i in by_id][: config.MAX_CARDS]
        if ids and len(chosen) < len(ids):
            log.warning("model asked to show unknown product ids: %s", [i for i in ids if i not in by_id])
        if chosen:
            session.pending_cards = [render.card(p, session.media.get(p.id)) for p in chosen]
            session.shown = chosen

    def _fallback(self, session: Session, lang: str, why: str) -> str:
        """Customer-facing text when the LLM cannot produce a usable reply. Valid results already fetched this turn survive:
        if the latest search was unambiguous (one category, or the model had narrowed by category) its products are shown."""
        found = session.found
        if found and found[0] and found[1]:
            products = found[0][: config.MAX_CARDS]
            session.pending_cards = [render.card(p, session.media.get(p.id)) for p in products]
            session.shown = list(products)
            return render.msg(lang, "found")
        if found and found[0] and why == "bad":
            return render.msg(lang, "unsure")  # ambiguous results + unusable reply: ask rather than guess
        return render.msg(lang, "busy" if why == "busy" else "llm_down")

    def _execute(self, call: dict, session: Session) -> str:
        try:
            args = json.loads(call["arguments"] or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return json.dumps({"error": "invalid_arguments", "hint": "Arguments must be a JSON object."})
        result = json.dumps(self.tools.run(call["name"], args, session), ensure_ascii=False, default=str)
        if len(result) > config.TOOL_RESULT_CHARS:  # hard cap: never flood the model
            result = json.dumps({"error": "result_too_large", "hint": "Narrow the request (category, brand, price)."})
        return result

    @staticmethod
    def _allowed_prices(session: Session, user_text: str) -> set[float]:
        allowed = set(session.allowed_prices)
        for x in re.findall(r"\d+(?:\.\d+)?", guards.normalize(user_text).replace(",", "")):
            allowed.update({float(x), float(x) * 1000})  # "15 ألف" may be echoed back as 15000
        for p in session.seen.values():  # "you save X" differences are derived from real prices
            if p.price and p.price_before and p.price_before > p.price:
                allowed.add(p.price_before - p.price)
        return allowed
