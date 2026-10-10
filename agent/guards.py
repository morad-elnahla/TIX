"""Small deterministic helpers: text normalisation and the checks that must never be left to an LLM
(budget, brand, hidden test products, duplicates, price statements)."""
from __future__ import annotations

import re

import config
from tix.schemas import Product

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DIAC = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def normalize(text: str) -> str:
    """Lower-case, strip diacritics, unify Arabic letter variants (أإآ→ا, ى→ي, ة→ه) and Arabic digits."""
    t = _DIAC.sub("", (text or "").lower().translate(_AR_DIGITS))
    t = re.sub("[أإآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = t.replace("é", "e").replace("è", "e").replace("’", "'")
    return t


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", normalize(text))


def to_number(v) -> float | None:
    """Tool arguments come from an LLM: accept 15000, "15000", "15,000", "١٥٠٠٠"; reject everything else."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"\d+(?:\.\d+)?", v.translate(_AR_DIGITS).replace(",", ""))
        return float(m.group()) if m else None
    return None


# ---------------------------------------------------------------- budget / brand
def within_budget(p: Product, lo: float | None, hi: float | None) -> bool:
    if lo is None and hi is None:
        return True
    if p.price is None:
        return False
    return (lo is None or p.price >= lo) and (hi is None or p.price <= hi)


def brand_matches(p: Product, brand: str) -> bool:
    """Plain normalised substring match on the brand OR the name (the store's brand field is sometimes empty or wrong: a New Balance
    sneaker filed under another brand must still be found by "New Balance"). The LLM passes the spelling it wants; if that misses, the
    search tool lists the brands that exist so it can retry. No alias table."""
    b = normalize(brand).strip()
    return not b or b in normalize(p.brand) or b in normalize(p.name)


# ---------------------------------------------------------------- test products / duplicates
def hide_test_products(products: list[Product], user_text: str = "") -> list[Product]:
    if not config.HIDE_TEST_PRODUCTS:
        return list(products)
    t = normalize(user_text)
    if any(normalize(k) in t for k in config.USER_TEST_KEYWORDS):
        return list(products)
    return [p for p in products if not re.search(config.TEST_PRODUCT_PATTERN, p.name or "", flags=re.IGNORECASE)]


def dedupe(products: list[Product]) -> list[Product]:
    seen, out = set(), []
    for p in products:
        key = p.id if p.id is not None else normalize(p.name)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def sort_products(products: list[Product], sort: str | None) -> list[Product]:
    if sort == "price_low":
        return sorted(products, key=lambda p: (p.price is None, p.price or 0))
    if sort == "price_high":
        return sorted(products, key=lambda p: -(p.price or 0))
    if sort == "newest":  # no date field is documented; a higher id is the best available proxy for "newer"
        return sorted(products, key=lambda p: -(p.id or 0))
    return list(products)


# ---------------------------------------------------------------- price statements in the model's reply
_CUR = r"(?:جنيه|جنية|ج\.م|ج\b|egp|le\b|l\.e|pounds?)"
_PRICE_RE = re.compile(rf"(\d[\d,]*(?:\.\d+)?)\s*{_CUR}|{_CUR}\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)


def prices_in(text: str) -> list[float]:
    out = []
    for m in _PRICE_RE.finditer(normalize(text)):
        raw = (m.group(1) or m.group(2)).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def unverified_prices(text: str, allowed: set[float]) -> list[float]:
    """Prices the reply states next to a currency word that appear nowhere in real data (or the customer's own words)."""
    return [x for x in prices_in(text) if not any(abs(x - a) < 0.51 for a in allowed)]


# ---------------------------------------------------------------- is the model's reply fit to show a customer?
_SHOW_RE = re.compile(r"\[\[\s*show\s*:?\s*([\d\s,،]*?)\s*\]\]", re.IGNORECASE)
_LEFTOVER_MARK = re.compile(r"\[\[[^\n]*|\]\]")


def extract_show(text: str) -> tuple[list[int], str]:
    """Split a reply into (product ids the model asked to display, clean customer-facing text).
    The ids still have to be validated against what the tools returned (see SalesAgent)."""
    ids: list[int] = []
    for m in _SHOW_RE.finditer(text or ""):
        ids += [int(x) for x in re.findall(r"\d+", m.group(1))]
    clean = _LEFTOVER_MARK.sub("", _SHOW_RE.sub("", text or ""))
    return list(dict.fromkeys(ids)), re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", clean)).strip()


_LEAK = re.compile(
    r"oops|something went wrong|an error occurred|internal (?:server )?error|rate.?limit|too many requests|(?:http|error|status|code)\W{0,3}(?:404|429)\b|\b(?:404|429)\W{0,3}(?:error|not found)|"
    r"traceback|exception|tool_call|function_call|<\s*/?\s*think|\[system\]|api.?key|\bgroq\b|openai|\bjson\b|"
    r"```|\bnull\b|\bundefined\b|حدث خطأ|حصل خطأ|خطأ داخلي|"
    r"as an ai|language model|نموذج لغوي|ذكاء اصطناعي من|\btool\b|\bapi\b",
    re.IGNORECASE)
_FOREIGN_SCRIPT = re.compile(r"[\u0400-\u04FF\u0590-\u05FF\u0E00-\u0E7F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]")
_MIXED_WORD = re.compile(r"[A-Za-z]+[\u0600-\u06FF]|[\u0600-\u06FF]+[A-Za-z]")


def reply_problem(text: str) -> str | None:
    """Why `text` must NOT reach a customer, or None if it is usable. Judges the whole reply (not character fixes):
    internals leaking, JSON/code, scripts that cannot belong, and the signature of a degenerated generation
    (dangling fragments, ellipsis spam, repeated words, a pile of 1-2 letter pieces)."""
    t = (text or "").strip()
    if len(t) < 2:
        return "empty"
    if len(t) > 1500:
        return "too long"
    if _LEAK.search(t):
        return "internal wording"
    if t[0] in "{[" or re.match(r'\s*"?\w+"?\s*:\s*[\[{"]', t):
        return "looks like JSON"
    if _FOREIGN_SCRIPT.search(t):
        return "foreign script"
    if len(_MIXED_WORD.findall(t)) >= 2:
        return "garbled mixed-script words"
    words = re.findall(r"[^\W\d_]+", t)
    if len(re.findall(r"(?:\.\.\.|…)", t)) >= 3 or len(re.findall(r"\S+(?:\.\.\.|…)", t)) >= 2:
        return "fragments"
    if len(words) >= 4 and any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        return "repetition"
    if len(words) >= 8 and sum(1 for w in words if len(w) <= 2) / len(words) > 0.55:
        return "fragments"
    letters = [w for w in words if len(w) >= 2]
    if len(words) >= 24 and len(set(letters)) < len(letters) * 0.28:  # a legitimate product list repeats brand/unit words, so be lenient
        return "repetition"
    return None


def plain_text(text: str) -> str:
    """The chat bubble is plain text: turn the markdown a model likes to emit (**bold**, # headings, - bullets, `code`) into
    clean text instead of showing stray asterisks."""
    t = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text or "", flags=re.S)
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)
    t = re.sub(r"(?m)^\s*[-*•]\s+", "• ", t)
    t = t.replace("*", "").replace("`", "")
    return re.sub(r"\n{3,}", "\n\n", t).strip()