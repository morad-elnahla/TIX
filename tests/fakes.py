"""LOCAL TEST doubles. Nothing here is real TIX data except where a docstring says VERIFIED.

IMPORTANT: ScriptedLLM does NOT understand language. Each test scripts what a competent model would do (which tool it
calls, with which arguments, what it finally says). The tests therefore verify the orchestration around the model:
tools, filters, state, guards, failure handling. They do not and cannot measure Arabic understanding quality."""
from __future__ import annotations

import json

from agent import guards


def prod(pid, name, after, brand="", sub=(None, ""), cat=(None, ""), before=None, discount=0, desc="", images=None, **extra):
    return {"id": pid, "name": name, "short_description": desc, "price_before": before if before is not None else after,
            "price_after": after, "discount": discount,
            "brand": {"id": 1, "name": brand} if brand else None,
            "category": {"id": cat[0], "name": cat[1]} if cat[0] else None,
            "subcategory": {"id": sub[0], "name": sub[1]} if sub[0] else None,
            "images": images if images is not None else [f"storage/p{pid}.jpg"], **extra}


MILK, MILKSUB = (5, "ألبان وأجبان"), (50, "حليب")
HAIR, SHAMPOO = (2, "العناية بالشعر"), (40, "شامبو")
PHONES, ELEC = (90, "موبايلات"), (9, "الإلكترونيات")
WATCH, ACC = (91, "ساعات"), (10, "إكسسوارات")
PERF, BEAUTY = (92, "عطور"), (11, "العناية الشخصية")
WATER, DRINK = (10, "مياه"), (1, "مشروبات")
MEN, TSHIRT = (8, "ملابس رجالي"), (80, "تيشرتات")
CATALOG = [
    prod(1, "Nestlé Pure Life 330ml", 120, "Nestlé", WATER, DRINK),
    prod(2, "Nestlé Test Product", 10, "Nestlé", WATER, DRINK),
    prod(3, "Cola 2L", 40, "Cola Co", WATER, DRINK, before=50, discount=20),
    prod(10, "حليب جهينة كامل الدسم 1 لتر", 60, "Juhayna", MILKSUB, MILK),
    prod(11, "Dina Farms Natural Rayeb Milk 250ml", 35, "Dina Farms", MILKSUB, MILK),
    prod(12, "حليب لوزين خالي اللاكتوز", 75, "Lozine", MILKSUB, MILK),
    prod(13, "OGX Coconut Milk Shampoo 385ml", 320, "OGX", SHAMPOO, HAIR),
    prod(14, "شامبو Pantene 400ml", 200, "Pantene", SHAMPOO, HAIR),
    prod(20, "iPhone 15 128GB", 35000, "Apple", PHONES, ELEC, images=["https://cdn.tix/i15.jpg"], video_url="https://cdn.tix/i15.mp4"),
    prod(21, "iPhone 11 64GB", 14500, "Apple", PHONES, ELEC),
    prod(22, "iPhone 12 128GB", 14900, "Apple", PHONES, ELEC, before=16500, discount=10),
    prod(23, "Samsung Galaxy A15", 8000, "Samsung", PHONES, ELEC),
    prod(24, "Xiaomi Redmi 13", 6000, "Xiaomi", PHONES, ELEC),
    prod(25, "Samsung Galaxy S24", 28000, "Samsung", PHONES, ELEC),
    prod(30, "ساعة يد رجالي كلاسيك", 950, "Casio", WATCH, ACC),
    prod(31, "عطر رجالي Bleu", 1800, "Chanel", PERF, BEAUTY),
    prod(32, "محفظة جلد رجالي", 450, "Zara", (93, "محافظ"), ACC),
    prod(40, "تيشيرت قطن أسود", 250, "Zara", TSHIRT, MEN),
]


def page(items, last=1):
    return {"data": items, "meta": {"current_page": 1, "last_page": last}}


class FakeClient:
    """Mimics TixClient. Like the real API (observed) its search ignores category and price."""

    def __init__(self, catalog=None, fail=False, search_fail=False, live_extra=None, live_fail=False):
        self.catalog = CATALOG if catalog is None else catalog
        self.fail, self.search_fail = fail, search_fail
        self.live_extra = live_extra or []   # products the live endpoints return but the cached snapshot (products()) lacks
        self.live_fail = live_fail
        self.calls: list[tuple] = []

    def _chk(self, *call, search=False):
        from tix.client import TixAPIError
        self.calls.append(call)
        if self.fail or (search and self.search_fail):
            raise TixAPIError("boom")

    def products(self, page_no=1): self._chk("products", page_no); return page(self.catalog if page_no == 1 else [])
    def filter(self, page=1, **params): self._chk("filter", params); return page_(self.catalog)
    def subcategory_products(self, sid, page_no=1):
        self._chk("subcategory_products", sid)
        if self.live_fail:
            from tix.client import TixAPIError
            raise TixAPIError("live down")
        rows = [p for p in self.catalog + self.live_extra if (p.get("subcategory") or {}).get("id") == sid]
        return page(rows if page_no == 1 else [])

    def discounted(self, page_no=1): self._chk("discounted"); return page([p for p in self.catalog if p["price_before"] != p["price_after"]])
    def categories(self): self._chk("categories"); return page([{"id": 1, "name": "مشروبات"}])
    def search(self, q):
        self._chk("search", q, search=True)
        n = guards.normalize(q)
        return page([p for p in self.catalog if n in guards.normalize(p["name"] + " " + (p["brand"] or {}).get("name", ""))])

    def product(self, pid):
        self._chk("product", pid)
        for p in self.catalog:
            if p["id"] == pid:
                return {"data": {**p, "long_description": "<b>وصف</b> طويل", "quantity": 12 if pid != 24 else 0,
                                 "vendor": {"name": "التوحيد ماركت"}, "is_returnable": 1,
                                 "features": [{"name": "ضمان سنة"}], "reviews": [{"rating": 4}, {"rating": 5}]}}
        from tix.client import TixAPIError
        raise TixAPIError("HTTP 404", 404)


page_ = page  # alias used inside FakeClient.filter (its `page` parameter shadows the function)


# ----------------------------------------------------------------------------------------------- scripted model
def say(text):
    return {"content": text, "tool_calls": []}


def call(name, **args):
    return {"content": "", "tool_calls": [{"id": f"c{abs(hash((name, json.dumps(args, sort_keys=True)))) % 10**6}",
                                           "name": name, "arguments": json.dumps(args, ensure_ascii=False)}]}


class ScriptedLLM:
    """Pops one scripted reply per LLM call (a dict, or a callable(messages) -> dict). Records every request."""

    def __init__(self, *replies, fail=False):
        self.replies, self.fail = list(replies), fail
        self.requests: list[list[dict]] = []
        self.tools_offered: list = []

    def chat(self, messages, tools=None):
        from agent.llm import LLMError
        self.requests.append(json.loads(json.dumps(messages, default=str)))
        self.tools_offered.append(tools)
        if self.fail:
            raise LLMError("down")
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies (the agent made an unexpected extra LLM call)")
        r = self.replies.pop(0)
        return r(messages) if callable(r) else r

    @property
    def calls(self):
        return len(self.requests)


def show(text, *ids):
    """A final reply that asks for product cards the way the real prompt instructs: a trailing [[show: ...]] line."""
    return say(f"{text}\n[[show: {', '.join(map(str, ids))}]]")


def tool_results(messages):
    """Parsed JSON of every tool message in a request."""
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
