"""Regression for the screenshots: the store has TWO Android phones (Redmi A5 at 4,500 and Redmi Note 14 at 10,864) but the
assistant said only one phone exists and nothing is under 5,000. Whatever dropped the cheaper phone (a stale snapshot, a page that
failed to download), a category answer must come from LIVE data too, and the model must be told what a filter excluded."""
import json
import time

_real_sleep = time.sleep

import pytest

from agent import guards
from agent.agent import SalesAgent
from agent.state import Session
from agent.tools import TixTools
from tix.catalog import Catalog
from tests.fakes import CATALOG, ELEC, FakeClient, PHONES, ScriptedLLM, call, prod, say, show, tool_results, page

NOTE14 = prod(482, "Xiaomi Redmi Note 14 8 جيجابايت / 256 جيجابايت - أسود", 10864, "Xiaomi", PHONES, ELEC, before=11064, discount=2)
A5 = prod(483, "Xiaomi Redmi A5 – 64GB. 4GB رام, شاشة", 4500, "Xiaomi", PHONES, ELEC, before=4700, discount=4)
STORE = [c for c in CATALOG if c["subcategory"] is None or c["subcategory"]["id"] != 90] + [NOTE14]   # snapshot WITHOUT the A5


def tools(**kw):
    return TixTools(FakeClient(STORE, live_extra=[A5], **kw))


def test_the_cheaper_phone_missing_from_the_snapshot_is_found_through_the_live_category_listing():
    r = tools().run("search_products", {"queries": ["موبايل", "phone"], "category": "موبايلات", "max_price": 5000}, Session())
    assert [p["id"] for p in r["products"]] == [483] and r["products"][0]["price"] == 4500


def test_without_the_live_listing_the_snapshot_alone_would_have_missed_it():
    t = TixTools(FakeClient(STORE))   # no live_extra: same code path, nothing live to add
    r = t.run("search_products", {"queries": ["موبايل", "phone"], "category": "موبايلات", "max_price": 5000}, Session())
    assert r["total"] == 0 and r["in_category"]["products"] == 1


def test_browsing_the_category_returns_both_phones():
    r = tools().run("search_products", {"category": "موبايلات"}, Session())
    assert {p["id"] for p in r["products"]} == {482, 483} and r["total"] == 2


def test_live_listing_failure_falls_back_to_the_snapshot_silently():
    r = tools(live_fail=True).run("search_products", {"queries": ["xiaomi"], "category": "موبايلات"}, Session())
    assert [p["id"] for p in r["products"]] == [482]


def test_live_added_products_still_respect_the_customers_words():
    """Live category members are added only when they fit the request: a colour query must not pull in every phone."""
    r = tools().run("search_products", {"queries": ["اسود", "black"], "category": "موبايلات"}, Session())
    assert [p["id"] for p in r["products"]] == [482]


def test_matches_excluded_by_the_category_filter_are_reported_to_the_model():
    r = tools().run("search_products", {"queries": ["xiaomi", "redmi"], "category": "ساعات"}, Session())
    assert r["total"] == 0 and r["outside_category"][0]["count"] >= 1 and "موبايلات" in r["outside_category"][0]["path"]


def test_the_exact_screenshot_conversation():
    llm = ScriptedLLM(
        call("search_products", queries=["موبايل", "phone"], category="موبايلات"),
        say("عندنا موبايلين شاومي: Redmi A5 بـ 4500 جنيه، و Redmi Note 14 بـ 10864 جنيه. تحب أوريهملك؟"),
        call("search_products", queries=["موبايل", "phone"], category="موبايلات", max_price=5000),
        show("لقيتلك موبايل تحت 5000 👌", 483))
    agent, sess = SalesAgent(tools(), llm), Session()
    agent.run("هو ده بس اللي موجود؟", sess)
    assert tool_results(llm.requests[1])[0]["total"] == 2
    agent.run("شوفلي موبايل تحت ال 5000", sess)
    assert [c["id"] for c in sess.messages[-1]["products"]] == [483]


# --------------------------------------------------------------------------- snapshot completeness
def test_a_page_that_fails_to_download_is_retried_then_only_the_missing_page_is_refetched(monkeypatch):
    monkeypatch.setattr("tix.catalog.time.sleep", lambda *_: None)

    class Flaky(FakeClient):
        broken = True

        def products(self, n=1):
            self.calls.append(("products", n))
            if n == 2 and self.broken:
                from tix.client import TixAPIError
                raise TixAPIError("HTTP 429", 429)
            return page(CATALOG[:9] if n == 1 else CATALOG[9:] if n == 2 else [], last=2)
    c = Flaky()
    cat = Catalog(c)
    cat.load()
    assert cat.missing == [2] and cat.state == "ready" and len(cat.products) == 9       # usable, but flagged INCOMPLETE
    assert cat.status()["missing_pages"] == [2]
    Flaky.broken = False
    c.calls.clear()
    cat._loaded_at -= cat.ttl                      # snapshot considered stale -> refresh
    cat.ensure(block=False)
    for _ in range(100):
        if not cat._busy:
            break
        _real_sleep(0.02)
    assert cat.missing == [] and len(cat.products) == len(CATALOG) and c.calls == [("products", 2)]  # page 1 was not downloaded again


def test_client_retries_429_honouring_retry_after_then_succeeds(monkeypatch):
    from tix.client import TixClient
    sleeps, calls = [], []
    monkeypatch.setattr("tix.client.time.sleep", sleeps.append)

    class R:
        def __init__(self, code, headers=None):
            self.status_code, self.headers = code, headers or {}

        def json(self):
            return {"data": []}
    seq = [R(429, {"Retry-After": "2"}), R(200)]
    c = TixClient(base_url="http://x", retries=2)
    c._http.get = lambda *a, **k: (calls.append(1), seq.pop(0))[1]
    assert c.get("/api/products") == {"data": []} and len(calls) == 2 and sleeps == [2.0]


def test_client_does_not_retry_a_404():
    from tix.client import TixAPIError, TixClient
    calls = []

    class R:
        status_code, headers = 404, {}
    c = TixClient(base_url="http://x", retries=2)
    c._http.get = lambda *a, **k: (calls.append(1), R())[1]
    with pytest.raises(TixAPIError):
        c.get("/api/products/1")
    assert len(calls) == 1


# --------------------------------------------------------------------------- what the customer sees
def test_markdown_is_never_shown_as_asterisks():
    raw = "فيه موبايل واحد:\n\n- **Xiaomi Redmi Note 14** – 8GB\n- `سعره` 10,864 جنيه\n# عنوان"
    out = guards.plain_text(raw)
    assert "*" not in out and "`" not in out and "#" not in out and "• Xiaomi Redmi Note 14" in out


def test_discount_badge_matches_the_storefront_figure():
    from agent import render
    from tix.schemas import Product
    card = render.card(Product.from_raw(NOTE14))
    assert card["discount_pct"] == 1.81 and render.card(Product.from_raw(A5))["discount_pct"] == 4.26
