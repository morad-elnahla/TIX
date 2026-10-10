"""LIVE smoke tests. Skipped unless you run:   TIX_LIVE=1 pytest tests/test_live_smoke.py -v -s
Add GROQ_API_KEY to also exercise the real model. Nothing is mocked and no product id/brand is hardcoded:
everything is taken from the CURRENT catalogue."""
import os
import re

import pytest

from agent.agent import SalesAgent
from agent.state import Session
from agent.tools import TixTools
from tix.client import TixClient
from tix.schemas import Product, extract_items, extract_last_page

pytestmark = pytest.mark.skipif(os.getenv("TIX_LIVE") != "1", reason="set TIX_LIVE=1 to hit the real API")


@pytest.fixture(scope="module")
def tools():
    t = TixTools()
    assert t.catalog.load(), "catalogue could not be loaded from GET /api/products"
    return t


def test_products_page_shape_and_media_fields():
    payload = TixClient().products()
    items = extract_items(payload)
    assert items, "no product list found in /api/products: adjust tix/schemas.extract_items"
    print("keys:", sorted(items[0]), "| last_page:", extract_last_page(payload))
    p = Product.from_raw(items[0])
    print("first image:", p.image or "(none parsed)", "| raw images field:", items[0].get("images"))


def test_catalogue_snapshot(tools):
    prods = tools.catalog.products
    print(len(prods), "products;", len({p.path for p in prods}), "category paths")
    assert len(prods) > 10 and all(p.name for p in prods)


def test_search_word_from_a_real_product_finds_it(tools):
    p = next(x for x in tools.catalog.products if x.id and re.search(r"[^\W\d_]{4,}", x.name))
    word = re.search(r"[^\W\d_]{4,}", p.name).group()
    r = tools.run("search_products", {"queries": [word], "limit": 15}, Session())
    print(word, "->", [x["name"] for x in r.get("products", [])])
    assert r.get("total", 0) >= 1


def test_details_and_media_of_a_real_product(tools):
    p = tools.catalog.products[0]
    s = Session()
    d = tools.run("get_product_details", {"product_id": p.id}, s)
    print({k: d[k] for k in ("name", "price", "in_stock", "images", "has_video") if k in d}, s.media.get(p.id))
    assert d.get("name")


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="needs GROQ_API_KEY")
def test_real_model_small_talk_does_not_search_and_a_request_shows_real_products(tools):
    agent, sess = SalesAgent(tools), Session()
    calls = []
    real = tools.run
    tools.run = lambda n, a, s: (calls.append(n), real(n, a, s))[1]
    try:
        print(agent.run("إزيك؟", sess))
        assert calls == []
        word = re.search(r"[^\W\d_]{4,}", tools.catalog.products[0].name).group()
        print(agent.run(f"عندكم {word}؟", sess))
        assert "search_products" in calls
        assert all(c["id"] in {p.id for p in sess.seen.values()} for c in sess.messages[-1].get("products", []))
    finally:
        tools.run = real
