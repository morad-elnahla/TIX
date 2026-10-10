"""Regressions from the hard manual test: categories filed one level up, 'cheapest in the store', placeholder brands, wrong brand fields,
and the output validator rejecting legitimate product lists. Each is tested on several product types, not only the one that exposed it."""
import pytest

from agent import guards
from agent.prompts import SYSTEM
from agent.state import Session
from agent.tools import TixTools
from tests.fakes import CATALOG, FakeClient, prod
from tix.schemas import Product

SUB, CAT = (15, "الموبايلات والتابلت وإكسسواراتها"), (11, "الإلكترونيات")
KSUB, KCAT = (200, "أدوات المطبخ"), (20, "المنزل والمطبخ")
EXTRA = [
    prod(300, "Xiaomi Redmi Note 14", 10864, "Xiaomi", SUB, CAT, subsubcategories=[{"id": 2, "name": "موبايلات أندرويد"}]),
    prod(301, "Xiaomi Redmi A5", 4500, "Xiaomi", SUB, CAT),                                   # list payload names no deepest level
    prod(302, "iPhone 13", 20000, "Apple", SUB, CAT, subsubcategories=[{"id": 3, "name": "آيفون"}]),
    prod(310, "Tefal Pan 28", 1200, "Tefal", KSUB, KCAT, subsubcategories=[{"id": 30, "name": "مقالي"}]),
    prod(311, "Generic Spatula", 40, "Generic", KSUB, KCAT),                                  # filed one level up
    prod(312, "Cheap Whisk", 15, "غير مذكور", KSUB, KCAT),
    prod(313, "New Balance 574 Sneakers", 2100, "Adidas", (60, "أحذية رياضية"), (6, "الأحذية")),   # wrong brand field in the store
]


def tools():
    return TixTools(FakeClient(CATALOG + EXTRA))


def ids(r):
    return {p["id"] for p in r["products"]}


def test_a_product_filed_one_level_up_is_kept_when_the_customer_narrows_to_a_child_category():
    r = tools().run("search_products", {"queries": ["موبايل"], "category": "موبايلات أندرويد", "max_price": 5000}, Session())
    assert ids(r) == {301}                                   # the Android phone under 5000 whose entry lacks "Android"
    r = tools().run("search_products", {"queries": ["موبايل"], "category": "موبايلات أندرويد"}, Session())
    assert ids(r) == {300, 301}                              # the iPhone (a different deepest category) stays out


def test_same_rule_in_an_unrelated_category():
    r = tools().run("search_products", {"queries": ["مطبخ"], "category": "مقالي", "max_price": 100}, Session())
    assert ids(r) == {311, 312}
    assert 310 not in ids(tools().run("search_products", {"queries": ["مطبخ"], "category": "مقالي", "max_price": 100}, Session()))


def test_cheapest_and_most_expensive_in_the_whole_store_need_no_other_filter():
    t, s = tools(), Session()
    cheap = t.run("search_products", {"sort": "price_low", "limit": 3}, s)
    assert cheap["products"][0]["id"] == 312 and cheap["products"][0]["price"] == 15 and "error" not in cheap
    top = t.run("search_products", {"sort": "price_high", "limit": 1}, s)
    assert top["products"][0]["price"] == 35000


def test_the_placeholder_brand_means_no_brand_and_is_not_shown_on_cards():
    from agent import render
    p = Product.from_raw(prod(1, "تيشرت", 450, "غير مذكور"))
    assert p.brand == "" and render.card(p)["brand"] == ""
    assert Product.from_raw(prod(2, "x", 5, "Zara")).brand == "Zara"


def test_brand_search_also_looks_in_the_name_when_the_brand_field_is_wrong():
    r = tools().run("search_products", {"queries": ["sneakers"], "brand": "new balance"}, Session())
    assert ids(r) == {313}


REAL_REPLIES = [  # copied from the manual test: legitimate, and must reach the customer
    "عندنا موبايلات شاومي:\n• Xiaomi Redmi Note 14 – 8 GB رام / 256 GB تخزين – 10864 جنيه\n• Xiaomi Redmi A5 – 4 GB رام / 64 GB تخزين – 4500 جنيه\n\nتحب أشوف تفاصيل أكتر أو أضيف أي حاجة؟",
    "أه، في حاجات بين 200 و 500 جنيه:\n\n1. كابل Anker 322 USB‑C إلى USB‑C – 336 جنيه\n2. إضاءة PLOKAMA ML‑50 Pro – 335 جنيه\n3. حامل سيارة Earldom – 225 جنيه\n4. باور بانك Earldom 10,000 mAh – 475 جنيه\n5. كابل Apple USB‑C – 370 جنيه",
    "سنيكرز Adidas حريمي أسود وأبيض – 600 جنيه\nسنيكرز New Balance رياضي حريمي – 600 جنيه\nسنيكرز Miu Miu بنعل مرتفع حريمي – 600 جنيه\nسنيكرز Gucci مبطن حريمي – 600 جنيه\nسنيكرز Nike Air Max 97 حريمي – 729 جنيه",
    "هنا شوية تيشيرتات بأسعار مناسبة:\n\n1. تيشرت أوفر سايز رجالي قطن 80٪ بولو زيبر – 450 جنيه\n2. بلوزة حريمي بأكمام طويلة – 500 جنيه\n3. فستان قميص حريمي – 500 جنيه",
    "عندنا موديلين من الموبايلات: Redmi Note 14 بـ 10,864 جنيه و Redmi A5 بـ 4,500 جنيه، والاتنين من شاومي.",
]


@pytest.mark.parametrize("text", REAL_REPLIES)
def test_legitimate_product_lists_are_not_rejected_by_the_validator(text):
    assert guards.reply_problem(text) is None


@pytest.mark.parametrize("text", ["السعر 404 جنيه بس", "ده بـ 429 جنيه", "Samsung A404 بسعر كويس"])
def test_ordinary_numbers_that_look_like_http_codes_are_fine(text):
    assert guards.reply_problem(text) is None


@pytest.mark.parametrize("text", ["Error 429 Too Many Requests", "HTTP 404 not found", "status: 429", "حصل error 404"])
def test_real_error_wording_is_still_rejected(text):
    assert guards.reply_problem(text)


def test_prompt_carries_the_new_rules():
    for rule in ("ATTRIBUTES", "NEVER add a colour", "Loosely related", "Cheapest/most expensive", "anything", "displaying now"):
        assert rule in SYSTEM
