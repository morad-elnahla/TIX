"""Tools against the fake TIX API: filtering, category evidence, budget/brand, details/media, show_products validation."""
import pytest

from agent.state import Session
from agent.tools import TixTools
from tests.fakes import FakeClient


@pytest.fixture
def tools():
    return TixTools(FakeClient())


@pytest.fixture
def sess():
    return Session()


def names(r):
    return [p["name"] for p in r["products"]]


def test_milk_search_exposes_category_evidence_and_does_not_hide_the_look_alike(tools, sess):
    r = tools.run("search_products", {"queries": ["حليب", "milk"]}, sess)
    paths = {c["path"] for c in r["categories"]}
    assert "ألبان وأجبان > حليب" in paths and "العناية بالشعر > شامبو" in paths  # the model can SEE both meanings
    assert any("OGX" in n for n in names(r)) and any("جهينة" in n for n in names(r))  # round-robin: neither crowds out the other


def test_narrowing_by_category_removes_the_shampoo(tools, sess):
    r = tools.run("search_products", {"queries": ["milk", "حليب"], "category": "حليب"}, sess)
    assert names(r) and all("OGX" not in n for n in names(r)) and r["total"] == 3


def test_budget_is_enforced_in_code_and_sorts_cheapest_first(tools, sess):
    r = tools.run("search_products", {"queries": ["iphone", "آيفون"], "max_price": "15,000", "sort": "price_low"}, sess)
    assert names(r) == ["iPhone 11 64GB", "iPhone 12 128GB"] and all(p["price"] <= 15000 for p in r["products"])
    assert r["products"][1]["was"] == 16500


def test_no_match_inside_budget_reports_real_price_range(tools, sess):
    r = tools.run("search_products", {"queries": ["iphone"], "max_price": 1000}, sess)
    assert r["total"] == 0 and "14500" in r["note"] and "35000" in r["note"]


def test_brand_filter_and_brand_miss_lists_real_brands(tools, sess):
    r = tools.run("search_products", {"queries": ["موبايلات"], "brand": "samsung"}, sess)
    assert set(names(r)) == {"Samsung Galaxy A15", "Samsung Galaxy S24"}
    miss = tools.run("search_products", {"queries": ["موبايلات"], "brand": "nokia"}, sess)
    assert miss["total"] == 0 and "Apple" in miss["note"] and "Samsung" in miss["note"]


def test_arabic_plural_and_category_path_match_without_a_synonym_table(tools, sess):
    r = tools.run("search_products", {"queries": ["موبايلات"]}, sess)  # product names say iPhone/Galaxy; the PATH says موبايلات
    assert r["total"] == 6


def test_test_products_hidden_unless_customer_asks(tools, sess):
    sess.user_text = "عايز مياه نستله"
    assert all("Test" not in n for n in names(tools.run("search_products", {"queries": ["nestle"]}, sess)))
    sess.user_text = "show me the test product"
    assert any("Test" in n for n in names(tools.run("search_products", {"queries": ["nestle"]}, sess)))


def test_discount_browse_uses_real_discount_fields(tools, sess):
    r = tools.run("search_products", {"only_discounted": True}, sess)
    assert set(names(r)) == {"Cola 2L", "iPhone 12 128GB"}


def test_browse_a_category_without_any_query(tools, sess):
    r = tools.run("search_products", {"category": "ساعات"}, sess)
    assert names(r) == ["ساعة يد رجالي كلاسيك"]


def test_unknown_category_returns_the_real_list(tools, sess):
    r = tools.run("search_products", {"queries": ["x"], "category": "أثاث"}, sess)
    assert r["error"] == "unknown_category" and "الإلكترونيات" in r["categories"]


def test_empty_search_asks_for_more_information(tools, sess):
    assert tools.run("search_products", {}, sess)["error"] == "need_more_info"


def test_search_survives_api_search_failure_thanks_to_snapshot(sess):
    t = TixTools(FakeClient(search_fail=True))
    assert t.run("search_products", {"queries": ["حليب"]}, sess)["total"] >= 3


def test_total_api_failure_is_reported_not_raised(sess):
    r = TixTools(FakeClient(fail=True)).run("search_products", {"queries": ["حليب"]}, sess)
    assert r["error"] == "store_unavailable"


def test_search_works_without_snapshot_using_live_search_only(sess):
    class NoList(FakeClient):
        def products(self, page_no=1):
            from tix.client import TixAPIError
            raise TixAPIError("no list")
    r = TixTools(NoList()).run("search_products", {"queries": ["iphone"]}, sess)
    assert r["total"] == 3


def test_details_include_stock_media_flags_and_register_media(tools, sess):
    d = tools.run("get_product_details", {"product_id": 20}, sess)
    assert d["in_stock"] and d["quantity"] == 12 and d["has_video"] and d["rating"] == 4.5 and "ضمان" in d["features"][0]
    assert sess.media[20]["videos"] == ["https://cdn.tix/i15.mp4"]
    assert tools.run("get_product_details", {"product_id": 24}, sess)["in_stock"] is False
    assert tools.run("get_product_details", {"product_id": 999}, sess)["error"] == "product_not_found"


def test_compare_returns_each_product_and_flags_missing(tools, sess):
    r = tools.run("compare_products", {"product_ids": [21, 22, 999]}, sess)
    assert [p["id"] for p in r["products"]] == [21, 22] and r["unavailable_ids"] == [999]
    assert tools.run("compare_products", {"product_ids": [21]}, sess)["error"]


def test_list_categories_gives_the_real_tree(tools, sess):
    tree = tools.run("list_categories", {}, sess)["categories"]
    assert tree[0]["category"] == "الإلكترونيات" and {"name": "موبايلات", "count": 6} in tree[0]["subcategories"]


def test_search_reports_brands_price_range_and_remembers_what_it_found(tools, sess):
    r = tools.run("search_products", {"queries": ["موبايلات"]}, sess)
    assert r["brands"]["Apple"] == 3 and r["price_range"] == [6000, 35000]
    products, unambiguous = sess.found
    assert unambiguous and len(products) == r["total"] and 14500.0 in sess.allowed_prices


def test_found_is_flagged_ambiguous_when_matches_span_categories_unless_narrowed(tools, sess):
    tools.run("search_products", {"queries": ["milk", "حليب"]}, sess)
    assert sess.found[1] is False
    tools.run("search_products", {"queries": ["milk", "حليب"], "category": "حليب"}, sess)
    assert sess.found[1] is True


def test_albaan_does_not_match_pants_through_a_bad_prefix(sess):
    from tests.fakes import CATALOG, prod
    pants = prod(60, "بانطلون جينز رجالي", 400, "Zara", (81, "بناطيل"), (8, "ملابس رجالي"))
    t = TixTools(FakeClient(CATALOG + [pants]))
    r = t.run("search_products", {"queries": ["البان"]}, sess)
    assert all("بانطلون" not in n for n in names(r)) and {p["id"] for p in r["products"]} >= {10, 11, 12}


def test_garbage_arguments_never_raise(tools, sess):
    assert "error" in tools.run("nope", {}, sess)
    assert tools.run("search_products", {"queries": [None, 5], "max_price": "lots"}, sess)["error"] == "need_more_info"
