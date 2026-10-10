"""The 'phones under 5000' miss was a GENERAL weakness (a product whose own name lacks the category word, found only through its
category path in a different Arabic form). These tests use other product types on purpose: the fix must not be phone-specific."""
import pytest

from agent.state import Session
from agent.tools import TixTools
from tix.catalog import Catalog, stem
from tests.fakes import CATALOG, FakeClient, prod

KITCHEN = (200, "أدوات المطبخ"), (20, "المنزل والمطبخ")
EXTRA = [
    prod(100, "Tefal Ingenio Frying Pan 28cm", 1200, "Tefal", *KITCHEN[:1], KITCHEN[1]),
    prod(101, "Prestige Mixer 500W", 800, "Prestige", *KITCHEN[:1], KITCHEN[1]),
    prod(102, "Lamp RGB Desk", 300, "Philips", (210, "إضاءة"), (21, "الأجهزة المنزلية")),
]


def tools():
    return TixTools(FakeClient(CATALOG + EXTRA))


@pytest.mark.parametrize("a,b", [("الموبايلات", "موبايل"), ("موبايلات", "الموبايل"), ("شفرات", "شفرة"), ("ساعات", "ساعة"),
                                 ("ادوات", "الادوات"), ("تيشرتات", "تيشرت")])
def test_morphological_forms_have_the_same_stem(a, b):
    from agent.guards import normalize
    assert stem(normalize(a)) == stem(normalize(b))


def test_unrelated_words_do_not_collapse_into_the_same_stem():
    from agent.guards import normalize
    assert stem(normalize("البان")) != stem(normalize("بانطلون")) and stem(normalize("حليب")) != stem(normalize("حبيب"))


@pytest.mark.parametrize("q", ["المطبخ", "مطبخ", "أدوات المطبخ", "ادوات مطبخ", "الأدوات"])
def test_products_whose_name_lacks_the_category_word_are_found_through_the_category_path(q):
    r = tools().run("search_products", {"queries": [q]}, Session())
    assert {100, 101} <= {p["id"] for p in r["products"]}


@pytest.mark.parametrize("q,budget,expected", [("مطبخ", 1000, {101}), ("ساعة", 1000, {30}), ("الإضاءة", 500, {102})])
def test_budget_search_in_any_category_finds_the_cheaper_product(q, budget, expected):
    r = tools().run("search_products", {"queries": [q], "max_price": budget}, Session())
    assert {p["id"] for p in r["products"]} == expected


def test_weak_partial_matches_are_offered_instead_of_a_bare_nothing():
    r = tools().run("search_products", {"queries": ["Tefal premium nonstick induction cookware set"]}, Session())
    assert r["total"] == 0 and r["near_matches"][0]["id"] == 100 and "near_matches" in r["note"]


def test_near_matches_still_respect_budget_and_brand():
    t, s = tools(), Session()
    q = {"queries": ["Tefal premium nonstick induction cookware set"]}
    assert "near_matches" not in t.run("search_products", {**q, "max_price": 500}, s)
    assert "near_matches" not in t.run("search_products", {**q, "brand": "Prestige"}, s)


def test_no_near_matches_when_a_real_match_exists():
    r = tools().run("search_products", {"queries": ["Tefal"]}, Session())
    assert r["total"] == 1 and "near_matches" not in r


def test_broken_plurals_are_not_stemmed_by_rules_the_model_supplies_other_phrasings_instead():
    """Honest limit: الأجهزة (plural of جهاز) is not reducible by suffix rules; the prompt makes the model search several phrasings."""
    from agent.guards import normalize
    assert stem(normalize("الأجهزة")) != stem(normalize("جهاز"))
