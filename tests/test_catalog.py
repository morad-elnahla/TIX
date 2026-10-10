import time

from tix.catalog import Catalog
from tix.schemas import Product
from tests.fakes import CATALOG, FakeClient, page


def loaded():
    c = Catalog(FakeClient())
    assert c.load()
    return c


def test_loads_all_pages_and_survives_a_failed_page():
    class Paged(FakeClient):
        def products(self, n=1):
            self.calls.append(("p", n))
            if n == 2:
                from tix.client import TixAPIError
                raise TixAPIError("x")
            return page(CATALOG[:5] if n == 1 else CATALOG[5:8], last=3)
    c = Catalog(Paged())
    assert c.load() and len(c.products) == 8


def test_failed_first_load_is_reported_and_retried_later():
    c = Catalog(FakeClient(fail=True))
    assert c.ensure() is False and c.state == "failed"


def test_stale_snapshot_keeps_serving_while_refreshing_in_background():
    c = Catalog(FakeClient(), ttl=0)
    c.load()
    before = len(c.products)
    time.sleep(0.01)
    assert c.ensure(block=False) is True and len(c.products) == before


def test_score_uses_name_brand_and_category_path_with_light_morphology():
    c = loaded()
    p = {x.id: x for x in c.products}
    assert c.score(["موبايلات"], p[20]) == 1.0           # plural query vs path "موبايلات"/"موبايل"
    assert c.score(["الساعات"], p[30]) == 1.0            # definite article
    assert c.score(["milk"], p[13]) == 1.0 and c.score(["milk"], p[10]) == 0.0  # lexical alone cannot tell milk from shampoo...
    assert c.score(["حليب"], p[10]) == 1.0               # ...which is why the model gets the category breakdown
    assert c.score(["غسالة"], p[20]) == 0.0


def test_tree_counts_come_from_real_products():
    t = {r["category"]: r for r in loaded().tree()}
    assert t["الإلكترونيات"]["count"] == 6 and t["ألبان وأجبان"]["subcategories"] == [{"name": "حليب", "count": 3}]


def test_product_media_parsing_is_tolerant():
    p = Product.from_raw({"id": 1, "name": "x", "price_after": 5, "images": [{"url": "a.jpg"}, "https://c/d.png"], "promo_video": {"url": "v.mp4"}})
    assert p.image.endswith("/a.jpg") and p.images[1] == "https://c/d.png" and p.videos[0].endswith("/v.mp4")
    assert Product.from_raw({"id": 2, "name": "y", "price_after": 1}).image == ""


def test_spelling_variants_match_without_a_dictionary():
    c = loaded()
    p = {x.id: x for x in c.products}
    # the catalogue spells تيشرت; customers (and models) also write تيشيرت / اسود for أسود
    assert c.score(["تيشيرت"], p[40]) == 1.0 and c.score(["اسود"], p[40]) >= 0.5
    assert c.path_matches("تيشيرتات") == {"ملابس رجالي > تيشرتات"}
    assert c.score(["حبيب"], p[10]) == 0.0  # short words are never fuzzed


def test_albaan_is_not_reduced_to_a_prefix_that_matches_pants():
    from tix.catalog import _hit, _variants
    from agent.guards import tokens
    assert not _hit(set(tokens("بانطلون جينز")), _variants("البان"))
    assert _hit(set(tokens("ألبان وأجبان حليب")), _variants("البان"))
