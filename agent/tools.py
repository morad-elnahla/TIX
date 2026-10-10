"""The agent's tools: thin, honest wrappers around the real TIX API (plus the cached catalogue snapshot).

Every tool returns a small JSON-able dict and NEVER raises: failures come back as {"error": ...} so the model can
explain them to the customer. Nothing here invents data; products the model may put on screen are exactly the ones
a tool has returned during this conversation (Session.seen)."""
from __future__ import annotations

import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor

import config
from agent import guards, render
from agent.state import Session
from tix.catalog import Catalog
from tix.client import TixAPIError, TixClient
from tix.schemas import Product, clean_html, extract_items, extract_last_page, label, media_urls, parse_price, video_urls

log = logging.getLogger("tix.tools")

MIN_SCORE = 0.5  # a candidate must match at least half of a query's words

SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_products",
        "description": "Search the real TIX catalogue (find, browse, check availability/prices, get recommendations). Returns real products "
                       "(id, name, brand, price, category path), which brands/price range match, and which categories the matches fall in, "
                       "so you can tell the product the customer means from look-alikes and re-search with `category`.",
        "parameters": {"type": "object", "properties": {
            "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 4,
                        "description": "1-4 phrasings of the wanted product as the STORE might name it, Arabic and/or English. Omit to browse a category."},
            "category": {"type": "string", "description": "Restrict to a category/subcategory (words from a returned category path)."},
            "brand": {"type": "string", "description": "Only this brand (one spelling; retry another if nothing matches)."},
            "min_price": {"type": "number"}, "max_price": {"type": "number", "description": "EGP, e.g. 15 ألف -> 15000."},
            "sort": {"type": "string", "enum": ["relevance", "price_low", "price_high", "newest"]},
            "only_discounted": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 15}}}}},
    {"type": "function", "function": {
        "name": "get_product_details",
        "description": "Real details of ONE product: description, features, stock, rating, return info, whether it has images/video.",
        "parameters": {"type": "object", "properties": {"product_id": {"type": "integer"}}, "required": ["product_id"]}}},
    {"type": "function", "function": {
        "name": "compare_products",
        "description": "Real details of 2-4 products side by side.",
        "parameters": {"type": "object", "properties": {
            "product_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 4}}, "required": ["product_ids"]}}},
    {"type": "function", "function": {
        "name": "list_categories",
        "description": "The store's real category tree with counts. For vague requests, or when a search found nothing.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _ids(v) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else [v]:
        n = guards.to_number(x)
        if n is not None and int(n) == n:
            out.append(int(n))
    return list(dict.fromkeys(out))


def _compact(p: Product) -> dict:
    d = {"id": p.id, "name": p.name[:90], "price": p.price}
    if p.brand:
        d["brand"] = p.brand
    if p.has_discount and p.price_before and p.price and p.price_before > p.price:
        d["was"] = p.price_before
    if p.path:
        d["path"] = p.path
    return d


class TixTools:
    def __init__(self, client: TixClient | None = None, catalog: Catalog | None = None):
        self.client = client or TixClient()
        self.catalog = catalog or Catalog(self.client)

    def warmup(self, block: bool = False) -> None:
        self.catalog.ensure(block=block)

    # ------------------------------------------------------------------ dispatch
    def run(self, name: str, args: dict, session: Session) -> dict:
        fn = {"search_products": self.search_products, "get_product_details": self.get_product_details,
              "compare_products": self.compare_products, "list_categories": self.list_categories}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(args if isinstance(args, dict) else {}, session)
        except TixAPIError as e:
            log.warning("tool %s: API error %s", name, e)
            return {"error": "store_unavailable", "detail": "The TIX store could not be reached right now."}
        except Exception:  # noqa: BLE001 - a tool bug must not crash the conversation
            log.exception("tool %s failed", name)
            return {"error": "tool_failed"}

    # ------------------------------------------------------------------ search
    def _api_search(self, queries: list[str]) -> tuple[list[Product], bool]:
        """Live search endpoint, one call per phrasing in parallel. Returns (products, any_call_succeeded)."""
        def one(q):
            try:
                return [Product.from_raw(r) for r in extract_items(self.client.search(q))]
            except TixAPIError:
                return None
        with ThreadPoolExecutor(max_workers=len(queries)) as ex:
            results = list(ex.map(one, queries))
        ok = any(r is not None for r in results)
        return [p for r in results if r for p in r if p.name], ok

    def _api_discounted(self) -> list[Product]:
        out: list[Product] = []
        try:
            first = self.client.discounted(1)
            out += [Product.from_raw(r) for r in extract_items(first)]
            for page in range(2, min(extract_last_page(first), config.MAX_PAGES) + 1):
                out += [Product.from_raw(r) for r in extract_items(self.client.discounted(page))]
        except TixAPIError:
            pass
        return [p for p in out if p.name]

    def search_products(self, a: dict, s: Session) -> dict:
        queries = [q.strip() for q in (a.get("queries") or []) if isinstance(q, str) and q.strip()][:4]
        if not queries and isinstance(a.get("query"), str) and a["query"].strip():  # tolerate a singular `query`
            queries = [a["query"].strip()]
        category, brand = str(a.get("category") or "").strip(), str(a.get("brand") or "").strip()
        lo, hi = guards.to_number(a.get("min_price")), guards.to_number(a.get("max_price"))
        sort = a.get("sort") if a.get("sort") in ("price_low", "price_high", "newest") else None
        disc = bool(a.get("only_discounted"))
        limit = int(min(max(guards.to_number(a.get("limit")) or config.SEARCH_LIMIT, 1), 15))
        if not (queries or category or brand or disc or sort):  # a bare sort = "cheapest / most expensive / newest in the whole store"
            return {"error": "need_more_info", "hint": "Ask the customer what they are looking for."}

        cat_ok = self.catalog.ensure(block=True)
        scored: dict = {}  # key -> (score, Product)
        weak: dict = {}    # key -> (score, Product): partial matches, only used when nothing better exists
        api_ok = False
        if queries:
            if cat_ok:
                for p in self.catalog.products:
                    sc = self.catalog.score(queries, p)
                    if sc >= MIN_SCORE:
                        scored[Catalog._key(p)] = (sc, p)
                    elif sc > 0:
                        weak[Catalog._key(p)] = (sc, p)
            live, api_ok = self._api_search(queries)
            by_id = {p.id: p for p in self.catalog.products} if cat_ok else {}
            for p in live:
                k = Catalog._key(p)
                if k not in scored:
                    p = by_id.get(p.id, p)  # prefer the snapshot copy: it carries the full category path
                    scored[k] = (max(self.catalog.score(queries, p), MIN_SCORE) if cat_ok else 0.6, p)
        else:  # browsing: a category, a brand, discounts, or "what's new"
            pool: list[Product] = []
            if disc:
                pool = self._api_discounted()
                api_ok = bool(pool)
            if cat_ok:
                pool += self.catalog.products
            elif not pool:
                pool, api_ok = self._browse_without_snapshot(lo, hi)
            for p in pool:
                scored.setdefault(Catalog._key(p), (1.0, p))
        if not scored and not (cat_ok or api_ok):
            return {"error": "store_unavailable", "detail": "The TIX store could not be reached right now."}

        paths: set[str] = set()
        live_ids: set = set()
        if category and cat_ok:
            paths = self.catalog.path_matches(category)
            if not paths:
                return {"error": "unknown_category", "hint": "No category matches that. Pick from these or use list_categories.",
                        "categories": [r["category"] for r in self.catalog.tree(0)][:20]}
            # The cached snapshot can be stale or partial, so a category is also read LIVE from the API: products the
            # snapshot lacks (new, or on a page that failed to download) are added if they fit the request's words.
            for p in self._live_category_products(paths):
                k = Catalog._key(p)
                if k in scored:
                    continue
                sc = self.catalog.score(queries, p) if queries else 1.0
                if sc >= MIN_SCORE:
                    scored[k] = (sc, p)
                    live_ids.add(k)
        stages = {"matched": len(scored)}
        items = list(scored.values())
        items = [(sc, p) for sc, p in items if p in guards.hide_test_products([p], s.user_text)]
        pool_n = len(items)
        stages["after_test_filter"] = pool_n
        outside: dict[str, int] = {}
        if paths:
            kept = []
            for sc, p in items:
                # A product filed only at the parent level (its deepest category is not in the payload) is kept when the customer
                # narrows to a child of that parent, e.g. an Android phone whose list entry lacks "Android phones". Products that
                # DO name a different deepest category are excluded; the model sees every kept product's real path.
                ok = p.path in paths or (bool(p.path) and not p.subsubcategory and any(q.startswith(p.path + " > ") for q in paths))
                if ok:
                    kept.append((sc, p))
                else:
                    outside[p.path or "—"] = outside.get(p.path or "—", 0) + 1
            items = kept
        stages["after_category"] = len(items)
        if disc:
            items = [(sc, p) for sc, p in items if p.has_discount]
        by_brand = items
        if brand:
            items = [(sc, p) for sc, p in items if guards.brand_matches(p, brand)]
        by_budget = items
        items = [(sc, p) for sc, p in items if guards.within_budget(p, lo, hi)]
        stages["after_brand"], stages["after_budget"] = len(by_budget), len(items)
        log.info("search %s -> %s%s", {k: v for k, v in {"q": queries, "cat": category, "brand": brand, "min": lo, "max": hi,
                                                         "disc": disc or None, "sort": sort}.items() if v},
                 stages, f" live-added={len(live_ids)}" if live_ids else "")

        # dedupe, order
        uniq: dict = {}
        for sc, p in sorted(items, key=lambda t: -t[0]):
            uniq.setdefault(Catalog._key(p), (sc, p))
        items = list(uniq.values())
        products = [p for _, p in items]
        s.last_search = {k: v for k, v in {"queries": queries, "category": category, "brand": brand, "min_price": lo,
                                           "max_price": hi, "sort": sort, "only_discounted": disc or None}.items() if v}
        for v in (lo, hi):
            if v is not None:
                s.allowed_prices.add(float(v))

        out: dict = {"total": len(products)}
        if outside:  # matches that the category filter excluded: the model must not claim "that's all we have" blindly
            out["outside_category"] = [{"path": k, "count": v} for k, v in sorted(outside.items(), key=lambda kv: -kv[1])[:4]]
        if not products:
            out["products"] = []
            # Never answer "nothing" from strict word matching alone: offer the weaker partial matches that pass every filter
            # (category, brand, budget, discounts) so the model can judge them by category path.
            near = [(sc, p) for sc, p in weak.values()
                    if p in guards.hide_test_products([p], s.user_text)
                    and (not paths or p.path in paths) and (not disc or p.has_discount)
                    and (not brand or guards.brand_matches(p, brand)) and guards.within_budget(p, lo, hi)]
            if near:
                near = [p for _, p in sorted(near, key=lambda t: -t[0])[:5]]
                s.remember(near)
                out["near_matches"] = [_compact(p) for p in near]
            if category and cat_ok:  # tell the model what DOES exist there, so it can answer "no X, but we have Y" truthfully
                pool_cat = [p for p in guards.hide_test_products(self.catalog.products, s.user_text) if p.path in self.catalog.path_matches(category)]
                cb: dict[str, int] = {}
                for p in pool_cat:
                    if p.brand:
                        cb[p.brand] = cb.get(p.brand, 0) + 1
                cp = [p.price for p in pool_cat if p.price is not None]
                if pool_cat:
                    out["in_category"] = {"products": len(pool_cat), "brands": dict(sorted(cb.items(), key=lambda kv: -kv[1])[:8]),
                                          **({"price_range": [min(cp), max(cp)]} if cp else {})}
            if by_budget and not items:  # matches exist but none inside the budget
                prices = sorted(p.price for _, p in by_budget if p.price is not None)
                out["note"] = (f"{len(by_budget)} matching products exist but none within the price range; "
                               f"cheapest {prices[0]:g}, most expensive {prices[-1]:g}.") if prices else "none within the price range"
                s.allowed_prices.update(prices[:1] + prices[-1:])
            elif brand and by_brand and not by_budget:  # the brand filter removed everything
                brands = sorted({p.brand for _, p in by_brand if p.brand})[:12]
                out["note"] = f"No matching product of brand '{brand}'. Brands that do match: {', '.join(brands) or 'none'}."
            elif not pool_n:
                out["note"] = "Nothing in the store matches. Try other phrasings, or list_categories to see what exists."
            else:
                out["note"] = "Nothing matches these filters together."
            if near:
                out["note"] += " near_matches lists weaker partial matches that fit every filter: judge them by name and category path."
            return out

        ordered = guards.sort_products(products, sort) if sort else self._diversify(items)
        shown = ordered[:limit]
        s.remember(shown)
        # what this turn found, so an LLM failure later in the turn cannot throw valid results away (see SalesAgent)
        s.found = (shown, bool(category) or len({p.path for p in products}) == 1)
        out["products"] = [_compact(p) for p in shown]
        brands: dict[str, int] = {}
        for p in products:
            if p.brand:
                brands[p.brand] = brands.get(p.brand, 0) + 1
        if brands:
            out["brands"] = dict(sorted(brands.items(), key=lambda kv: -kv[1])[:8])
        prices = [p.price for p in products if p.price is not None]
        if prices:
            out["price_range"] = [min(prices), max(prices)]
            s.allowed_prices.update({min(prices), max(prices)})
        by_path: dict[str, int] = {}
        for p in products:
            by_path[p.path or "—"] = by_path.get(p.path or "—", 0) + 1
        if len(by_path) > 1:
            out["categories"] = [{"path": k, "count": v} for k, v in sorted(by_path.items(), key=lambda kv: -kv[1])[:6]]
        if len(products) > len(shown):
            out["note"] = f"{len(products) - len(shown)} more matches not listed; narrow with category/brand/price."
        return out

    @staticmethod
    def _diversify(items: list[tuple[float, Product]]) -> list[Product]:
        """Round-robin over category paths (each ordered by score) so look-alike names in another category cannot
        crowd the right category out of the short list the model sees."""
        groups: dict[str, list[Product]] = {}
        order: list[str] = []
        for _, p in items:
            k = p.path or "—"
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(p)
        out: list[Product] = []
        while any(groups.values()):
            for k in order:
                if groups[k]:
                    out.append(groups[k].pop(0))
        return out

    def _live_category_products(self, paths: set[str]) -> list[Product]:
        """Live listing (GET /api/subcategory/{id}/products) of the subcategories behind `paths`. Missing names are filled from
        the snapshot so the products have a category path. Never raises: on API trouble the snapshot alone is used."""
        subs: dict[int, tuple[str, str]] = {}
        for p in self.catalog.products:
            if p.path in paths and p.subcategory_id is not None:
                subs.setdefault(p.subcategory_id, (p.category, p.subcategory))

        def fetch(sid: int) -> list[Product]:
            rows = []
            try:
                first = self.client.subcategory_products(sid)
                rows += extract_items(first)
                for page in range(2, min(extract_last_page(first), config.MAX_PAGES) + 1):
                    rows += extract_items(self.client.subcategory_products(sid, page))
            except TixAPIError:
                log.info("live listing of subcategory %s unavailable", sid)
            cat, sub = subs[sid]
            out = []
            for raw in rows:
                p = Product.from_raw(raw)
                if p.name:
                    out.append(dataclasses.replace(p, category=p.category or cat, subcategory=p.subcategory or sub))
            return out

        sids = list(subs)[:3]
        if not sids:
            return []
        with ThreadPoolExecutor(max_workers=len(sids)) as ex:
            return [p for batch in ex.map(fetch, sids) for p in batch]

    def _browse_without_snapshot(self, lo, hi) -> tuple[list[Product], bool]:
        out: list[Product] = []
        try:
            params = {"price_min": lo, "price_max": hi}  # parameter names from the Postman collection
            first = self.client.filter(1, **params)
            out += [Product.from_raw(r) for r in extract_items(first)]
            for page in range(2, min(extract_last_page(first), config.MAX_PAGES) + 1):
                out += [Product.from_raw(r) for r in extract_items(self.client.filter(page, **params))]
            return [p for p in out if p.name], True
        except TixAPIError:
            return [p for p in out if p.name], bool(out)

    # ------------------------------------------------------------------ details / compare
    def _details(self, pid: int, s: Session) -> dict | None:
        try:
            payload = self.client.product(pid)
        except TixAPIError as e:
            if e.status == 404:
                return None
            raise
        raw = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else payload
        if not isinstance(raw, dict) or not raw.get("name"):
            return None
        p = Product.from_raw({**raw, "id": raw.get("id", pid)})
        s.remember([p])
        images, videos = media_urls(raw.get("images")) or list(p.images), video_urls(raw)
        s.media[pid] = {"images": images, "videos": videos}
        d = _compact(p)
        d["description"] = (clean_html(raw.get("short_description")) + " " + clean_html(raw.get("long_description") or raw.get("description")))[:500].strip()
        feats = [label(f) for f in (raw.get("features") or []) if label(f)][:8]
        if feats:
            d["features"] = feats
        if raw.get("quantity") is not None:
            q = parse_price(raw.get("quantity"))
            d["in_stock"] = bool(q and q > 0)
            d["quantity"] = int(q) if q is not None else None
        if label(raw.get("vendor")):
            d["vendor"] = label(raw["vendor"])
        for k in ("is_returnable", "max_order_quantity"):
            if raw.get(k) is not None:
                d[k] = raw[k]
        ratings = [parse_price(r.get("rating")) for r in raw.get("reviews") or [] if isinstance(r, dict)]
        ratings = [r for r in ratings if r is not None]
        if ratings:
            d["rating"] = round(sum(ratings) / len(ratings), 1)
            d["reviews_count"] = len(ratings)
        d["images"], d["has_video"] = len(images), bool(videos)
        return d

    def get_product_details(self, a: dict, s: Session) -> dict:
        ids = _ids(a.get("product_id"))
        if not ids:
            return {"error": "product_id_required"}
        d = self._details(ids[0], s)
        return d if d else {"error": "product_not_found"}

    def compare_products(self, a: dict, s: Session) -> dict:
        ids = _ids(a.get("product_ids"))[:4]
        if len(ids) < 2:
            return {"error": "need_two_or_more_product_ids"}
        with ThreadPoolExecutor(max_workers=len(ids)) as ex:
            futs = [ex.submit(self._details, i, s) for i in ids]
            rows, missing = [], []
            for i, f in zip(ids, futs):
                try:
                    d = f.result()
                except TixAPIError:
                    d = None
                (rows.append({**d, "description": d["description"][:200]}) if d else missing.append(i))
        out: dict = {"products": rows}
        if missing:
            out["unavailable_ids"] = missing
        return out

    # ------------------------------------------------------------------ categories
    def list_categories(self, a: dict, s: Session) -> dict:
        if self.catalog.ensure(block=True):
            return {"categories": self.catalog.tree()}
        payload = self.client.categories()
        names = [label(c.get("name")) if isinstance(c, dict) else str(c) for c in extract_items(payload)]
        return {"categories": [{"category": n} for n in names if n]}