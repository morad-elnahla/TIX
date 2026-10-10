"""A cached copy of the real TIX catalogue (all pages of GET /api/products) plus cheap lexical matching and the
category tree. It is NOT a second product database: every field comes from the API, it is refreshed in the
background, and the live search endpoint is still queried on every search."""
from __future__ import annotations

import logging
import threading
from difflib import SequenceMatcher
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import config
from agent.guards import normalize, tokens
from tix.schemas import Product, extract_items, extract_last_page

log = logging.getLogger("tix.catalog")

_STOP = {"the", "and", "for", "with", "من", "في", "مع", "علي", "او", "عايز", "عندكم", "ال"}


_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("ات", "ان", "ين", "ون", "ية", "ها", "ه", "ي")


def stem(w: str) -> str:
    """Tiny, symmetric Arabic stemmer (article, plural/feminine endings): الموبايلات / موبايلات / موبايل -> موبايل. It is applied to BOTH
    the query word and the product's words, so morphology differences stop mattering without any dictionary. Words stay >= 3 letters."""
    if not w or not ("\u0600" <= w[0] <= "\u06ff"):
        return w[:-1] if w.isascii() and w.endswith("s") and len(w) > 4 else w
    for p in _PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 3:
            w = w[len(p):]
            break
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _variants(w: str) -> set[str]:
    """Forms of ONE query word to look for: itself, without the article, and its stem."""
    out = {w, stem(w)}
    if w.startswith("ال") and len(w) - 2 >= 4:  # "الحليب"->"حليب", but never "البان"->"بان" as a prefix match ("ألبان" is a root)
        out.add(w[2:])
    return {v for v in out if v}


def _expand(ws: set[str]) -> set[str]:
    """The words of a product/category plus their article-less forms and stems, so equality of stems is a plain set lookup."""
    out = set(ws)
    for w in ws:
        out.add(stem(w))
        if w.startswith("ال") and len(w) - 2 >= 4:
            out.add(w[2:])
    return out


def _close(a: str, b: str) -> bool:
    """Spelling variants / typos with no dictionary: تيشيرت~تيشرت, اسود~أسود, بنطلون~بنطلون. Long words only, high similarity."""
    if len(a) < 5 or len(b) < 5 or abs(len(a) - len(b)) > 2:
        return False
    m = SequenceMatcher(None, a, b)
    return m.real_quick_ratio() >= 0.86 and m.quick_ratio() >= 0.86 and m.ratio() >= 0.86


def _hit(words: set[str], variants: set[str]) -> bool:
    for v in variants:
        if v in words:
            return True
        if len(v) >= 4 and any(t.startswith(v) or (len(t) >= 4 and v.startswith(t)) for t in words):
            return True
    return any(_close(v, t) for v in variants for t in words)


class Catalog:
    def __init__(self, client, ttl: int | None = None, max_pages: int | None = None):
        self.client = client
        self.ttl = config.CATALOG_TTL if ttl is None else ttl
        self.max_pages = max_pages or config.CATALOG_MAX_PAGES
        self.products: list[Product] = []
        self.state = "idle"  # idle | ready | failed
        self._index: dict[int, tuple[set[str], set[str]]] = {}
        self._loaded_at = self._failed_at = 0.0
        self._pages: dict[int, list] = {}
        self.missing: list[int] = []   # pages that could not be downloaded (the snapshot is then INCOMPLETE)
        self.last_page = 0
        self.expected_total: int | None = None
        self._busy = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    @property
    def ready(self) -> bool:
        return self.state == "ready" and bool(self.products)

    def ensure(self, block: bool = True) -> bool:
        """Make the snapshot available. Only the very first load blocks; stale refreshes run in the background."""
        if self._busy:
            return self.ready
        now = time.time()
        if self.state == "idle" or (self.state == "failed" and now - self._failed_at > 60):
            self.load() if block else self._spawn()
        elif self.state == "ready" and now - self._loaded_at > self.ttl:
            self._spawn()
        return self.ready

    def load(self) -> bool:
        with self._lock:
            if self._busy:
                return self.ready
            self._busy = True
        self._run()
        return self.ready

    def _spawn(self) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=self._run, daemon=True, name="tix-catalog").start()

    def _fetch_pages(self, pages: list[int], workers: int) -> None:
        def one(n):
            try:
                return n, extract_items(self.client.products(n))
            except Exception as e:  # noqa: BLE001 - one missing page must not discard the rest; it is retried below
                log.warning("catalogue page %s failed: %s", n, e)
                return n, None
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for n, items in ex.map(one, pages):
                if items is not None:
                    self._pages[n] = items

    def _run(self) -> None:
        try:
            if self.missing and self._pages:  # incomplete snapshot: only fetch what is missing
                self._fetch_pages(list(self.missing), 1)
            else:
                first = self.client.products(1)
                self._pages = {1: extract_items(first)}
                self.last_page = min(extract_last_page(first), self.max_pages)
                meta = first.get("meta") if isinstance(first, dict) and isinstance(first.get("meta"), dict) else {}
                self.expected_total = int(meta["total"]) if str(meta.get("total", "")).isdigit() else None
                self._fetch_pages(list(range(2, self.last_page + 1)), config.CATALOG_WORKERS)
            for _ in range(2):  # retry what failed, gently (a burst of parallel requests is the usual cause of losses)
                todo = [n for n in range(2, self.last_page + 1) if n not in self._pages]
                if not todo:
                    break
                time.sleep(0.5)
                self._fetch_pages(todo, 1)
            self.missing = [n for n in range(1, self.last_page + 1) if n not in self._pages]
            products, seen = [], set()
            for n in sorted(self._pages):
                for raw in self._pages[n]:
                    p = Product.from_raw(raw)
                    key = p.id if p.id is not None else p.name
                    if p.name and key not in seen:
                        seen.add(key)
                        products.append(p)
            if not products:
                raise ValueError("empty catalogue")
            self._index = {self._key(p): self._words(p) for p in products}
            self.products = products
            self.state, self._loaded_at = "ready", time.time() - (self.ttl - 30 if self.missing else 0)  # incomplete: refresh in ~30s
            if self.missing:
                log.warning("catalogue INCOMPLETE: pages %s missing (%d products loaded); will retry shortly", self.missing, len(products))
            if self.expected_total and self.expected_total != len(products):
                log.warning("catalogue has %d products but the API reports %d", len(products), self.expected_total)
            log.info("catalogue loaded: %d products (%d pages)", len(products), self.last_page)
        except Exception as e:  # noqa: BLE001
            log.warning("catalogue load failed: %s", e)
            if not self.products:
                self.state, self._failed_at = "failed", time.time()
        finally:
            self._busy = False

    def status(self) -> dict:
        return {"state": self.state, "products": len(self.products), "pages": self.last_page, "missing_pages": list(self.missing),
                "api_reports_total": self.expected_total, "age_s": round(time.time() - self._loaded_at) if self._loaded_at else None}

    # ------------------------------------------------------------------ matching
    @staticmethod
    def _key(p: Product):
        return p.id if p.id is not None else p.name

    @staticmethod
    def _words(p: Product) -> tuple[set[str], set[str]]:
        strong = _expand(set(tokens(f"{p.name} {p.brand} {p.path}")))
        weak = _expand(set(tokens(f"{p.short_description} {p.keywords}"))) - strong
        return strong, weak

    def words_of(self, p: Product) -> tuple[set[str], set[str]]:
        return self._index.get(self._key(p)) or self._words(p)

    def score(self, queries: list[str], p: Product) -> float:
        """0..1: best fraction of a query's words found in the product's name/brand/category path (full weight)
        or description/keywords (half weight). The LLM supplies several phrasings, so no synonym table is needed."""
        strong, weak = self.words_of(p)
        best = 0.0
        for q in queries:
            words = [w for w in tokens(q) if w not in _STOP and len(w) >= 2]
            if not words:
                continue
            got = 0.0
            for w in words:
                v = _variants(w)
                got += 1.0 if _hit(strong, v) else 0.5 if _hit(weak, v) else 0.0
            best = max(best, got / len(words))
        return best

    def path_matches(self, text: str) -> set[str]:
        """Category/subcategory/subsubcategory paths whose names contain every word of `text`."""
        want = [_variants(w) for w in tokens(text) if w not in _STOP and len(w) >= 2]
        if not want:
            return set()
        out = set()
        for path in {p.path for p in self.products if p.path}:
            words = _expand(set(tokens(path)))
            if all(_hit(words, v) for v in want):
                out.add(path)
        return out

    def tree(self, per_category: int = 10) -> list[dict]:
        """The real category tree with product counts (subcategories trimmed to the biggest ones)."""
        cats: dict[str, Counter] = {}
        for p in self.products:
            cats.setdefault(p.category or "—", Counter())[p.subcategory or ""] += 1
        rows = []
        for name, subs in sorted(cats.items(), key=lambda kv: -sum(kv[1].values())):
            top = [{"name": s, "count": n} for s, n in subs.most_common(per_category) if s]
            rows.append({"category": name, "count": sum(subs.values()), "subcategories": top})
        return rows
