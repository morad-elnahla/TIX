"""The only place that talks HTTP to TIX."""
from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

import config
from tix import endpoints as ep

log = logging.getLogger("tix.client")


class TixAPIError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class TixClient:
    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: float | None = None, retries: int | None = None):
        self.base_url = (base_url or config.TIX_BASE_URL).rstrip("/")
        self.timeout = timeout or config.TIX_TIMEOUT
        self.retries = config.TIX_RETRIES if retries is None else retries
        self._http = requests.Session()
        self._http.headers.update({"Accept": "application/json", "Accept-Language": config.TIX_LANG})
        token = token if token is not None else config.TIX_ACCESS_TOKEN
        if token:
            self._http.headers["Authorization"] = f"Bearer {token}"
        # Shared catalog cache (categories/brands only: public data, never user data).
        self._cache: dict[str, tuple[float, Any]] = {}

    def get(self, path: str, params: dict | None = None, cache: bool | float = False) -> Any:
        """GET + JSON. `cache` is False, True (CATALOG_CACHE_TTL) or a TTL in seconds. 429 and 5xx are retried with a short,
        jittered backoff (honouring Retry-After); other 4xx are not (they will not improve)."""
        params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        key = f"{path}?{sorted(params.items())}"
        ttl = config.CATALOG_CACHE_TTL if cache is True else float(cache or 0)
        if ttl and key in self._cache and time.time() - self._cache[key][0] < ttl:
            return self._cache[key][1]
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            delay = 0.4 * (attempt + 1) + random.uniform(0, 0.3)
            try:
                r = self._http.get(self.base_url + path, params=params, timeout=self.timeout)
                if r.status_code != 200:
                    if r.status_code == 429:
                        try:
                            delay = min(float(r.headers.get("Retry-After", delay)), 5.0)
                        except (TypeError, ValueError):
                            pass
                    raise TixAPIError(f"HTTP {r.status_code} for {path}", r.status_code)
                try:
                    data = r.json()
                except ValueError as e:
                    raise TixAPIError(f"invalid JSON from {path}") from e
                if ttl:
                    self._cache[key] = (time.time(), data)
                return data
            except TixAPIError as e:
                last = e
                if e.status and e.status < 500 and e.status != 429:  # other 4xx will not improve on retry
                    break
            except requests.Timeout as e:
                last = TixAPIError(f"timeout calling {path}")
                last.__cause__ = e
            except requests.RequestException as e:
                last = TixAPIError(f"network error calling {path}: {type(e).__name__}")
                last.__cause__ = e
            if attempt < self.retries:
                time.sleep(delay)
        log.warning("TIX API failure: %s", last)
        raise last  # type: ignore[misc]

    def products(self, page: int = 1): return self.get(ep.PRODUCTS, {"page": page})
    def product(self, pid: int): return self.get(ep.PRODUCT.format(id=pid))
    def search(self, q: str): return self.get(ep.SEARCH, {"q": q})
    def filter(self, page: int = 1, **params): return self.get(ep.FILTER, {**params, "page": page})
    def discounted(self, page: int = 1): return self.get(ep.DISCOUNTED, {"page": page})
    def offers(self): return self.get(ep.OFFERS)
    def offer_products(self, oid: int): return self.get(ep.OFFER_PRODUCTS.format(id=oid))
    def categories(self): return self.get(ep.CATEGORIES, cache=True)
    def subcategories(self): return self.get(ep.SUBCATEGORIES, cache=True)
    def category_subcategories(self, cid: int): return self.get(ep.CATEGORY_SUBCATEGORIES.format(id=cid), cache=True)
    def category_products(self, cid: int, page: int = 1): return self.get(ep.CATEGORY_PRODUCTS.format(id=cid), {"page": page})
    def subcategory_products(self, sid: int, page: int = 1): return self.get(ep.SUBCATEGORY_PRODUCTS.format(id=sid), {"page": page})
    def brands(self): return self.get(ep.BRANDS, cache=True)
