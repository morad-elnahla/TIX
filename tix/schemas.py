"""Tolerant parsing of TIX payloads into a small Product model.

The exact wrapper of list responses (data / meta) is not documented; extractors search for it.
Brand/category may be a string or an object {id, name}; both are handled.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

import config

_DIGITS = str.maketrans({**{c: str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}, "٫": ".", "٬": "", ",": ""})


def parse_price(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v.translate(_DIGITS))
        return float(m.group()) if m else None
    return None


def label(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, dict):
        for k in ("name", "title", "ar", "en", "label"):
            if v.get(k):
                return label(v[k])
    return ""


def ref_id(v: Any) -> int | None:
    try:
        if isinstance(v, dict) and v.get("id") is not None:
            return int(v["id"])
    except (TypeError, ValueError):
        pass
    return None


def labels(v: Any) -> tuple[list[str], list[int]]:
    """Names and ids from a field that may be a list of {id, name}, one object, or a plain string
    (used for `subsubcategories`, whose exact shape is not documented)."""
    items = v if isinstance(v, list) else ([v] if v not in (None, "", {}) else [])
    names = [n for n in (label(i) for i in items) if n]
    ids = [i for i in (ref_id(x) for x in items) if i is not None]
    return names, ids


def clean_html(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(s))).strip()


def _keywords(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(x for x in (label(k) or (k if isinstance(k, str) else "") for k in v) if x)[:200]
    return clean_html(v)[:200] if isinstance(v, str) else ""


_MEDIA_KEYS = ("url", "path", "image", "src", "link", "file", "video", "name")


def _resolve(url: str) -> str:
    """Absolute http(s) URL for an image/video reference. Relative paths are joined to TIX_MEDIA_BASE (default: the API host)."""
    url = url.strip()
    if url.startswith(("http://", "https://")):
        return url
    if url.startswith("//"):
        return "https:" + url
    base = config.TIX_MEDIA_BASE or config.TIX_BASE_URL
    return f"{base}/{url.lstrip('/')}"


def media_urls(v: Any, _depth: int = 0) -> list[str]:
    """URLs from a media field that may be a string, a list of strings, or a list of {url|path|...} objects (shape UNVERIFIED)."""
    out: list[str] = []
    if isinstance(v, str) and v.strip():
        out.append(_resolve(v))
    elif isinstance(v, dict) and _depth < 3:
        for k in _MEDIA_KEYS:
            if isinstance(v.get(k), str) and v[k].strip():
                out.append(_resolve(v[k]))
                break
    elif isinstance(v, list) and _depth < 3:
        for x in v:
            out.extend(media_urls(x, _depth + 1))
    return list(dict.fromkeys(out))


def video_urls(raw: dict) -> list[str]:
    """Any field whose key mentions 'video' (the video field name is not documented, so scan for it)."""
    out: list[str] = []
    for k, v in raw.items():
        if "video" in str(k).lower():
            out.extend(media_urls(v))
    return list(dict.fromkeys(out))


_NO_BRAND = {"غير مذكور", "غير محدد", "غير معروف", "لا يوجد", "بدون", "n/a", "na", "none", "null", "unknown", "not specified", "-", "--"}


def _brand(v: Any) -> str:
    """The store writes "غير مذكور" (= "not mentioned") when a product has no brand: that is an absence, not a brand."""
    b = label(v)
    return "" if b.strip().lower() in _NO_BRAND else b


@dataclass(frozen=True)
class Product:
    id: int | None
    name: str
    price: float | None  # effective price: price_after if > 0 else price_before
    price_before: float | None = None
    discount: float = 0.0
    brand: str = ""
    category: str = ""
    subcategory: str = ""
    category_id: int | None = None
    subcategory_id: int | None = None
    short_description: str = ""
    keywords: str = ""
    slug: str = ""
    description: str = ""
    subsubcategory: str = ""  # real third level of the catalogue tree (names joined by " / ")
    subsubcategory_ids: tuple = ()
    images: tuple = ()  # absolute image URLs (list responses carry `images`)
    videos: tuple = ()  # absolute video URLs, when the payload has any

    @property
    def image(self) -> str:
        return self.images[0] if self.images else ""

    @property
    def path(self) -> str:
        """The product's place in the real catalogue tree: "category > subcategory > subsubcategory"."""
        return " > ".join(x for x in (self.category, self.subcategory, self.subsubcategory) if x)

    @property
    def search_text(self) -> str:
        """What embeddings/lexical retrieval see: every real catalogue field, with category + subcategory kept
        prominent because they often carry the meaning ("ملابس رجالي > تيشرتات") the product name lacks."""
        where = self.path
        parts = [self.name, self.brand, f"{where}" if where else "", self.short_description, self.description, self.keywords]
        return ". ".join(x for x in parts if x)[:600]

    @property
    def has_discount(self) -> bool:
        if self.discount and self.discount > 0:
            return True
        return bool(self.price and self.price_before and self.price_before > self.price)

    @classmethod
    def from_raw(cls, raw: dict) -> "Product":
        after, before = parse_price(raw.get("price_after")), parse_price(raw.get("price_before"))
        price = after if after and after > 0 else (before if before else parse_price(raw.get("price")))
        try:
            pid = int(raw["id"]) if raw.get("id") is not None else None
        except (TypeError, ValueError):
            pid = None
        sub_names, sub_ids = labels(raw.get("subsubcategories") if raw.get("subsubcategories") is not None
                                    else raw.get("subsubcategory"))
        return cls(
            id=pid,
            name=label(raw.get("name")),
            price=price,
            price_before=before,
            discount=parse_price(raw.get("discount")) or 0.0,
            brand=_brand(raw.get("brand")),
            category=label(raw.get("category")),
            subcategory=label(raw.get("subcategory")),
            category_id=ref_id(raw.get("category")),
            subcategory_id=ref_id(raw.get("subcategory")),
            short_description=clean_html(raw.get("short_description"))[:300],
            keywords=_keywords(raw.get("keywords")),
            slug=raw["slug"].strip() if isinstance(raw.get("slug"), str) else "",
            description=clean_html(raw.get("description") or raw.get("long_description"))[:300],
            subsubcategory=" / ".join(sub_names),
            subsubcategory_ids=tuple(sub_ids),
            images=tuple(media_urls(raw.get("images") if raw.get("images") is not None else raw.get("image"))),
            videos=tuple(video_urls(raw)),
        )


def extract_items(payload: Any, _depth: int = 0) -> list[dict]:
    """Find the list of item objects inside a list-style response."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict) and _depth < 3:
        for key in ("data", "products", "items", "results"):
            if key in payload:
                found = extract_items(payload[key], _depth + 1)
                if found or isinstance(payload[key], list):
                    return found
    return []


def extract_last_page(payload: Any, _depth: int = 0) -> int:
    if isinstance(payload, dict) and _depth < 4:
        if isinstance(payload.get("last_page"), int):
            return payload["last_page"]
        for v in payload.values():
            if isinstance(v, dict):
                n = extract_last_page(v, _depth + 1)
                if n > 1:
                    return n
    return 1