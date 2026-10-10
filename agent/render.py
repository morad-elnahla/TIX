"""Fixed messages for the failure paths (the LLM is not available there) and the product-card dict for the UI."""
from __future__ import annotations

import re
from urllib.parse import quote

import config
from tix.schemas import Product

MSG = {
    "ar": {
        "greeting": "أهلاً 👋 إزيك؟ بتدور على حاجة معينة النهارده؟",
        "llm_down": "حصلت مشكلة بسيطة من عندي 🙏 ممكن تبعتلي رسالتك تاني؟",
        "busy": "الدنيا زحمة شوية دلوقتي 🙏 ابعتلي رسالتك تاني بعد لحظات.",
        "found": "لقيتلك دول 👇",
        "unsure": "عشان أجيبلك الأنسب بالظبط، ممكن توضحلي أكتر عايز إيه؟ 🙏",
        "empty": "ممكن توضحلي أكتر عايز إيه؟ 🙏",
        "fix_failed": "الأسعار مكتوبة على كل منتج، قولّي لو عايز أفلتر حسب ميزانية معينة 🙏",
        "currency": "جنيه",
    },
    "en": {
        "greeting": "Hi 👋 How are you? Looking for something in particular today?",
        "llm_down": "Something hiccuped on my side 🙏 could you send that again?",
        "busy": "It's a bit busy right now 🙏 please send your message again in a moment.",
        "found": "Here's what I found 👇",
        "unsure": "To find the best match, could you tell me a bit more about what you're after? 🙏",
        "empty": "Could you tell me a bit more about what you're looking for? 🙏",
        "fix_failed": "Prices are shown on each product. Tell me if you'd like to filter by budget 🙏",
        "currency": "EGP",
    },
}


def detect_lang(text: str) -> str:
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    la = len(re.findall(r"[A-Za-z]", text))
    return "ar" if ar >= la and ar > 0 else ("en" if la else "ar")


def msg(lang: str, key: str) -> str:
    return MSG.get(lang, MSG["ar"])[key]


def fmt_price(v: float | None) -> str:
    return "—" if v is None else f"{v:,.2f}".rstrip("0").rstrip(".")


def product_url(p: Product) -> str | None:
    if p.id is None:
        return None
    return f"{config.TIX_SITE_URL}/product/{p.id}" + (f"/{quote(p.slug)}" if p.slug else "")


def card(p: Product, media: dict | None = None) -> dict:
    """Plain dict for the UI's product card. Only real API fields; the discount % is derived from real prices."""
    media = media or {}
    pct = None
    if p.price and p.price_before and p.price_before > p.price:
        pct = round((p.price_before - p.price) / p.price_before * 100, 2)  # same figure the storefront shows (1.81%)
    videos = list(media.get("videos") or p.videos)
    images = list(media.get("images") or p.images)
    return {"id": p.id, "name": p.name, "brand": p.brand, "price": p.price,
            "price_before": p.price_before if pct else None, "discount_pct": pct, "url": product_url(p),
            "image": images[0] if images else None, "video": videos[0] if videos else None}
