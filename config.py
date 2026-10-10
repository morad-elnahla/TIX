"""Central configuration. Every tunable lives here; override through environment variables / .env."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


# ---- TIX API -------------------------------------------------------------
TIX_BASE_URL = os.getenv("TIX_BASE_URL", "http://admin.tix-eg.com").rstrip("/")
TIX_SITE_URL = os.getenv("TIX_SITE_URL", "https://tix-eg.com").rstrip("/")  # storefront, used for product links
TIX_MEDIA_BASE = os.getenv("TIX_MEDIA_BASE", "").rstrip("/")  # prefix for relative image/video paths (default: TIX_BASE_URL)
TIX_ACCESS_TOKEN = os.getenv("TIX_ACCESS_TOKEN", "")  # optional; the endpoints used here are public
TIX_LANG = os.getenv("TIX_LANG", "ar")  # sent as Accept-Language
TIX_TIMEOUT = float(os.getenv("TIX_TIMEOUT", "10"))
TIX_RETRIES = int(os.getenv("TIX_RETRIES", "2"))  # extra attempts for 429/5xx/timeouts
MAX_PAGES = int(os.getenv("MAX_PAGES", "3"))  # pages read from paged endpoints when the catalogue snapshot is missing
CURRENCY = "EGP"

# ---- Catalogue snapshot (a cached copy of GET /api/products, refreshed in the background) ----
CATALOG_TTL = int(os.getenv("CATALOG_TTL", "900"))  # seconds
CATALOG_MAX_PAGES = int(os.getenv("CATALOG_MAX_PAGES", "100"))
CATALOG_WORKERS = int(os.getenv("CATALOG_WORKERS", "4"))  # parallel page downloads (kept modest: a burst can trigger 429)
LIVE_CATEGORY_TTL = int(os.getenv("LIVE_CATEGORY_TTL", "60"))  # seconds a live category listing is reused

# ---- UI ------------------------------------------------------------------
ASSISTANT_SIDE = "left" if os.getenv("ASSISTANT_SIDE", "right").strip().lower() == "left" else "right"  # screen edge of the floating assistant

# ---- Agent ---------------------------------------------------------------
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "4"))  # tool-calling rounds per customer message (a normal product request uses 1)
HISTORY_MESSAGES = int(os.getenv("HISTORY_MESSAGES", "14"))  # recent chat messages sent to the LLM
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "8"))  # products returned to the LLM per search
MAX_CARDS = int(os.getenv("MAX_CARDS", "5"))  # product cards shown to the customer per answer
TOOL_RESULT_CHARS = int(os.getenv("TOOL_RESULT_CHARS", "5000"))  # hard cap on one tool result sent to the LLM

# ---- Groq ----------------------------------------------------------------
# Provider = Groq. Models = this list only (comma-separated, tried in order; each model has its own rate-limit bucket,
# so the second one is a real fallback). Groq retires/renames models: verify with `python chat_cli.py --check-llm`.
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODELS = [m.strip() for m in os.getenv("GROQ_MODEL", "openai/gpt-oss-120b,openai/gpt-oss-20b").split(",") if m.strip()]
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
LLM_RETRIES = int(os.getenv("LLM_RETRIES", "1"))  # extra attempts on the SAME model for 5xx/empty/malformed output (never for 429)
LLM_MAX_WAIT = float(os.getenv("LLM_MAX_WAIT", "6"))  # longest we will pause inside one customer turn waiting for a rate limit
LLM_DEFAULT_COOLDOWN = float(os.getenv("LLM_DEFAULT_COOLDOWN", "10"))  # when a 429 says nothing about when to retry
LLM_DAILY_COOLDOWN = float(os.getenv("LLM_DAILY_COOLDOWN", "900"))  # a daily quota will not recover in seconds
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))  # reasoning models spend part of this on hidden reasoning
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")  # only sent to gpt-oss models

def get_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key

    try:
        import streamlit as st
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        return ""


# ---- Guards --------------------------------------------------------------
HIDE_TEST_PRODUCTS = _bool("HIDE_TEST_PRODUCTS", True)
TEST_PRODUCT_PATTERN = os.getenv("TEST_PRODUCT_PATTERN", r"(?<![a-z])(test|demo|dummy)(?![a-z])|تجريبي|تجربة")
USER_TEST_KEYWORDS = ("test", "demo", "تيست", "تجريبي")  # if the customer says one of these, test products are allowed
CATALOG_CACHE_TTL = int(os.getenv("CATALOG_CACHE_TTL", "3600"))  # categories/brands cache inside the client