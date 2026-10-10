# TIX shopping assistant (v6.3)

A conversational sales assistant for the TIX online store. The customer just talks (Egyptian Arabic, English or a mix);
the LLM understands, decides when to look something up, and the **TIX API is the only source of product facts**.

```
customer message
   -> LLM (Groq) sees: recent chat + what is on screen + last search
        -> [ tool call -> real TIX data -> LLM ]*       (0 calls for small talk)
   -> short natural reply, optionally ending with [[show: id, id]]  -> product cards (image / price / link / video)
```

A product request normally costs **2 LLM calls** (search, then the reply that also says which products to display); small talk costs 1.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # put your GROQ_API_KEY in .env (never commit it)
streamlit run app.py          # the assistant IS the main page (centred chat card)
python chat_cli.py            # same agent in the terminal, with a trace of every tool call
python chat_cli.py --script scenarios.txt   # replay realistic multi-turn conversations against the real model + API
python chat_cli.py --catalog-check "Redmi A5"   # is a product in the cache? in live search? in its live category listing?
python chat_cli.py --check-llm   # which configured Groq models exist for your key + do they answer / call tools
python -m pytest              # offline tests (fake TIX API, scripted model, simulated Groq errors)
TIX_LIVE=1 pytest tests/test_live_smoke.py -v -s   # live API (+ real model if GROQ_API_KEY is set)
```

## Layout

| Path | Role |
|---|---|
| `agent/agent.py` | The whole agent: one tool-calling loop, reply validation, price check, failure fallbacks |
| `agent/prompts.py` | The single system prompt + the compact "what's on screen" context |
| `agent/tools.py` | 4 tools: `search_products`, `get_product_details`, `compare_products`, `list_categories` |
| `agent/llm.py` | Groq client: 429 cooldowns (Retry-After), dead-model memory, empty-reasoning / `tool_use_failed` handling |
| `agent/guards.py` | Budget, brand, test products, dedupe, price check, reply validation, `[[show]]` parsing |
| `agent/state.py`, `agent/render.py` | Session data; product-card dict + fixed failure messages |
| `tix/` | `client.py` (HTTP), `schemas.py` (tolerant parsing incl. media), `catalog.py` (cached catalogue + word matching) |
| `ui/`, `app.py` | Streamlit main-page UI: symmetric header, greeting, 2x2 quick-action grid, pill composer (no attachments); product cards with image / video link |
| `docs/api_notes.md` | What is VERIFIED vs assumed about the API |

## How it understands a conversation

* **One brain.** The same model that chats decides whether a lookup is needed. "إزيك؟" costs one LLM call and zero API calls.
* **Memory.** The last 14 chat messages are sent each turn, plus a tiny block listing the products currently on screen (id, name, price,
  category) and the last search filters. "Cheaper", "the second one", "another kind" resolve against that, never from scratch.
* **Understand first, search second.** The model writes 1-4 *phrasings* for the store (e.g. لبن → حليب / لبن / milk), so there is
  no synonym table: language knowledge stays in the model, matching stays dumb and fast.
* **Meaning, not keywords.** Every result carries its category path, and each search also returns *which categories the matches fall in*.
  "OGX Coconut Milk Shampoo" shows up under `العناية بالشعر > شامبو`, the real milk under `ألبان وأجبان > حليب`; the model picks the right
  one (or re-searches with `category=`). The look-alike cannot crowd out the real product, because results are interleaved across categories.
* **Missing information → one useful question** ("عنده كام سنة وميزانيتك كام؟"), not a random search.

## How facts stay real

* Cards are produced only from the `[[show: ids]]` line, and only for **ids that a tool returned in this conversation**; the marker is
  stripped from the text. Unknown ids are ignored.
* Budget, brand and test-product filters are enforced in code. When a budget or brand excludes everything, the tool reports what really exists
  (price range, brands present, what the category holds) so the model can say "no Samsung, but we have Xiaomi and Apple" truthfully.
* Any price the model writes next to a currency word must exist in real data (or be the customer's own number).
* **Every reply is validated before a customer sees it** (`guards.reply_problem`): internal wording (errors, 429, JSON, `<think>`, "Oops"),
  foreign scripts, garbled mixed-script words, fragment/ellipsis spam and repetition are rejected. The model gets ONE clean retry (never while
  rate-limited); if that also fails, a safe fixed message is shown instead.

## Matching words to products (general, not per product type)

* The same small Arabic stemmer is applied to the customer's words and to every product/category word (article and plural/feminine endings:
  الموبايلات / موبايلات / موبايل, شفرات / شفرة), plus a fuzzy match for long words (تيشيرت ~ تيشرت). No dictionary of products or slang.
* Category paths are part of the matched text, so a product whose own name lacks the category word is still found through its category.
* If strict matching finds nothing, the tool returns `near_matches` (weaker partial matches that still pass category/brand/budget filters)
  for the model to judge, instead of a bare "nothing".
* Known limit: broken plurals (الأجهزة vs جهاز) cannot be reduced by rules; the prompt makes the model search several phrasings.

## Keeping the catalogue answer complete

* The cached snapshot is a convenience, not the truth. Whenever the model narrows to a category, that category is **also read live**
  (`/api/subcategory/{id}/products`) and any product the snapshot lacks is added if it fits the request, so a stale or partly downloaded
  snapshot cannot hide a product (the "only one Android phone, nothing under 5000" case).
* Pages that fail to download are retried gently, tracked (`Catalog.missing`) and re-fetched alone; the TIX client retries 429/5xx
  (honouring `Retry-After`) instead of dropping the page.
* When a category filter excluded matches, the tool says so (`outside_category`), and the prompt forbids "that's all we have" claims without it.
* The terminal log shows each search's stages, e.g. `matched=…, after_category=…, after_budget=…, live-added=1`.

## When Groq is limited or down

* A **429 is not a failure**. The limited model goes on cooldown (from `Retry-After`, the "try again in 7.2s" hint, or a default; a daily quota
  gets a long cooldown) and the next configured model, which has its own rate-limit bucket, is tried **immediately**. If every model is cooling:
  a short wait (`LLM_MAX_WAIT`, jittered, once) or, if longer, no request at all: the customer message is answered locally.
* Cooldowns persist across messages, so the next message does not spend a request on a model known to be limited. A model that answers 404 is
  marked dead for the process (and the configured list is checked once against `GET /models`).
* **Valid results survive.** If a search succeeded and the LLM then fails or returns garbage, the products found are shown with "لقيتلك دول 👇"
  when the search was unambiguous (one category, or narrowed by the model). If the matches spanned several categories, nothing is guessed:
  the customer gets a short "please clarify / try again" message.
* The customer never sees provider errors, JSON, stack traces or `<think>` text.

## Configuration (`.env` / environment)

`GROQ_API_KEY` (required), `GROQ_MODEL` (comma-separated, tried in order; default `openai/gpt-oss-120b,openai/gpt-oss-20b`; Groq retires
models: run `python chat_cli.py --check-llm` to see what your key can use), `TIX_BASE_URL`, `TIX_SITE_URL` (product links), `TIX_MEDIA_BASE` (prefix for relative
image/video paths), `CATALOG_TTL`, `AGENT_MAX_STEPS`, `HISTORY_MESSAGES`, `LLM_MAX_WAIT`, `LLM_DEFAULT_COOLDOWN`. All defaults live in `config.py`.

## Known limits (see also the delivery report)

* The tests prove orchestration, filtering, state, call counts and failure handling with a **scripted** model and a **simulated** Groq; they cannot
  measure Arabic understanding or real rate-limit headers. Run `chat_cli.py --script scenarios.txt` with your key to judge real conversations.
* Image/video field *shapes* are unverified (`docs/api_notes.md`): confirm with `python chat_cli.py --raw-media <id>`.
* In-stock status relies on `quantity` from the product-details endpoint; search results do not carry it.
* No cart / checkout / order tracking (out of scope).
