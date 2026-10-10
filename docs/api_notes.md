# TIX API notes

Base URL: `http://admin.tix-eg.com`.  Status labels are strict:
**VERIFIED** = observed by the project owner against the real API. **FROM POSTMAN** = present in the supplied collection only.
**NOT YET VERIFIED** = assumed by the code; confirm with `TIX_LIVE=1 pytest tests/test_live_smoke.py -v -s`.

## VERIFIED
| Endpoint | Result |
|---|---|
| `GET /api/products` | HTTP 200. Fields: id, name, short_description, price_before, price_after, discount, brand, category, subcategory, subsubcategories, images, is_fav. total 635, 15/page, 43 pages. |
| `GET /api/search?q=Nestle` | HTTP 200, 3 products (Nestlé Splash Lemon 1L, Nestlé Pure Life 330ml, Nestlé Pure Fizz 24×240ml). Prices matched product details in tested cases. |
| `GET /api/products/1070` | HTTP 200. name, slug, short/long description, keywords, price_before/after, quantity, brand, category, subcategory, subsubcategories, return info, max order quantity, features, FAQs, reviews, vendor, images. Product 1070 = مياه نستله سبلاش بنكهة الليمون 1 لتر, 500 EGP, quantity 12, brand Nestlé, vendor التوحيد ماركت. |

Observed data quirks: the `brand` filter can be ignored by the API; some product names contain `test`; many supermarket products cost 500 EGP; `/api/offers` can be empty.

## FROM POSTMAN (public, no auth header in the collection)
`GET /api/product/filter` (category, subcategory, name, brand, price_min, price_max, sort = newest | price_low | price_high | name_asc | name_desc),
`GET /api/product/discounted`, `GET /api/offers`, `GET /api/offers/{id}/products`,
`GET /api/category/{id}/products`, `GET /api/subcategory/{id}/products`, `GET /api/return-policy`.

Marked **Bearer** in Postman (anonymous access unknown): `GET /api/categories`, `/api/subcategories`, `/api/category/{id}/subcategories`, `/api/brands`.
If these return 401, set `TIX_ACCESS_TOKEN` in `.env` (token via `/api/auth/user/login`); the agent then degrades gracefully without category matching.
Cart, favorites, orders, checkout, comments, shipping, payments: out of scope for this MVP.

## NOT YET VERIFIED (assumptions in code, isolated in `tix/schemas.py` / `agent/tools.py`)
- List responses wrap items in `data` (optionally with `meta.last_page`). `extract_items` also accepts a bare list, `products`, `items`, `results`.
- `brand`, `category`, `subcategory` are strings or `{id, name}` objects; names may be `{ar,en}`.
- `page` works on products / filter / discounted. `/api/search` returns everything in one response (only page 1 is read).
- **Media.** `images` exists on list and detail responses (VERIFIED) but its inner shape is not: `media_urls()` accepts a string, a list of
  strings, or objects with `url|path|image|src|link|file`. Relative paths are joined to `TIX_MEDIA_BASE` (default: the API host): if images
  do not load in the UI, set `TIX_MEDIA_BASE` (e.g. the storage/CDN origin). **No video field is documented**: `video_urls()` picks up any
  key containing "video". If the real field has another name, the agent will say "no video" truthfully for every product.
  Confirm both with `python chat_cli.py --raw-media <product_id>`.
- "Newest" has no documented date field: `sort=newest` uses the highest product id as a proxy.
- Detail keys for stock (`quantity`), `vendor`, `features`, `reviews[].rating`, `is_returnable`, `max_order_quantity` come from the
  VERIFIED description of `/api/products/1070`; exact value types are parsed tolerantly.
- `/api/product/discounted` shape; a product counts as discounted when `discount > 0` or `price_before > price_after`.
- `/api/product/filter` parameters `price_min` / `price_max` (Postman) are used only when the catalogue snapshot is unavailable.

## Implementation notes
- Effective price = `price_after` if > 0, else `price_before`.
- Budget, brand, test-product and duplicate rules run in code (`agent/guards.py`) on every search result.
- The agent keeps a **catalogue snapshot**: every page of `GET /api/products` (page 1 gives `last_page`, the rest load in parallel) held in
  memory and refreshed in the background every `CATALOG_TTL` seconds. It is a cache of API data, not a second database. It supplies the
  category path of each product (category > subcategory > subsubcategory), the category tree, and word matching; the live
  `/api/search` endpoint is also called on every search, so products added after the snapshot still appear.
- If the snapshot cannot load, search falls back to `/api/search` alone, and browsing falls back to `/api/product/filter`.
