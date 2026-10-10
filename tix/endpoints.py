"""Paths taken from the supplied Postman collection (public unless noted)."""
PRODUCTS = "/api/products"
PRODUCT = "/api/products/{id}"
SEARCH = "/api/search"
FILTER = "/api/product/filter"
DISCOUNTED = "/api/product/discounted"
OFFERS = "/api/offers"
OFFER_PRODUCTS = "/api/offers/{id}/products"
CATEGORIES = "/api/categories"  # Postman marks Bearer; anonymous access NOT YET VERIFIED
SUBCATEGORIES = "/api/subcategories"  # same caveat
CATEGORY_SUBCATEGORIES = "/api/category/{id}/subcategories"
CATEGORY_PRODUCTS = "/api/category/{id}/products"
SUBCATEGORY_PRODUCTS = "/api/subcategory/{id}/products"
BRANDS = "/api/brands"  # same caveat
RETURN_POLICY = "/api/return-policy"
