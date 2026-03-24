# Browser Action Verification Runbook

## Instructions for the Verifier (Claude Code with Chrome MCP)

**Prerequisites:**
1. Start the Next.js app (`npm run start` or `npm run dev`)
2. Seed entities via `POST /api/init` with test data
3. Open the app in Chrome via `mcp__claude-in-chrome__navigate`

**Protocol — you MUST follow this exactly:**

For EVERY action below, you must:
1. Navigate to the specified page URL
2. Wait for page load (2-3 seconds for client-rendered pages)
3. Execute the specified check using `javascript_tool` or `computer`
4. Record the result as PASS or FAIL with evidence
5. If FAIL, describe what's wrong

**Do NOT skip any action. Do NOT batch multiple actions into one check. One action = one explicit verification.**

After completing all checks, produce a summary table with the pass/fail count.

---

## HOME page (`/`)

### Action 1: navigate_to_amazon
- **Type:** Observation (no element)
- **Check:** Navigate to `/`. Page renders with entity content visible.
- **Verify:** Page contains category names from entities AND product names from entities.
- **Result:** ___

### Action 2: enter_search_query
- **Type:** Interaction — `search_input` / type
- **Check:** Find `[data-action="search_input"]`. Must be an `<input>` element, visible, typeable.
- **Verify:** Element exists, tag=INPUT, type=text, visible=true.
- **Result:** ___

### Action 3: submit_search
- **Type:** Interaction — `search_submit` / click
- **Check:** Find `[data-action="search_submit"]`. Must be a clickable button.
- **Verify:** Element exists, tag=BUTTON, visible=true. Click it (after typing a query) → URL changes to `/search?q=...`.
- **Result:** ___

### Action 4: navigate_to_category
- **Type:** Interaction — `category_menu` / click
- **Check:** Find `[data-action="category_menu"]`. Must be a link to a category page.
- **Verify:** Element exists, tag=A, href contains `/category/`, visible=true. Click it → lands on category page.
- **Result:** ___

### Action 5: navigate_to_deals
- **Type:** Interaction — `deals_link` / click
- **Check:** Find `[data-action="deals_link"]`. Must be a link to `/deals`.
- **Verify:** Element exists, tag=A, href contains `/deals`, visible=true. Click it → lands on deals page.
- **Result:** ___

---

## SEARCH RESULTS page (`/search?q=test+alpha+product`)

### Action 6: read_results_text
- **Type:** Observation (no element)
- **Check:** Navigate to search results. Page renders entity products.
- **Verify:** Page text contains ALL product names from entities (Alpha, Beta, Gamma), plus prices, ratings, brands.
- **Result:** ___

### Action 7: compile_results
- **Type:** Observation (no element)
- **Check:** Same page as read_results_text — agent screenshots to compile.
- **Verify:** Product information is visually present and readable (already verified by Action 6).
- **Result:** ___

### Action 8: refine_search
- **Type:** Interaction — `search_input` / type
- **Check:** On search results page, find `[data-action="search_input"]`. Must be typeable.
- **Verify:** Element exists on search results page (via layout), tag=INPUT, visible=true.
- **Result:** ___

### Action 9: click_product_1
- **Type:** Interaction — `product_card_1` / click
- **Check:** Find `[data-action="product_card_1"]`. Must link to `/product/0`.
- **Verify:** Element exists, visible=true, href ends with `/product/0`. Click it → lands on product detail page showing product #1 entity data.
- **Result:** ___

### Action 10: click_product_2
- **Type:** Interaction — `product_card_2` / click
- **Check:** Find `[data-action="product_card_2"]`. Must link to `/product/1`.
- **Verify:** Element exists, visible=true, href ends with `/product/1`. Click it → lands on product detail page showing product #2 entity data.
- **Result:** ___

### Action 11: click_product_3
- **Type:** Interaction — `product_card_3` / click
- **Check:** Find `[data-action="product_card_3"]`. Must link to `/product/2`.
- **Verify:** Element exists, visible=true, href ends with `/product/2`. Click it → lands on product detail page showing product #3 entity data.
- **Result:** ___

### Action 12: filter_condition_new
- **Type:** Interaction — `filter_condition_new` / click
- **Check:** Find `[data-action="filter_condition_new"]`. Must be clickable and add filter param to URL.
- **Verify:** Element exists, visible=true. Click it → URL contains `f_new=1`.
- **Result:** ___

### Action 13: filter_price_range
- **Type:** Interaction — `filter_price_range` / click
- **Check:** Find `[data-action="filter_price_range"]`. Must be clickable.
- **Verify:** Element exists, visible=true. Click it → URL contains `f_price=1`.
- **Result:** ___

### Action 14: filter_prime_only
- **Type:** Interaction — `filter_prime_only` / click
- **Check:** Find `[data-action="filter_prime_only"]`. Must be clickable.
- **Verify:** Element exists, visible=true. Click it → URL contains `f_prime=1`. Non-Prime products removed from results.
- **Result:** ___

### Action 15: sort_price_asc
- **Type:** Interaction — `sort_price_asc` / click
- **Check:** Find `[data-action="sort_price_asc"]`. Must be clickable.
- **Verify:** Element exists, visible=true. Click it → URL contains `sort=price-asc`. Products reordered by price (cheapest first).
- **Result:** ___

### Action 16: sort_avg_review
- **Type:** Interaction — `sort_avg_review` / click
- **Check:** Find `[data-action="sort_avg_review"]`. Must be clickable.
- **Verify:** Element exists, visible=true. Click it → URL contains `sort=avg-review`. Products reordered by rating (highest first).
- **Result:** ___

### Action 17: open_cart
- **Type:** Interaction — `cart_icon` / click
- **Check:** Find `[data-action="cart_icon"]`. Must link to `/cart`.
- **Verify:** Element exists, tag=A, href contains `/cart`, visible=true.
- **Result:** ___

---

## PRODUCT DETAIL page — Product 1 (`/product/0`)

Navigate to `/product/0`. Wait 3 seconds for client-side render.

### Action 18: read_p1_detail
- **Type:** Observation (no element)
- **Check:** Page renders product #1 entity data.
- **Verify:** Page text contains: product name, brand, ASIN, price, rating, features. ALL must be from the entity data, not hardcoded.
- **Result:** ___

### Action 19: check_p1_shipping
- **Type:** Interaction — `shipping_zip_input` / type
- **Check:** Find `[data-action="shipping_zip_input"]`. Must be an input that accepts ZIP code.
- **Verify:** Element exists, tag=INPUT, visible=true. After entering ZIP and clicking Check, shipping estimate appears with delivery days from entity data.
- **Result:** ___

### Action 20: check_p1_reviews
- **Type:** Interaction — `reviews_section` / click
- **Check:** Find `[data-action="reviews_section"]`. Must be clickable to expand reviews.
- **Verify:** Element exists, visible=true. Click it → review content appears (review title, text, reviewer name from entity data).
- **Result:** ___

### Action 21: check_p1_seller
- **Type:** Interaction — `seller_link` / click
- **Check:** Find `[data-action="seller_link"]`. Must be clickable to expand seller info.
- **Verify:** Element exists, visible=true. Click it → seller name, rating, feedback percentage from entity data appear.
- **Result:** ___

### Action 22: check_p1_qa
- **Type:** Interaction — `qa_section` / click
- **Check:** Find `[data-action="qa_section"]`. Must be clickable to expand Q&A.
- **Verify:** Element exists, visible=true. Click it → Q&A question and answer from entity data appear.
- **Result:** ___

### Action 23: add_p1_to_cart
- **Type:** Interaction — `add_to_cart_button` / click
- **Check:** Find `[data-action="add_to_cart_button"]`. Must be a clickable button.
- **Verify:** Element exists, tag=BUTTON, visible=true. Click it → button text changes to indicate item added.
- **Result:** ___

### Action 24: select_p1_variant_a
- **Type:** Interaction — `variant_option_a` / click
- **Check:** Find `[data-action="variant_option_a"]`. Must be clickable.
- **Verify:** Element exists, visible=true, shows variant option name from entity data. Click it → variant becomes selected (visual change).
- **Result:** ___

### Action 25: select_p1_variant_b
- **Type:** Interaction — `variant_option_b` / click
- **Check:** Find `[data-action="variant_option_b"]`. Must be clickable.
- **Verify:** Element exists, visible=true, shows second variant option from entity data. Click it → variant becomes selected, price may change.
- **Result:** ___

### Action 26: back_to_results_from_p1
- **Type:** Interaction — `back_to_results` / click
- **Check:** Find `[data-action="back_to_results"]`. Must navigate back to search results.
- **Verify:** Element exists, tag=A, visible=true. Click it → URL changes to `/search?q=...`.
- **Result:** ___

---

## PRODUCT DETAIL page — Product 2 (`/product/1`)

Navigate to `/product/1`. Wait 3 seconds for client-side render.

### Action 27: read_p2_detail
- **Type:** Observation (no element)
- **Check:** Page renders product #2 entity data.
- **Verify:** Page text contains product #2 name and brand from entities (NOT product #1 data).
- **Result:** ___

### Action 28: check_p2_shipping
- **Type:** Interaction — `shipping_zip_input` / type
- **Check:** Find `[data-action="shipping_zip_input"]` on product #2 page.
- **Verify:** Element exists, tag=INPUT, visible=true. After ZIP entry, shows product #2's shipping info (may differ from product #1).
- **Result:** ___

### Action 29: check_p2_reviews
- **Type:** Interaction — `reviews_section` / click
- **Check:** Find `[data-action="reviews_section"]` on product #2 page.
- **Verify:** Element exists, visible=true. Click it → expands (may show "No reviews yet" if product #2 has no reviews in entities).
- **Result:** ___

### Action 30: check_p2_seller
- **Type:** Interaction — `seller_link` / click
- **Check:** Find `[data-action="seller_link"]` on product #2 page.
- **Verify:** Element exists, visible=true. Click it → shows product #2's seller info from entities.
- **Result:** ___

### Action 31: check_p2_qa
- **Type:** Interaction — `qa_section` / click
- **Check:** Find `[data-action="qa_section"]` on product #2 page.
- **Verify:** Element exists if product #2 has qa_pairs in entities. If qa_pairs is empty, element should NOT exist (section hidden). Record which case applies.
- **Result:** ___

### Action 32: add_p2_to_cart
- **Type:** Interaction — `add_to_cart_button` / click
- **Check:** Find `[data-action="add_to_cart_button"]` on product #2 page.
- **Verify:** Element exists, tag=BUTTON, visible=true. Click it → button text changes.
- **Result:** ___

### Action 33: back_to_results_from_p2
- **Type:** Interaction — `back_to_results` / click
- **Check:** Find `[data-action="back_to_results"]` on product #2 page.
- **Verify:** Element exists, tag=A, visible=true. Click it → returns to search results.
- **Result:** ___

---

## PRODUCT DETAIL page — Product 3 (`/product/2`)

Navigate to `/product/2`. Wait 3 seconds for client-side render.

### Action 34: read_p3_detail
- **Type:** Observation (no element)
- **Check:** Page renders product #3 entity data.
- **Verify:** Page text contains product #3 name and brand from entities.
- **Result:** ___

### Action 35: check_p3_shipping
- **Type:** Interaction — `shipping_zip_input` / type
- **Check:** Find `[data-action="shipping_zip_input"]` on product #3 page.
- **Verify:** Element exists, tag=INPUT, visible=true.
- **Result:** ___

### Action 36: check_p3_reviews
- **Type:** Interaction — `reviews_section` / click
- **Check:** Find `[data-action="reviews_section"]` on product #3 page.
- **Verify:** Element exists, visible=true.
- **Result:** ___

### Action 37: back_to_results_from_p3
- **Type:** Interaction — `back_to_results` / click
- **Check:** Find `[data-action="back_to_results"]` on product #3 page.
- **Verify:** Element exists, tag=A, visible=true. Click it → returns to search results.
- **Result:** ___

---

## CATEGORY page (`/category/{slug}`)

### Action 38: read_category_listings
- **Type:** Observation (no element)
- **Check:** Navigate to category page. Page renders category data.
- **Verify:** Page text contains category name, subcategories, AND product listings from entities.
- **Result:** ___

### Action 39: search_within_category
- **Type:** Interaction — `search_input` / type
- **Check:** On category page, find `[data-action="search_input"]` (in navbar).
- **Verify:** Element exists, tag=INPUT, visible=true. Typing and submitting navigates to `/search?q=...`.
- **Result:** ___

---

## DEALS page (`/deals`)

### Action 40: read_deal_listings
- **Type:** Observation (no element)
- **Check:** Navigate to deals page. Page renders deal data.
- **Verify:** Page text contains deal product name, deal price, deal type badge from entities.
- **Result:** ___

### Action 41: click_deal_product
- **Type:** Interaction — `deal_card_1` / click
- **Check:** Find `[data-action="deal_card_1"]`. Must link to a product page.
- **Verify:** Element exists, visible=true, href contains `/product/`. Click it → lands on product detail page.
- **Result:** ___

---

## TOOL CALL (not browser)

### Action 42: submit_results
- **Type:** Tool call — `submit_result`
- **Check:** This is a verifiers tool, not a browser element. Skip browser verification.
- **Verify:** Confirmed: tool_call with tool_name=submit_result exists in graph contract.
- **Result:** PASS (by contract definition)

---

## CART page (`/cart`)

### Action 43: read_cart
- **Type:** Observation (no element)
- **Check:** Navigate to `/cart`. If items have been added via add_to_cart, they should appear.
- **Verify:** Page renders cart contents OR empty cart message. If items were added during this test session, verify they appear with correct names and prices from entities.
- **Result:** ___

---

## Summary Table

Fill in after completing ALL checks:

| # | Action ID | Type | Result | Notes |
|---|-----------|------|--------|-------|
| 1 | navigate_to_amazon | observation | ___ | |
| 2 | enter_search_query | search_input/type | ___ | |
| 3 | submit_search | search_submit/click | ___ | |
| 4 | navigate_to_category | category_menu/click | ___ | |
| 5 | navigate_to_deals | deals_link/click | ___ | |
| 6 | read_results_text | observation | ___ | |
| 7 | compile_results | observation | ___ | |
| 8 | refine_search | search_input/type | ___ | |
| 9 | click_product_1 | product_card_1/click | ___ | |
| 10 | click_product_2 | product_card_2/click | ___ | |
| 11 | click_product_3 | product_card_3/click | ___ | |
| 12 | filter_condition_new | filter_condition_new/click | ___ | |
| 13 | filter_price_range | filter_price_range/click | ___ | |
| 14 | filter_prime_only | filter_prime_only/click | ___ | |
| 15 | sort_price_asc | sort_price_asc/click | ___ | |
| 16 | sort_avg_review | sort_avg_review/click | ___ | |
| 17 | open_cart | cart_icon/click | ___ | |
| 18 | read_p1_detail | observation | ___ | |
| 19 | check_p1_shipping | shipping_zip_input/type | ___ | |
| 20 | check_p1_reviews | reviews_section/click | ___ | |
| 21 | check_p1_seller | seller_link/click | ___ | |
| 22 | check_p1_qa | qa_section/click | ___ | |
| 23 | add_p1_to_cart | add_to_cart_button/click | ___ | |
| 24 | select_p1_variant_a | variant_option_a/click | ___ | |
| 25 | select_p1_variant_b | variant_option_b/click | ___ | |
| 26 | back_to_results_from_p1 | back_to_results/click | ___ | |
| 27 | read_p2_detail | observation | ___ | |
| 28 | check_p2_shipping | shipping_zip_input/type | ___ | |
| 29 | check_p2_reviews | reviews_section/click | ___ | |
| 30 | check_p2_seller | seller_link/click | ___ | |
| 31 | check_p2_qa | qa_section/click | ___ | |
| 32 | add_p2_to_cart | add_to_cart_button/click | ___ | |
| 33 | back_to_results_from_p2 | back_to_results/click | ___ | |
| 34 | read_p3_detail | observation | ___ | |
| 35 | check_p3_shipping | shipping_zip_input/type | ___ | |
| 36 | check_p3_reviews | reviews_section/click | ___ | |
| 37 | back_to_results_from_p3 | back_to_results/click | ___ | |
| 38 | read_category_listings | observation | ___ | |
| 39 | search_within_category | search_input/type | ___ | |
| 40 | read_deal_listings | observation | ___ | |
| 41 | click_deal_product | deal_card_1/click | ___ | |
| 42 | submit_results | tool_call | PASS | |
| 43 | read_cart | observation | ___ | |

**Total: ___/43 PASS**
