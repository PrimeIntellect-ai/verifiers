"""Unified task spec generator for Amazon Shopping domain.

Produces the task description AND answer key in a single pass,
guaranteeing they are aligned — every piece of information the
description asks the agent to find is present in the answer key.

Replaces the separate descriptions.py and answer_key.py modules.
"""

from __future__ import annotations

from typing import Any

from entities import TaskEntitySet, Product


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _goal_has(goal_world: list[dict], path: str, value: Any = True) -> bool:
    for pred in goal_world:
        if pred["path"] == path and pred.get("value") == value:
            return True
    return False


def _fmt_price(cents: int) -> str:
    return f"${cents / 100:.2f}"


def _product_summary(product: Product) -> dict[str, Any]:
    """Basic product info — name, brand, price, rating, Prime."""
    summary: dict[str, Any] = {
        "name": product.name,
        "brand": product.brand,
        "price": _fmt_price(product.price_cents),
        "rating": product.rating,
        "review_count": product.review_count,
        "prime_eligible": product.prime_eligible,
    }
    if product.list_price_cents and product.list_price_cents > product.price_cents:
        summary["list_price"] = _fmt_price(product.list_price_cents)
        summary["discount_pct"] = round(
            (1 - product.price_cents / product.list_price_cents) * 100
        )
    return summary


# ---------------------------------------------------------------------------
# Section builders — each returns (description_fragment, answer_key_fragment)
# ---------------------------------------------------------------------------


def _opening(entities: TaskEntitySet, entry_point: str) -> tuple[str, dict]:
    """How to enter the site."""
    if entry_point == "category":
        cat = next((c for c in entities.categories if c.subcategories), None)
        cat_name = cat.name if cat else "Electronics"
        return (
            f"Go to Amazon.com and navigate to the {cat_name} category from the main menu.",
            {},
        )
    elif entry_point == "deals":
        return (
            "Go to Amazon.com and navigate to Today's Deals.",
            {},
        )
    else:
        return (
            f'Go to Amazon.com and search for "{entities.search_query}".',
            {},
        )


def _category_section(
    goal_world: list[dict],
    entities: TaskEntitySet,
) -> tuple[str | None, dict]:
    if not _goal_has(goal_world, "category.read", True):
        return None, {}
    cat = next((c for c in entities.categories if c.subcategories), None)
    cat_name = cat.name if cat else "the category"
    desc = f"Browse the {cat_name} category listings and review the available products."
    ak = {
        "category": {
            "name": cat_name,
            "subcategories": cat.subcategories if cat else [],
        },
        "search_results": [_product_summary(p) for p in entities.products],
    }
    return desc, ak


def _deals_section(
    goal_world: list[dict],
    entities: TaskEntitySet,
) -> tuple[str | None, dict]:
    if not _goal_has(goal_world, "deals.read", True):
        return None, {}
    desc = "Review the current deals and note the discount percentages and original prices."
    deals = []
    for deal in entities.deals:
        deals.append(
            {
                "product_name": deal.product.name,
                "deal_price": _fmt_price(deal.deal_price_cents),
                "original_price": _fmt_price(deal.original_price_cents),
                "discount_pct": round(
                    (1 - deal.deal_price_cents / deal.original_price_cents) * 100
                ),
                "deal_type": deal.deal_type.value
                if hasattr(deal.deal_type, "value")
                else str(deal.deal_type),
            }
        )
    return desc, {"deals": deals}


def _results_section(
    goal_world: list[dict], entities: TaskEntitySet
) -> tuple[str | None, dict]:
    if not _goal_has(goal_world, "extract.results_read", True):
        return None, {}
    desc = (
        "Read through the search results and note the product names, "
        "prices, ratings, and Prime eligibility for each listing."
    )
    ak = {"search_results": [_product_summary(p) for p in entities.products]}
    return desc, ak


def _filter_section(
    goal_world: list[dict], entities: TaskEntitySet
) -> tuple[str | None, dict]:
    parts = []
    condition = next(
        (p.get("value") for p in goal_world if p["path"] == "filter.condition"), None
    )
    if condition and condition != "none":
        parts.append(f"filter results to show only {condition.upper()} condition items")
    if _goal_has(goal_world, "filter.price_range", True):
        if entities.products:
            mid = entities.products[len(entities.products) // 2].price_cents
            low = max(0, mid - 5000)
            high = mid + 5000
            parts.append(f"set a price range of ${low / 100:.0f}–${high / 100:.0f}")
        else:
            parts.append("set a price range filter")
    if _goal_has(goal_world, "filter.prime_only", True):
        parts.append("enable the Prime-only filter")
    sort = next(
        (p.get("value") for p in goal_world if p["path"] == "sort.applied"), None
    )
    if sort and sort != "none":
        sort_labels = {
            "price_asc": "price: low to high",
            "price_desc": "price: high to low",
            "avg_review": "average customer review",
        }
        parts.append(f"sort by {sort_labels.get(sort, sort)}")
    if not parts:
        return None, {}
    return "Apply the following filters: " + ", ".join(parts) + ".", {}


def _refine_section(goal_world: list[dict]) -> tuple[str | None, dict]:
    if not _goal_has(goal_world, "search.refined", True):
        return None, {}
    return (
        "If the initial results don't match what you're looking for, "
        "refine your search query and try again."
    ), {}


def _product_detail_section(
    goal_world: list[dict],
    entities: TaskEntitySet,
    config: dict[str, Any],
) -> tuple[str | None, dict]:
    detail_products = [
        i for i in range(1, 4) if _goal_has(goal_world, f"p{i}.detail_read", True)
    ]
    if not detail_products:
        return None, {}

    products = entities.products
    ak = {}

    if len(detail_products) == 1:
        p = products[0] if products else None
        name = p.name if p else "the product"
        desc = f"Click on {name} to open its product detail page and read the full details."
    elif len(detail_products) == 2:
        names = [
            products[i - 1].name if i - 1 < len(products) else f"product #{i}"
            for i in detail_products
        ]
        desc = (
            f"Open the detail pages for the following products and read their full "
            f"details:\n1. {names[0]}\n2. {names[1]}"
        )
    else:
        names = [
            products[i - 1].name if i - 1 < len(products) else f"product #{i}"
            for i in detail_products
        ]
        items = "\n".join(f"{j + 1}. {name}" for j, name in enumerate(names))
        desc = f"Open the detail pages for each of the following {len(detail_products)} products and read their full details:\n{items}"

    for i in detail_products:
        idx = i - 1
        if idx >= len(products):
            continue
        product = products[idx]
        key = f"product_{i}"
        ak[key] = _product_summary(product)
        ak[key]["features"] = product.features

    return desc, ak


def _shipping_section(
    goal_world: list[dict],
    entities: TaskEntitySet,
) -> tuple[str | None, dict]:
    shipping_products = [
        i for i in range(1, 4) if _goal_has(goal_world, f"p{i}.shipping_checked", True)
    ]
    if not shipping_products:
        return None, {}

    zip_code = entities.zip_code
    if len(shipping_products) == 1:
        desc = (
            f"On the product detail page, check the shipping and delivery information "
            f"for ZIP code {zip_code}. Note the estimated delivery date and shipping cost."
        )
    else:
        desc = (
            f"For each product, check the shipping and delivery information "
            f"for ZIP code {zip_code}. Note the estimated delivery date and shipping cost."
        )

    ak = {}
    for i in shipping_products:
        idx = i - 1
        if idx >= len(entities.products):
            continue
        product = entities.products[idx]
        s = product.shipping
        ship: dict[str, Any] = {"zip_code": zip_code}
        if product.prime_eligible:
            ship["delivery"] = (
                f"FREE Prime delivery in {s.prime_delivery_days} day{'s' if s.prime_delivery_days > 1 else ''}"
            )
            ship["cost"] = "$0.00"
        elif s.cost_cents == 0:
            ship["delivery"] = f"FREE delivery in {s.delivery_days} days"
            ship["cost"] = "$0.00"
        else:
            ship["delivery"] = f"{s.delivery_days} days"
            ship["cost"] = _fmt_price(s.cost_cents)
        key = f"product_{i}"
        ak.setdefault(key, {})["shipping"] = ship

    return desc, ak


def _reviews_section(
    goal_world: list[dict], entities: TaskEntitySet
) -> tuple[str | None, dict]:
    review_products = [
        i for i in range(1, 4) if _goal_has(goal_world, f"p{i}.reviews_read", True)
    ]
    if not review_products:
        return None, {}

    if len(review_products) == 1:
        desc = (
            "Read the customer reviews section. Note the overall sentiment, "
            "common praise, and any recurring complaints."
        )
    else:
        desc = (
            "For each product, read the customer reviews. Note the overall sentiment, "
            "the top positive and negative reviews, and any patterns across products."
        )

    ak = {}
    for i in review_products:
        idx = i - 1
        if idx >= len(entities.products):
            continue
        product = entities.products[idx]
        reviews = {
            "overall_rating": product.rating,
            "total_reviews": product.review_count,
            "reviews": [
                {
                    "reviewer": r.reviewer_name,
                    "rating": r.rating,
                    "title": r.title,
                    "verified": r.verified_purchase,
                }
                for r in product.reviews
            ],
        }
        key = f"product_{i}"
        ak.setdefault(key, {})["reviews"] = reviews

    return desc, ak


def _seller_section(
    goal_world: list[dict], entities: TaskEntitySet
) -> tuple[str | None, dict]:
    seller_products = [
        i for i in range(1, 3) if _goal_has(goal_world, f"p{i}.seller_checked", True)
    ]
    if not seller_products:
        return None, {}

    desc = (
        "Check the seller information: seller name, rating, feedback percentage, "
        "and whether it's fulfilled by Amazon."
    )

    ak = {}
    for i in seller_products:
        idx = i - 1
        if idx >= len(entities.products):
            continue
        s = entities.products[idx].seller
        seller = {
            "name": s.name,
            "rating": s.rating,
            "positive_feedback_pct": s.positive_feedback_pct,
            "ships_from": s.ships_from,
            "fulfilled_by_amazon": s.fulfilled_by_amazon,
        }
        key = f"product_{i}"
        ak.setdefault(key, {})["seller"] = seller

    return desc, ak


def _variants_section(
    goal_world: list[dict], entities: TaskEntitySet
) -> tuple[str | None, dict]:
    if not (
        _goal_has(goal_world, "p1.variant_a_viewed", True)
        or _goal_has(goal_world, "p1.variant_checked", True)
    ):
        return None, {}

    desc = (
        "Check the available product variants (size, color, storage options) "
        "and note any price differences between configurations."
    )

    ak = {}
    if entities.products and entities.products[0].variants:
        v = entities.products[0].variants
        options = []
        for i, opt in enumerate(v.options):
            delta = v.price_deltas_cents[i] if i < len(v.price_deltas_cents) else 0
            price = entities.products[0].price_cents + delta
            options.append(
                {
                    "option": opt,
                    "price": _fmt_price(price),
                    "price_delta": _fmt_price(delta) if delta != 0 else "$0.00",
                }
            )
        ak.setdefault("product_1", {})["variants"] = {
            "type": v.variant_type,
            "options": options,
        }

    return desc, ak


def _qa_section(
    goal_world: list[dict], entities: TaskEntitySet
) -> tuple[str | None, dict]:
    qa_products = [
        i for i in range(1, 3) if _goal_has(goal_world, f"p{i}.qa_read", True)
    ]
    if not qa_products:
        return None, {}

    if len(qa_products) == 1:
        desc = (
            "Read the Questions & Answers section on the product page. "
            "Look for answers about compatibility, sizing, and common concerns."
        )
    else:
        desc = (
            "For each product, check the Questions & Answers section for information "
            "about compatibility, sizing, and common buyer questions."
        )

    ak = {}
    for i in qa_products:
        idx = i - 1
        if idx >= len(entities.products):
            continue
        qa = [
            {"question": q.question, "answer": q.answer, "votes": q.votes}
            for q in entities.products[idx].qa_pairs
        ]
        key = f"product_{i}"
        ak.setdefault(key, {})["qa"] = qa

    return desc, ak


def _cart_section(
    goal_world: list[dict], entities: TaskEntitySet
) -> tuple[str | None, dict]:
    cart_products = [
        i for i in range(1, 3) if _goal_has(goal_world, f"p{i}.added_to_cart", True)
    ]
    if not cart_products:
        return None, {}

    parts = []
    if len(cart_products) == 1:
        parts.append("Add the product to your cart.")
    else:
        parts.append(f"Add all {len(cart_products)} products to your cart.")
    if _goal_has(goal_world, "cart.read", True):
        parts.append(
            "Open the cart and verify the items, quantities, individual prices, and cart total."
        )
    desc = " ".join(parts)

    ak_items = []
    subtotal = 0
    for i in cart_products:
        idx = i - 1
        if idx < len(entities.products):
            p = entities.products[idx]
            ak_items.append({"name": p.name, "price": _fmt_price(p.price_cents)})
            subtotal += p.price_cents

    ak = {"cart": {"items": ak_items, "subtotal": _fmt_price(subtotal)}}
    return desc, ak


def _submission_section(
    goal_world: list[dict],
    config: dict[str, Any],
    entities: TaskEntitySet,
) -> tuple[str, dict]:
    """Submission instructions — always present. Returns description only (answer key built above)."""
    entry_point = config.get("task.entry_point", "search")
    num_products = config.get("task.num_products", 0)

    if _goal_has(goal_world, "cart.read", True):
        desc = (
            "Submit your findings including: each product's name, price, and "
            "the cart total. Note any discrepancies between listed prices and the cart total."
        )
    elif num_products >= 2:
        desc = (
            "Compile a comparison of the products and submit your findings. "
            "Include: product name, price, rating, seller, and any other "
            "relevant details you gathered for each product."
        )
    elif entry_point == "deals":
        desc = (
            "Submit your findings including: deal product name, original price, "
            "deal price, discount percentage, and deal type."
        )
    elif entry_point == "category":
        desc = (
            "Submit your findings with the product names, prices, and ratings "
            "from the category listings."
        )
    elif num_products == 1:
        desc = (
            "Submit your findings with the product name, price, rating, "
            "and all the details you gathered."
        )
    else:
        desc = (
            "Submit your findings with the product names, prices, and ratings "
            "from the search results."
        )
    return desc, {}


# ---------------------------------------------------------------------------
# Main unified generator
# ---------------------------------------------------------------------------


def generate_task_spec(
    task_id: str,
    start_world: dict[str, Any],
    goal_world: list[dict],
    entities: TaskEntitySet,
) -> tuple[str, dict[str, Any]]:
    """Generate aligned task description and answer key in one pass.

    Returns:
        (description, answer_key) — guaranteed to be aligned.
        Every piece of information the description asks about is
        present in the answer key.
    """
    entry_point = start_world.get("task.entry_point", "search")
    config = start_world

    # Build sections — each returns (description_fragment, answer_key_fragment)
    sections = [
        _opening(entities, entry_point),
        _category_section(goal_world, entities),
        _deals_section(goal_world, entities),
        _results_section(goal_world, entities),
        _filter_section(goal_world, entities),
        _refine_section(goal_world),
        _product_detail_section(goal_world, entities, config),
        _shipping_section(goal_world, entities),
        _reviews_section(goal_world, entities),
        _seller_section(goal_world, entities),
        _variants_section(goal_world, entities),
        _qa_section(goal_world, entities),
        _cart_section(goal_world, entities),
        _submission_section(goal_world, config, entities),
    ]

    # Merge descriptions and answer keys
    desc_parts: list[str] = []
    answer_key: dict[str, Any] = {"task_type": entry_point}

    for desc_frag, ak_frag in sections:
        if desc_frag:
            desc_parts.append(desc_frag)
        if ak_frag:
            # Deep merge: product_1 fields accumulate across sections
            for key, value in ak_frag.items():
                if (
                    key.startswith("product_")
                    and key in answer_key
                    and isinstance(value, dict)
                ):
                    answer_key[key].update(value)
                else:
                    answer_key[key] = value

    description = " ".join(desc_parts)
    return description, answer_key
