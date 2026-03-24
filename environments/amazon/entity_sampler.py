"""Deterministic entity sampler for Amazon Shopping domain.

Given a task_id and its start_world config, produces a TaskEntitySet
with realistic products, reviews, sellers, Q&A pairs, deals, and
categories. The same task_id always produces the same entities.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from entities import (
    Category,
    Deal,
    DealType,
    Product,
    ProductCondition,
    ProductVariant,
    QAPair,
    Review,
    Seller,
    ShippingOption,
    TaskEntitySet,
)


# ---------------------------------------------------------------------------
# Product catalog — realistic Amazon product data
# ---------------------------------------------------------------------------

_PRODUCT_CATALOG: list[dict[str, Any]] = [
    # Electronics
    {
        "name": "Sony WH-1000XM5 Wireless Headphones",
        "brand": "Sony",
        "cat": "Electronics",
        "sub": "Headphones",
        "price": 34800,
        "features": [
            "Industry-leading noise cancellation",
            "30-hour battery life",
            "Multipoint connection",
            "Speak-to-Chat auto pause",
        ],
    },
    {
        "name": "Apple AirPods Pro (2nd Gen)",
        "brand": "Apple",
        "cat": "Electronics",
        "sub": "Headphones",
        "price": 24900,
        "features": [
            "Active noise cancellation",
            "Adaptive transparency",
            "USB-C charging",
            "6 hours listening time",
        ],
    },
    {
        "name": "JBL Tune 510BT Wireless Headphones",
        "brand": "JBL",
        "cat": "Electronics",
        "sub": "Headphones",
        "price": 2995,
        "features": [
            "40-hour battery",
            "JBL Pure Bass sound",
            "Lightweight design",
            "Multi-point connection",
        ],
    },
    {
        "name": "Samsung Galaxy Buds2 Pro",
        "brand": "Samsung",
        "cat": "Electronics",
        "sub": "Headphones",
        "price": 17999,
        "features": [
            "24-bit Hi-Fi audio",
            "ANC with 3 mics",
            "IPX7 water resistance",
            "Intelligent ANC",
        ],
    },
    {
        "name": "Anker Soundcore Life Q20+",
        "brand": "Anker",
        "cat": "Electronics",
        "sub": "Headphones",
        "price": 4999,
        "features": [
            "Hybrid ANC",
            "40-hour playtime",
            "Hi-Res Audio",
            "Memory foam earcups",
        ],
    },
    {
        "name": "Logitech MX Master 3S Wireless Mouse",
        "brand": "Logitech",
        "cat": "Electronics",
        "sub": "Mice",
        "price": 9999,
        "features": [
            "8K DPI sensor",
            "Quiet clicks",
            "USB-C charging",
            "Multi-device support",
        ],
    },
    {
        "name": "Razer DeathAdder V3 Pro",
        "brand": "Razer",
        "cat": "Electronics",
        "sub": "Mice",
        "price": 8999,
        "features": [
            "Focus Pro 30K sensor",
            "63g ultralight",
            "90-hour battery",
            "HyperSpeed wireless",
        ],
    },
    {
        "name": "Apple Magic Mouse",
        "brand": "Apple",
        "cat": "Electronics",
        "sub": "Mice",
        "price": 7900,
        "features": [
            "Multi-touch surface",
            "Lightning rechargeable",
            "Bluetooth",
            "Optimized foot design",
        ],
    },
    {
        "name": "Samsung 990 EVO 2TB NVMe SSD",
        "brand": "Samsung",
        "cat": "Electronics",
        "sub": "Storage",
        "price": 14999,
        "features": [
            "PCIe 5.0 x2",
            "Up to 7,250 MB/s read",
            "TLC V-NAND",
            "5-year warranty",
        ],
    },
    {
        "name": "SanDisk 1TB Extreme Portable SSD",
        "brand": "SanDisk",
        "cat": "Electronics",
        "sub": "Storage",
        "price": 8999,
        "features": [
            "1050 MB/s read",
            "IP65 water/dust resistant",
            "USB 3.2 Gen 2",
            "Carabiner loop",
        ],
    },
    {
        "name": "Anker 737 Power Bank 24000mAh",
        "brand": "Anker",
        "cat": "Electronics",
        "sub": "Chargers",
        "price": 10999,
        "features": [
            "140W max output",
            "Smart digital display",
            "24,000mAh capacity",
            "USB-C PD 3.1",
        ],
    },
    {
        "name": "Apple 20W USB-C Power Adapter",
        "brand": "Apple",
        "cat": "Electronics",
        "sub": "Chargers",
        "price": 1900,
        "features": [
            "20W fast charging",
            "USB-C connector",
            "Compatible with iPhone/iPad",
            "Compact design",
        ],
    },
    # Home & Kitchen
    {
        "name": "Ninja Professional Plus Blender",
        "brand": "Ninja",
        "cat": "Home & Kitchen",
        "sub": "Blenders",
        "price": 8999,
        "features": [
            "1400-peak-watt motor",
            "72oz pitcher",
            "4 Auto-iQ programs",
            "Dishwasher safe",
        ],
    },
    {
        "name": "Instant Pot Duo 7-in-1 Electric Pressure Cooker",
        "brand": "Instant Pot",
        "cat": "Home & Kitchen",
        "sub": "Cookers",
        "price": 8995,
        "features": [
            "7-in-1 functionality",
            "6-quart capacity",
            "13 one-touch programs",
            "Stainless steel inner pot",
        ],
    },
    {
        "name": "COSORI Air Fryer Pro LE 5-Qt",
        "brand": "COSORI",
        "cat": "Home & Kitchen",
        "sub": "Air Fryers",
        "price": 6999,
        "features": [
            "5-quart capacity",
            "9 one-touch functions",
            "Shake reminder",
            "Dishwasher-safe basket",
        ],
    },
    {
        "name": "Keurig K-Supreme Coffee Maker",
        "brand": "Keurig",
        "cat": "Home & Kitchen",
        "sub": "Coffee Makers",
        "price": 14999,
        "features": [
            "MultiStream technology",
            "5 brew sizes",
            "66oz reservoir",
            "Over ice setting",
        ],
    },
    {
        "name": "iRobot Roomba j7+ Robot Vacuum",
        "brand": "iRobot",
        "cat": "Home & Kitchen",
        "sub": "Vacuums",
        "price": 59900,
        "features": [
            "PrecisionVision navigation",
            "Auto-empty base",
            "Avoids pet waste",
            "Smart mapping",
        ],
    },
    # Sports & Outdoors
    {
        "name": "Hydro Flask 32oz Wide Mouth Water Bottle",
        "brand": "Hydro Flask",
        "cat": "Sports & Outdoors",
        "sub": "Water Bottles",
        "price": 4495,
        "features": [
            "TempShield insulation",
            "BPA-free",
            "Powder coated",
            "Lifetime warranty",
        ],
    },
    {
        "name": "Fitbit Charge 6 Fitness Tracker",
        "brand": "Fitbit",
        "cat": "Sports & Outdoors",
        "sub": "Fitness Trackers",
        "price": 15995,
        "features": [
            "Built-in GPS",
            "Heart rate monitoring",
            "7-day battery",
            "Google integration",
        ],
    },
    {
        "name": "YETI Rambler 30oz Tumbler",
        "brand": "YETI",
        "cat": "Sports & Outdoors",
        "sub": "Drinkware",
        "price": 3800,
        "features": [
            "Double-wall vacuum insulation",
            "DuraCoat color",
            "MagSlider lid",
            "Dishwasher safe",
        ],
    },
    # Books & Office
    {
        "name": "Kindle Paperwhite (16GB)",
        "brand": "Amazon",
        "cat": "Electronics",
        "sub": "E-Readers",
        "price": 14999,
        "features": [
            "6.8-inch display",
            "Adjustable warm light",
            "Waterproof IPX8",
            "10 weeks battery",
        ],
    },
    {
        "name": "Rocketbook Smart Reusable Notebook",
        "brand": "Rocketbook",
        "cat": "Office Products",
        "sub": "Notebooks",
        "price": 2799,
        "features": [
            "Reusable pages",
            "Cloud connected",
            "AI handwriting recognition",
            "Includes Pilot FriXion pen",
        ],
    },
    # Clothing & Accessories
    {
        "name": "Columbia Men's Watertight II Rain Jacket",
        "brand": "Columbia",
        "cat": "Clothing",
        "sub": "Jackets",
        "price": 5490,
        "features": [
            "Omni-Tech waterproof",
            "Packable hood",
            "Zippered hand pockets",
            "Adjustable cuffs",
        ],
    },
    {
        "name": "Hanes Men's EcoSmart Hoodie (6-Pack)",
        "brand": "Hanes",
        "cat": "Clothing",
        "sub": "Hoodies",
        "price": 4200,
        "features": [
            "50% cotton/50% polyester",
            "Pill-resistant fleece",
            "Front pouch pocket",
            "Ribbed cuffs",
        ],
    },
    {
        "name": "New Balance 574 Core Sneakers",
        "brand": "New Balance",
        "cat": "Clothing",
        "sub": "Shoes",
        "price": 7999,
        "features": [
            "ENCAP midsole",
            "Suede/mesh upper",
            "Classic silhouette",
            "Rubber outsole",
        ],
    },
    # Toys & Games
    {
        "name": "LEGO Technic McLaren Formula 1 Race Car",
        "brand": "LEGO",
        "cat": "Toys & Games",
        "sub": "Building Sets",
        "price": 18999,
        "features": [
            "1,432 pieces",
            "V6 engine with moving pistons",
            "DRS system",
            "Collectible display model",
        ],
    },
    {
        "name": "Nintendo Switch Pro Controller",
        "brand": "Nintendo",
        "cat": "Toys & Games",
        "sub": "Gaming",
        "price": 6499,
        "features": [
            "40-hour battery",
            "Amiibo compatible",
            "Motion controls",
            "HD Rumble",
        ],
    },
    # Health & Personal Care
    {
        "name": "Oral-B iO Series 9 Electric Toothbrush",
        "brand": "Oral-B",
        "cat": "Health",
        "sub": "Oral Care",
        "price": 19999,
        "features": [
            "AI-powered brushing",
            "7 smart modes",
            "3D teeth tracking",
            "Magnetic charging",
        ],
    },
    {
        "name": "Theragun Mini 2.0 Massage Gun",
        "brand": "Therabody",
        "cat": "Health",
        "sub": "Recovery",
        "price": 19900,
        "features": [
            "3 speeds",
            "QuietForce technology",
            "150-min battery",
            "Portable design",
        ],
    },
]

_REVIEWER_NAMES = [
    "J. Smith",
    "Sarah M.",
    "TechReviewer42",
    "Mike D.",
    "A. Johnson",
    "Lisa K.",
    "ProductTester",
    "Dave W.",
    "Maria G.",
    "Chris L.",
    "RealBuyer2025",
    "HomeCook99",
    "FitnessFan",
    "GadgetGuru",
    "Karen P.",
    "BudgetShopper",
    "QualityMatters",
    "Alex T.",
    "Pat R.",
    "ThriftyBuyer",
]

_REVIEW_TEMPLATES_POSITIVE = [
    (
        "Excellent product!",
        "Works exactly as described. Very happy with this purchase. {feature} is the standout feature.",
    ),
    (
        "Great value",
        "For the price, you can't beat this. {feature} really makes a difference. Would buy again.",
    ),
    (
        "Highly recommended",
        "I've been using this for a month now and it's been fantastic. {feature} works perfectly.",
    ),
    (
        "Five stars!",
        "This exceeded my expectations. The build quality is great and {feature} is impressive.",
    ),
    (
        "Perfect for my needs",
        "Exactly what I was looking for. {feature} was the deciding factor and it delivers.",
    ),
]

_REVIEW_TEMPLATES_NEGATIVE = [
    (
        "Disappointed",
        "Expected better for the price. {feature} didn't work as advertised. Returning this.",
    ),
    (
        "Not worth it",
        "Cheaply made and {feature} is mediocre at best. Save your money.",
    ),
    (
        "Broke after a week",
        "Stopped working after 7 days. {feature} was fine initially but quality is poor.",
    ),
]

_REVIEW_TEMPLATES_MIXED = [
    (
        "Good but not great",
        "{feature} works well but the overall build quality could be better. 3.5 stars rounded up.",
    ),
    (
        "Decent for the price",
        "It does the job. {feature} is adequate but nothing special. Good budget option.",
    ),
]

_QA_TEMPLATES = [
    (
        "Does this work with {compat}?",
        "Yes, it is compatible with {compat}. I've been using it without issues.",
    ),
    (
        "Is {feature} really as good as described?",
        "In my experience, {feature} works as advertised. Very satisfied.",
    ),
    (
        "How long does the battery actually last?",
        "I get about {hours} hours of use on a full charge, which matches the specs.",
    ),
    (
        "Does this come with a warranty?",
        "Yes, it comes with a {warranty} warranty from the manufacturer.",
    ),
    (
        "Is this the latest model?",
        "Yes, this is the current 2025 model. Make sure the listing says '{brand}' as the seller.",
    ),
    (
        "Will this fit in {container}?",
        "It should fit — the dimensions are listed on the product page. I'd measure first.",
    ),
    (
        "Is the {color} color accurate to the photos?",
        "Pretty close to the photos. The {color} version looks great in person.",
    ),
    (
        "Can this be used for {use_case}?",
        "Absolutely, I use mine for {use_case} regularly and it works great.",
    ),
]

_COMPAT_OPTIONS = [
    "iPhone 15",
    "MacBook Pro",
    "Samsung Galaxy S24",
    "iPad Pro",
    "Windows 11",
    "USB-C devices",
    "Bluetooth 5.3",
]
_COLOR_OPTIONS = ["Black", "White", "Navy Blue", "Silver", "Red", "Forest Green"]
_CONTAINER_OPTIONS = [
    "a backpack",
    "a carry-on bag",
    "a desk drawer",
    "a kitchen cabinet",
]
_USE_CASE_OPTIONS = [
    "daily commuting",
    "office work",
    "outdoor activities",
    "travel",
    "home workouts",
    "meal prep",
]

_SELLER_NAMES = [
    ("TechDirect USA", 4.7, 96, "USA", True),
    ("BestDeals Electronics", 4.3, 91, "USA", False),
    ("PrimeShip Global", 4.8, 98, "USA", True),
    ("ValueMart Online", 4.1, 88, "China", False),
    ("GadgetWorld", 4.5, 94, "USA", True),
    ("HomeGoods Plus", 4.6, 95, "USA", True),
    ("BudgetFinds", 3.9, 85, "China", False),
    ("TopBrand Store", 4.9, 99, "USA", True),
    ("QuickShip Deals", 4.2, 90, "USA", False),
    ("MegaSave Outlet", 4.0, 87, "USA", False),
]

_CATEGORY_TREE = {
    "Electronics": ["Headphones", "Mice", "Storage", "Chargers", "E-Readers"],
    "Home & Kitchen": ["Blenders", "Cookers", "Air Fryers", "Coffee Makers", "Vacuums"],
    "Sports & Outdoors": ["Water Bottles", "Fitness Trackers", "Drinkware"],
    "Office Products": ["Notebooks"],
    "Clothing": ["Jackets", "Hoodies", "Shoes"],
    "Toys & Games": ["Building Sets", "Gaming"],
    "Health": ["Oral Care", "Recovery"],
}

_ZIP_CODES = ["10001", "90210", "60601", "98101", "02101", "30301", "85001", "93722"]


# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------


def _stable_rng(seed_str: str) -> random.Random:
    h = int(hashlib.sha256(seed_str.encode()).hexdigest()[:16], 16)
    return random.Random(h)


# ---------------------------------------------------------------------------
# Entity generators
# ---------------------------------------------------------------------------


def _make_seller(rng: random.Random, idx: int) -> Seller:
    name, rating, feedback, ships_from, fba = rng.choice(_SELLER_NAMES)
    return Seller(
        seller_id=f"S{idx:04d}",
        name=name,
        rating=rating,
        positive_feedback_pct=feedback,
        ships_from=ships_from,
        fulfilled_by_amazon=fba,
    )


def _make_shipping(rng: random.Random, zip_code: str, prime: bool) -> ShippingOption:
    if prime:
        return ShippingOption(
            zip_code=zip_code,
            delivery_days=rng.choice([1, 2, 2, 3]),
            cost_cents=0,
            prime_eligible=True,
            prime_delivery_days=rng.choice([1, 1, 2]),
        )
    else:
        return ShippingOption(
            zip_code=zip_code,
            delivery_days=rng.choice([3, 5, 7, 10, 14]),
            cost_cents=rng.choice([0, 499, 599, 799, 999, 1299]),
            prime_eligible=False,
        )


_CATEGORY_VARIANT_TYPES: dict[str, list[str]] = {
    "Electronics": ["color", "storage", "style"],
    "Home & Kitchen": ["color", "size", "style"],
    "Sports & Outdoors": ["color", "size"],
    "Clothing": ["color", "size"],
    "Toys & Games": ["color", "style"],
    "Health & Household": ["size", "style"],
    "Office Products": ["color", "style"],
}


def _make_variants(rng: random.Random, product_cat: dict) -> ProductVariant:
    category = product_cat.get("cat", "")
    allowed = _CATEGORY_VARIANT_TYPES.get(category, ["color", "size", "style"])
    vtype = rng.choice(allowed)
    if vtype == "color":
        opts = rng.sample(_COLOR_OPTIONS, k=min(4, len(_COLOR_OPTIONS)))
        deltas = [0] * len(opts)
    elif vtype == "size":
        opts = rng.choice(
            [
                ["Small", "Medium", "Large", "X-Large"],
                ["6-inch", "8-inch", "10-inch"],
            ]
        )
        deltas = [i * rng.choice([500, 1000, 2000, 5000]) for i in range(len(opts))]
    elif vtype == "storage":
        opts = ["128GB", "256GB", "512GB", "1TB"]
        deltas = [0, 5000, 15000, 30000]
    else:
        opts = ["Standard", "Premium", "Pro"]
        deltas = [0, 3000, 8000]
    return ProductVariant(variant_type=vtype, options=opts, price_deltas_cents=deltas)


def _make_reviews(rng: random.Random, product: dict, count: int) -> list[Review]:
    reviews = []
    feature = rng.choice(product.get("features", ["the product"]))
    for i in range(count):
        if rng.random() < 0.6:
            title, text_tpl = rng.choice(_REVIEW_TEMPLATES_POSITIVE)
            rating = rng.choice([4.0, 4.0, 5.0, 5.0, 5.0])
        elif rng.random() < 0.5:
            title, text_tpl = rng.choice(_REVIEW_TEMPLATES_MIXED)
            rating = rng.choice([3.0, 3.0, 4.0])
        else:
            title, text_tpl = rng.choice(_REVIEW_TEMPLATES_NEGATIVE)
            rating = rng.choice([1.0, 2.0, 2.0])
        text = text_tpl.format(feature=feature)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        reviews.append(
            Review(
                reviewer_name=rng.choice(_REVIEWER_NAMES),
                rating=rating,
                title=title,
                text=text,
                date=f"2025-{month:02d}-{day:02d}",
                verified_purchase=rng.random() > 0.15,
            )
        )
    return reviews


def _make_qa_pairs(rng: random.Random, product: dict, count: int) -> list[QAPair]:
    pairs = []
    brand = product.get("brand", "the manufacturer")
    feature = rng.choice(product.get("features", ["the product"]))
    # Sample without replacement to avoid duplicate questions
    available = list(_QA_TEMPLATES)
    rng.shuffle(available)
    count = min(count, len(available))
    for i in range(count):
        q_tpl, a_tpl = available[i]
        q = q_tpl.format(
            compat=rng.choice(_COMPAT_OPTIONS),
            feature=feature,
            hours=rng.choice([6, 8, 10, 12, 20, 30]),
            warranty=rng.choice(["1-year", "2-year", "limited lifetime"]),
            brand=brand,
            container=rng.choice(_CONTAINER_OPTIONS),
            color=rng.choice(_COLOR_OPTIONS),
            use_case=rng.choice(_USE_CASE_OPTIONS),
        )
        a = a_tpl.format(
            compat=rng.choice(_COMPAT_OPTIONS),
            feature=feature,
            hours=rng.choice([6, 8, 10, 12, 20, 30]),
            warranty=rng.choice(["1-year", "2-year", "limited lifetime"]),
            brand=brand,
            container=rng.choice(_CONTAINER_OPTIONS),
            color=rng.choice(_COLOR_OPTIONS),
            use_case=rng.choice(_USE_CASE_OPTIONS),
        )
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        pairs.append(
            QAPair(
                question=q,
                answer=a,
                votes=rng.randint(0, 50),
                date=f"2025-{month:02d}-{day:02d}",
            )
        )
    return pairs


def _make_product(
    rng: random.Random,
    catalog_entry: dict,
    idx: int,
    zip_code: str,
    include_reviews: bool = True,
    include_qa: bool = False,
    include_variants: bool = False,
) -> Product:
    base_price = catalog_entry["price"]
    # Price jitter: ±15%
    price = int(base_price * rng.uniform(0.85, 1.15))
    # Occasional discount
    list_price = None
    if rng.random() < 0.35:
        list_price = int(price * rng.uniform(1.1, 1.5))

    prime = rng.random() < 0.6

    review_count = rng.randint(50, 5000) if include_reviews else rng.randint(10, 200)
    rating = round(rng.uniform(3.2, 4.9), 1)

    reviews = (
        _make_reviews(rng, catalog_entry, rng.randint(3, 6)) if include_reviews else []
    )
    qa = _make_qa_pairs(rng, catalog_entry, rng.randint(3, 5)) if include_qa else []
    variants = _make_variants(rng, catalog_entry) if include_variants else None

    asin = f"B{rng.randint(10000000, 99999999):08d}"

    return Product(
        product_id=f"P{idx:04d}",
        name=catalog_entry["name"],
        brand=catalog_entry["brand"],
        category=catalog_entry["cat"],
        subcategory=catalog_entry["sub"],
        price_cents=price,
        list_price_cents=list_price,
        condition=ProductCondition.NEW,
        rating=rating,
        review_count=review_count,
        prime_eligible=prime,
        asin=asin,
        image_description=f"Product image of {catalog_entry['name']}",
        features=catalog_entry.get("features", []),
        seller=_make_seller(rng, idx),
        shipping=_make_shipping(rng, zip_code, prime),
        variants=variants,
        reviews=reviews,
        qa_pairs=qa,
    )


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------


def sample_entities(task_id: str, start_world: dict[str, Any]) -> TaskEntitySet:
    """Generate a deterministic TaskEntitySet for a task.

    Args:
        task_id: Unique task identifier (used as RNG seed).
        start_world: Flat dict of start_world fields from the sampled task.

    Returns:
        TaskEntitySet with products, deals, categories, and search context.
    """
    rng = _stable_rng(task_id)
    zip_code = rng.choice(_ZIP_CODES)

    # Determine what entities we need from task config
    num_products = start_world.get("task.num_products", 1)
    requires_reviews = start_world.get("task.requires_reviews", False)
    requires_qa = start_world.get("task.requires_qa", False)
    requires_variants = start_world.get("task.requires_variants", False)
    entry_point = start_world.get("task.entry_point", "search")

    # Pick a product subcategory to focus the search
    available = list(_PRODUCT_CATALOG)
    rng.shuffle(available)

    # Pick primary category for category-browse tasks
    cat_name = rng.choice(list(_CATEGORY_TREE.keys()))

    # For category/deals tasks, bias toward that category
    if entry_point == "category":
        cat_products = [p for p in available if p["cat"] == cat_name]
        other_products = [p for p in available if p["cat"] != cat_name]
        available = cat_products + other_products

    # Generate products (at least 5 for search results, more for detail)
    product_count = max(5, num_products + 3) if num_products > 0 else 5
    products = []
    for i in range(min(product_count, len(available))):
        p = _make_product(
            rng,
            available[i],
            i + 1,
            zip_code,
            include_reviews=requires_reviews,
            include_qa=requires_qa,
            include_variants=(
                requires_variants and i == 0
            ),  # variants only on first product
        )
        products.append(p)

    # Sort by price for consistent search results, but keep the first product
    # (which may have variants/reviews/qa) pinned at index 0 so that tasks
    # referencing "product 1" always find those features at the expected index.
    if len(products) > 1:
        first = products[0]
        rest = sorted(products[1:], key=lambda p: p.price_cents)
        products = [first] + rest

    # Generate deals for deal-hunt tasks
    deals = []
    if entry_point == "deals":
        for i, p in enumerate(products[:4]):
            deal_price = int(p.price_cents * rng.uniform(0.5, 0.8))
            deals.append(
                Deal(
                    deal_id=f"D{i + 1:04d}",
                    product=p,
                    deal_type=rng.choice(list(DealType)),
                    deal_price_cents=deal_price,
                    original_price_cents=p.price_cents,
                    time_remaining_minutes=rng.randint(30, 480),
                )
            )

    # Generate categories
    categories = []
    for cat, subs in _CATEGORY_TREE.items():
        categories.append(
            Category(
                category_id=cat.lower().replace(" & ", "_").replace(" ", "_"),
                name=cat,
                subcategories=subs,
                product_count=rng.randint(100, 10000),
            )
        )

    # Generate search query based on products
    if products:
        p0 = products[0]
        search_queries = [
            p0.name.split("(")[0].strip() if "(" in p0.name else p0.name[:40],
            f"{p0.brand} {p0.subcategory.lower()}",
            p0.subcategory.lower(),
            f"best {p0.subcategory.lower()} {rng.choice(['2025', 'under $100', 'wireless', 'portable'])}",
        ]
        search_query = rng.choice(search_queries)
    else:
        search_query = "wireless headphones"

    return TaskEntitySet(
        task_id=task_id,
        products=products,
        deals=deals,
        categories=categories,
        search_query=search_query,
        zip_code=zip_code,
    )
