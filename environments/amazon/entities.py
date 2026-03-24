"""Entity definitions for Amazon Shopping domain.

These model the data that populates synthetic Amazon pages:
products, reviews, sellers, Q&A pairs, categories, and deals.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProductCondition(StrEnum):
    NEW = "new"
    USED_LIKE_NEW = "used_like_new"
    USED_GOOD = "used_good"
    RENEWED = "renewed"


class DealType(StrEnum):
    LIGHTNING = "lightning_deal"
    DAILY = "deal_of_the_day"
    COUPON = "coupon"


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class ShippingOption(BaseModel):
    zip_code: str
    delivery_days: int
    cost_cents: int  # 0 = free shipping
    prime_eligible: bool = False
    prime_delivery_days: Optional[int] = None


class ProductVariant(BaseModel):
    variant_type: str  # "color", "size", "storage", "style"
    options: list[str]  # ["Black", "White", "Navy"]
    price_deltas_cents: list[int]  # [0, 0, 500] — price offset per option


class Seller(BaseModel):
    seller_id: str
    name: str
    rating: float  # 1.0–5.0
    positive_feedback_pct: int  # 0–100
    ships_from: str  # "USA", "China", etc.
    fulfilled_by_amazon: bool = False


class Review(BaseModel):
    reviewer_name: str
    rating: float  # 1–5
    title: str
    text: str
    date: str  # "2025-11-15"
    verified_purchase: bool = True


class QAPair(BaseModel):
    question: str
    answer: str
    votes: int = 0
    date: str = ""


class Product(BaseModel):
    product_id: str
    name: str
    brand: str
    category: str
    subcategory: str
    price_cents: int
    list_price_cents: Optional[int] = None  # original/MSRP if on sale
    condition: ProductCondition = ProductCondition.NEW
    rating: float = 4.0  # 1.0–5.0
    review_count: int = 0
    prime_eligible: bool = False
    asin: str = ""  # Amazon Standard ID
    image_description: str = ""
    features: list[str] = Field(default_factory=list)  # bullet points
    seller: Optional[Seller] = None
    shipping: Optional[ShippingOption] = None
    variants: Optional[ProductVariant] = None
    reviews: list[Review] = Field(default_factory=list)
    qa_pairs: list[QAPair] = Field(default_factory=list)

    @property
    def price_dollars(self) -> str:
        return f"${self.price_cents / 100:.2f}"

    @property
    def has_discount(self) -> bool:
        return (
            self.list_price_cents is not None
            and self.list_price_cents > self.price_cents
        )

    @property
    def discount_pct(self) -> int:
        if not self.has_discount:
            return 0
        return round((1 - self.price_cents / self.list_price_cents) * 100)


class Deal(BaseModel):
    deal_id: str
    product: Product
    deal_type: DealType
    deal_price_cents: int
    original_price_cents: int
    time_remaining_minutes: int = 120

    @property
    def discount_pct(self) -> int:
        return round((1 - self.deal_price_cents / self.original_price_cents) * 100)


class Category(BaseModel):
    category_id: str
    name: str
    subcategories: list[str] = Field(default_factory=list)
    product_count: int = 0


class CartItem(BaseModel):
    product: Product
    quantity: int = 1

    @property
    def line_total_cents(self) -> int:
        return self.product.price_cents * self.quantity


# ---------------------------------------------------------------------------
# Task entity set — everything needed to render pages for one task
# ---------------------------------------------------------------------------


class TaskEntitySet(BaseModel):
    """All entities needed to render pages for a single task.

    Deterministically generated from task_id by the entity sampler.
    """

    task_id: str
    products: list[Product] = Field(default_factory=list)
    deals: list[Deal] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)
    search_query: str = ""
    zip_code: str = ""
