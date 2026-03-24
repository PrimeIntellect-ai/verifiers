"use client";

import { useState, useEffect, useCallback } from "react";

export const dynamic = "force-dynamic";

interface Product {
  name: string;
  brand: string;
  price_cents: number;
  list_price_cents?: number;
  rating: number;
  review_count: number;
  prime_eligible: boolean;
  features: string[];
  asin: string;
  seller: { name: string; rating: number; total_ratings?: number; positive_feedback_pct: number; ships_from?: string; fulfilled_by_amazon?: boolean };
  shipping: { cost_cents: number; delivery_days: number; prime_delivery_days: number };
  variants?: { type: string; options: string[]; price_deltas_cents: number[] };
  reviews: { reviewer?: string; reviewer_name?: string; rating: number; title: string; text: string; date: string; verified?: boolean; verified_purchase?: boolean }[];
  qa_pairs: { question: string; answer: string; votes: number }[];
}

function fmt(cents: number) {
  return `$${Math.floor(cents / 100)}.${(cents % 100).toString().padStart(2, "0")}`;
}

function stars(r: number) {
  const f = Math.floor(r);
  const h = r - f >= 0.3;
  return "\u2605".repeat(f) + (h ? "\u00bd" : "") + "\u2606".repeat(5 - f - (h ? 1 : 0));
}

function disc(orig: number, curr: number) {
  return orig > 0 ? Math.round(((orig - curr) / orig) * 100) : 0;
}

export default function ProductDetailPage({ params }: { params: { id: string } }) {
  const [product, setProduct] = useState<Product | null>(null);
  const [query, setQuery] = useState("");
  const [zipCode, setZipCode] = useState("");
  const [showReviews, setShowReviews] = useState(false);
  const [showQA, setShowQA] = useState(false);
  const [showSeller, setShowSeller] = useState(false);
  const [shippingZip, setShippingZip] = useState("");
  const shippingInputRef = { current: null as HTMLInputElement | null };
  const [shippingChecked, setShippingChecked] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState<number | null>(null);
  const [addedToCart, setAddedToCart] = useState(false);

  const idx = parseInt(params.id, 10);

  useEffect(() => {
    fetch("/api/entities")
      .then((r) => r.json())
      .then((data) => {
        if (data.products && data.products[idx]) {
          setProduct(data.products[idx]);
        }
        setQuery(data.search_query || "");
        setZipCode(data.zip_code || "10001");
      })
      .catch(() => {});
  }, [idx]);

  const handleAddToCart = useCallback(() => {
    fetch("/api/cart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_index: idx }),
    }).catch(() => {});
    setAddedToCart(true);
  }, [idx]);

  const handleShippingCheck = useCallback(() => {
    // Read directly from DOM to support native browser typing (CUA agent)
    const input = document.querySelector('[data-action="shipping_zip_input"]') as HTMLInputElement;
    const val = input?.value || shippingZip;
    if (val.length >= 5) {
      setShippingZip(val);
      setShippingChecked(true);
    }
  }, [shippingZip]);

  if (!product) {
    return <div style={{ padding: 40, textAlign: "center" }}>Loading product...</div>;
  }

  const hasDis = product.list_price_cents != null && product.list_price_cents > product.price_cents;
  let price = product.price_cents;
  let varLabel = "";
  if (selectedVariant !== null && product.variants) {
    price += product.variants.price_deltas_cents[selectedVariant] ?? 0;
    varLabel = product.variants.options[selectedVariant] ?? "";
  }

  return (
    <>
      <div className="breadcrumb">
        <a href="/">Amazon</a> &gt;{" "}
        <a href={`/search?q=${encodeURIComponent(query)}`}>Results</a> &gt; {product.name}
      </div>

      <div style={{ display: "flex", gap: 16, marginBottom: 12, fontSize: 13 }}>
        <a
          data-action="back_to_results"
          href={`/search?q=${encodeURIComponent(query)}`}
        >
          &larr; Back to results
        </a>
        <a
          data-action="back_to_deals"
          href="/deals"
        >
          &larr; Back to deals
        </a>
      </div>

      <div className="detail-layout">
        {/* Image column */}
        <div className="detail-image-col">
          <div className="detail-image">
            <div>{product.brand}</div>
            <div>{product.name}</div>
            {varLabel && <div style={{ marginTop: 8, color: "#007185" }}>Variant: {varLabel}</div>}
          </div>
        </div>

        {/* Info column */}
        <div className="detail-info-col">
          <h1 className="detail-title">{product.name}</h1>
          <div style={{ fontSize: 13, color: "#565959" }}>by {product.brand}</div>
          <div style={{ fontSize: 12, color: "#565959", marginTop: 2 }}>ASIN: {product.asin}</div>

          <div style={{ margin: "8px 0" }}>
            <span className="stars">{stars(product.rating)}</span>
            <span className="rating-count">
              {" "}{product.rating} out of 5 ({product.review_count.toLocaleString()} ratings)
            </span>
          </div>

          <div className="detail-price">
            {hasDis && (
              <span className="discount" style={{ fontSize: 16 }}>
                -{disc(product.list_price_cents!, price)}%{" "}
              </span>
            )}
            {fmt(price)}
            {hasDis && (
              <span className="list-price" style={{ marginLeft: 8 }}>
                List: {fmt(product.list_price_cents!)}
              </span>
            )}
          </div>

          {product.prime_eligible && (
            <div style={{ marginBottom: 8 }}>
              <span className="prime-badge">Prime</span>
              <span style={{ marginLeft: 8, fontSize: 14 }}>
                FREE delivery in {product.shipping.prime_delivery_days} day
                {product.shipping.prime_delivery_days > 1 ? "s" : ""}
              </span>
            </div>
          )}

          {/* Variants */}
          {product.variants && (
            <div className="variant-section">
              <div className="variant-label">
                {product.variants.type}:{" "}
                {selectedVariant !== null ? product.variants.options[selectedVariant] : "Select"}
              </div>
              <div className="variant-options">
                {product.variants.options.map((opt, i) => (
                  <button
                    key={i}
                    data-action={i === 0 ? "variant_option_a" : i === 1 ? "variant_option_b" : undefined}
                    className={`variant-option ${selectedVariant === i ? "selected" : ""}`}
                    onClick={() => setSelectedVariant(i)}
                    aria-label={`Select ${product.variants!.type}: ${opt}`}
                  >
                    {opt}
                    {product.variants!.price_deltas_cents[i] !== 0 && (
                      <div style={{ fontSize: 12, color: "#565959" }}>
                        {product.variants!.price_deltas_cents[i] > 0 ? "+" : ""}
                        {fmt(product.variants!.price_deltas_cents[i])}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Features */}
          <div className="feature-bullets">
            <h3 style={{ fontSize: 16, marginBottom: 8 }}>About this item</h3>
            <ul>
              {product.features.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </div>

          {/* Shipping */}
          <div className="shipping-section">
            <strong>Delivery</strong>
            {!shippingChecked ? (
              <>
                <div style={{ fontSize: 13, color: "#565959", marginTop: 4 }}>
                  Enter your ZIP code for delivery estimate
                </div>
                <div className="shipping-input">
                  <input
                    data-action="shipping_zip_input"
                    type="text"
                    placeholder="ZIP code"
                    maxLength={5}
                    defaultValue=""
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleShippingCheck();
                    }}
                    aria-label="ZIP code for delivery estimate"
                  />
                  <button onClick={handleShippingCheck} aria-label="Check delivery">
                    Check
                  </button>
                </div>
              </>
            ) : (
              <div className="shipping-result">
                {product.prime_eligible ? (
                  <div style={{ color: "#007600" }}>
                    <strong>FREE Prime delivery</strong> to {shippingZip} in{" "}
                    {product.shipping.prime_delivery_days} day
                    {product.shipping.prime_delivery_days > 1 ? "s" : ""}
                  </div>
                ) : product.shipping.cost_cents === 0 ? (
                  <div style={{ color: "#007600" }}>
                    <strong>FREE delivery</strong> to {shippingZip} in{" "}
                    {product.shipping.delivery_days} days
                  </div>
                ) : (
                  <div>
                    Delivery to {shippingZip}: <strong>{fmt(product.shipping.cost_cents)}</strong>{" "}
                    ({product.shipping.delivery_days} days)
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Buy box */}
        <div className="buy-box">
          <div className="price" style={{ fontSize: 18 }}>{fmt(price)}</div>
          {product.prime_eligible && (
            <div style={{ fontSize: 13, color: "#007600", margin: "4px 0" }}>
              FREE Prime delivery
            </div>
          )}
          <div style={{ fontSize: 14, color: "#007600", margin: "8px 0" }}>In Stock</div>

          <button
            data-action="add_to_cart_button"
            className="btn-add-to-cart"
            onClick={handleAddToCart}
            aria-label="Add to Cart"
          >
            {addedToCart ? "\u2713 Added to Cart" : "Add to Cart"}
          </button>

          <div style={{ marginTop: 12, fontSize: 13, color: "#565959" }}>
            <div>Ships from: {product.seller.name}</div>
            <div>
              Sold by:{" "}
              <button
                data-action="seller_link"
                onClick={() => setShowSeller(!showSeller)}
                style={{
                  background: "none",
                  border: "none",
                  color: "#007185",
                  cursor: "pointer",
                  padding: 0,
                  fontSize: 13,
                }}
                aria-label={`View seller info for ${product.seller.name}`}
              >
                {product.seller.name}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Seller info (expandable) */}
      {showSeller && (
        <div className="seller-info" style={{ marginTop: 16 }}>
          <h3>Seller Information</h3>
          <div style={{ fontSize: 14, marginTop: 8 }}>
            <div><strong>{product.seller.name}</strong></div>
            <div>
              <span className="stars">{stars(product.seller.rating)}</span>{" "}
              {product.seller.rating} out of 5{product.seller.total_ratings ? ` (${product.seller.total_ratings.toLocaleString()} ratings)` : ""}
            </div>
            <div>{product.seller.positive_feedback_pct}% positive feedback over last 12 months</div>
          </div>
        </div>
      )}

      {/* Reviews section (expandable) */}
      <div style={{ marginTop: 16, background: "white", padding: 16, borderRadius: 4 }}>
        <button
          data-action="reviews_section"
          onClick={() => setShowReviews(!showReviews)}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            width: "100%",
            textAlign: "left",
            padding: 0,
          }}
          aria-label={`${showReviews ? "Hide" : "Show"} customer reviews`}
          aria-expanded={showReviews}
        >
          <h2 className="section-header" style={{ borderTop: "none", marginTop: 0 }}>
            Customer Reviews ({product.reviews.length})
            <span style={{ float: "right", fontSize: 14 }}>
              {showReviews ? "\u25bc" : "\u25b6"}
            </span>
          </h2>
        </button>

        {showReviews && (
          <div>
            <div style={{ marginBottom: 12 }}>
              <span className="stars" style={{ fontSize: 18 }}>{stars(product.rating)}</span>{" "}
              {product.rating} out of 5 stars
            </div>
            {product.reviews.length === 0 && (
              <div style={{ padding: 16, color: "#565959" }}>No reviews yet.</div>
            )}
            {product.reviews.map((review, i) => (
              <div key={i} className="review-card">
                <div className="review-stars">
                  {"\u2605".repeat(review.rating)}{"\u2606".repeat(5 - review.rating)}
                </div>
                <div className="review-title">{review.title}</div>
                <div className="review-meta">
                  By {review.reviewer_name || review.reviewer} on {review.date}
                  {(review.verified_purchase || review.verified) && (
                    <span style={{ color: "#C7511F" }}> &mdash; Verified Purchase</span>
                  )}
                </div>
                <div className="review-text">{review.text}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Q&A section (expandable) */}
      {product.qa_pairs.length > 0 && (
        <div style={{ marginTop: 16, background: "white", padding: 16, borderRadius: 4 }}>
          <button
            data-action="qa_section"
            onClick={() => setShowQA(!showQA)}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              width: "100%",
              textAlign: "left",
              padding: 0,
            }}
            aria-label={`${showQA ? "Hide" : "Show"} questions and answers`}
            aria-expanded={showQA}
          >
            <h2 className="section-header" style={{ borderTop: "none", marginTop: 0 }}>
              Questions &amp; Answers ({product.qa_pairs.length})
              <span style={{ float: "right", fontSize: 14 }}>
                {showQA ? "\u25bc" : "\u25b6"}
              </span>
            </h2>
          </button>

          {showQA && (
            <div>
              {product.qa_pairs.map((qa, i) => (
                <div key={i} className="qa-card">
                  <div className="qa-question">Q: {qa.question}</div>
                  <div className="qa-answer">A: {qa.answer}</div>
                  <div style={{ fontSize: 12, color: "#565959" }}>
                    {qa.votes} people found this helpful
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );
}
