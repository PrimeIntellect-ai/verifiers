import { getEntities, formatPrice, starsDisplay, discountPct } from "../store";

export const dynamic = "force-dynamic";

export default function DealsPage() {
  const entities = getEntities();
  const { deals } = entities;

  return (
    <>
      <div className="breadcrumb">
        <a href="/">Amazon</a> &gt; Today&apos;s Deals
      </div>

      <h1 style={{ fontSize: 24, marginBottom: 16 }}>Today&apos;s Deals</h1>

      <div className="deal-grid">
        {deals.map((deal, i) => {
          const discount = discountPct(deal.original_price_cents, deal.deal_price_cents);

          return (
            <a
              key={i}
              data-action={i === 0 ? "deal_card_1" : undefined}
              href={`/product/${i}`}
              className="deal-card"
            >
              <span className="deal-badge">
                {deal.deal_type === "lightning" ? "⚡ Lightning Deal" : "Deal of the Day"}
              </span>

              <div style={{ fontWeight: 500, margin: "8px 0" }}>
                {deal.product.name}
              </div>

              <div>
                <span className="stars">{starsDisplay(deal.product.rating)}</span>
                <span className="rating-count">
                  ({deal.product.review_count.toLocaleString()})
                </span>
              </div>

              <div style={{ marginTop: 8 }}>
                <span className="price">{formatPrice(deal.deal_price_cents)}</span>
                <span className="list-price">{formatPrice(deal.original_price_cents)}</span>
                <span className="discount">-{discount}%</span>
              </div>

              {deal.product.prime_eligible && (
                <span className="prime-badge" style={{ marginTop: 8 }}>Prime</span>
              )}

              {deal.deal_type === "lightning" && deal.claimed_pct && (
                <div style={{ marginTop: 8 }}>
                  <div
                    style={{
                      background: "#F0F2F2",
                      height: 6,
                      borderRadius: 3,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        background: "#CC0C39",
                        width: `${deal.claimed_pct}%`,
                        height: "100%",
                      }}
                    />
                  </div>
                  <div style={{ fontSize: 11, color: "#CC0C39", marginTop: 2 }}>
                    {deal.claimed_pct}% claimed
                  </div>
                </div>
              )}
            </a>
          );
        })}
      </div>

      {deals.length === 0 && (
        <div style={{ padding: 40, textAlign: "center", background: "white", borderRadius: 4 }}>
          <h2>No deals available right now</h2>
          <p>Check back later for new deals.</p>
        </div>
      )}
    </>
  );
}
