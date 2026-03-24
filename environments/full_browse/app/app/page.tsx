import { getEntities, getCategorySlug, formatPrice, starsDisplay } from "./store";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const entities = getEntities();
  const { products, deals, categories } = entities;

  return (
    <>
      <div className="home-hero">
        <h1>Welcome to Amazon</h1>
        <p>Search for products, browse categories, or check today&apos;s deals.</p>
      </div>

      {/* Categories section */}
      {categories.length > 0 && (
        <section className="home-section">
          <h2>Shop by Category</h2>
          <div className="home-grid">
            {categories.map((cat) => (
              <a
                key={getCategorySlug(cat)}
                href={`/category/${getCategorySlug(cat)}`}
                className="deal-card"
              >
                <h3>{cat.name}</h3>
                <p style={{ fontSize: 13, color: "#565959" }}>
                  {cat.subcategories.slice(0, 3).join(", ")}
                </p>
                <span style={{ color: "#007185", fontSize: 13 }}>Shop now →</span>
              </a>
            ))}
          </div>
        </section>
      )}

      {/* Deals section */}
      {deals.length > 0 && (
        <section className="home-section">
          <h2>Today&apos;s Deals</h2>
          <div className="home-grid">
            {deals.slice(0, 4).map((deal, i) => (
              <a
                key={i}
                href={`/deals`}
                className="deal-card"
              >
                <span className="deal-badge">
                  {deal.deal_type === "lightning" ? "⚡ Lightning Deal" : "Deal"}
                </span>
                <div style={{ fontWeight: 500, marginBottom: 4 }}>
                  {deal.product.name}
                </div>
                <div>
                  <span className="price">{formatPrice(deal.deal_price_cents)}</span>
                  <span className="list-price">{formatPrice(deal.original_price_cents)}</span>
                </div>
              </a>
            ))}
          </div>
        </section>
      )}

      {/* Trending products */}
      {products.length > 0 && (
        <section className="home-section">
          <h2>Trending Products</h2>
          <div className="home-grid">
            {products.slice(0, 4).map((p, i) => (
              <div key={i} className="deal-card">
                <div style={{ fontWeight: 500, marginBottom: 4 }}>{p.name}</div>
                <div>
                  <span className="stars">{starsDisplay(p.rating)}</span>
                  <span className="rating-count">({p.review_count})</span>
                </div>
                <div className="price">{formatPrice(p.price_cents)}</div>
                {p.prime_eligible && <span className="prime-badge">Prime</span>}
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  );
}
