import { getEntities, formatPrice, starsDisplay, discountPct } from "../store";

export const dynamic = "force-dynamic";

interface Props {
  searchParams: { q?: string; sort?: string; f_new?: string; f_price?: string; f_prime?: string };
}

export default function SearchResultsPage({ searchParams }: Props) {
  const entities = getEntities();
  const query = searchParams.q || entities.search_query || "";
  const sortBy = searchParams.sort || "featured";
  const filterNew = searchParams.f_new === "1";
  const filterPrice = searchParams.f_price === "1";
  const filterPrime = searchParams.f_prime === "1";

  // Apply filters
  let products = [...entities.products];
  if (filterNew) {
    products = products.filter(() => true); // All simulated products are "new"
  }
  if (filterPrime) {
    products = products.filter((p) => p.prime_eligible);
  }

  // Apply sort
  if (sortBy === "price-asc") {
    products.sort((a, b) => a.price_cents - b.price_cents);
  } else if (sortBy === "price-desc") {
    products.sort((a, b) => b.price_cents - a.price_cents);
  } else if (sortBy === "avg-review") {
    products.sort((a, b) => b.rating - a.rating);
  }

  // Build filter URL helpers
  const baseUrl = `/search?q=${encodeURIComponent(query)}`;
  function filterUrl(key: string, value: string): string {
    const params = new URLSearchParams();
    params.set("q", query);
    if (sortBy !== "featured") params.set("sort", sortBy);
    if (filterNew || key === "f_new") params.set("f_new", "1");
    if (filterPrice || key === "f_price") params.set("f_price", "1");
    if (filterPrime || key === "f_prime") params.set("f_prime", "1");
    params.set(key, value);
    return `/search?${params.toString()}`;
  }

  function sortUrl(sort: string): string {
    const params = new URLSearchParams();
    params.set("q", query);
    params.set("sort", sort);
    if (filterNew) params.set("f_new", "1");
    if (filterPrice) params.set("f_price", "1");
    if (filterPrime) params.set("f_prime", "1");
    return `/search?${params.toString()}`;
  }

  return (
    <>
      <div className="breadcrumb">
        <a href="/">Amazon</a> &gt; Search results for &quot;{query}&quot;
      </div>

      <div className="search-layout">
        {/* Filter sidebar */}
        <aside className="filter-sidebar">
          <div className="filter-section">
            <h3>Condition</h3>
            <a
              data-action="filter_condition_new"
              href={filterUrl("f_new", "1")}
              className="filter-option"
            >
              <input type="checkbox" readOnly checked={filterNew} /> New
            </a>
          </div>

          <div className="filter-section">
            <h3>Price</h3>
            <a
              data-action="filter_price_range"
              href={filterUrl("f_price", "1")}
              className="filter-option"
            >
              <input type="checkbox" readOnly checked={filterPrice} /> Under $50
            </a>
          </div>

          <div className="filter-section">
            <h3>Delivery</h3>
            <a
              data-action="filter_prime_only"
              href={filterUrl("f_prime", "1")}
              className="filter-option"
            >
              <input type="checkbox" readOnly checked={filterPrime} /> Prime FREE Shipping
            </a>
          </div>
        </aside>

        {/* Results area */}
        <div className="results-area">
          <div className="sort-bar">
            <span>{products.length} results for &quot;{query}&quot;</span>
            <div>
              Sort by:
              <a data-action="sort_price_asc" href={sortUrl("price-asc")}
                style={{ marginLeft: 8, fontWeight: sortBy === "price-asc" ? 700 : 400 }}>
                Price: Low to High
              </a>
              <a data-action="sort_avg_review" href={sortUrl("avg-review")}
                style={{ marginLeft: 8, fontWeight: sortBy === "avg-review" ? 700 : 400 }}>
                Avg. Review
              </a>
            </div>
          </div>

          {/* Product cards */}
          {products.map((product, index) => {
            const dataAction = `product_card_${index + 1}`;
            const hasDiscount = product.list_price_cents && product.list_price_cents > product.price_cents;

            return (
              <a
                key={index}
                data-action={dataAction}
                href={`/product/${index}`}
                className="product-card"
              >
                <div className="product-card-image">
                  {product.brand} - {product.name}
                </div>
                <div className="product-card-info">
                  <div className="product-card-title">{product.name}</div>
                  <div style={{ fontSize: 13, color: "#565959", marginBottom: 4 }}>
                    by {product.brand}
                  </div>
                  <div>
                    <span className="stars">{starsDisplay(product.rating)}</span>
                    <span className="rating-count">({product.review_count.toLocaleString()})</span>
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <span className="price">
                      <span className="price-symbol">$</span>
                      {Math.floor(product.price_cents / 100)}
                      <span className="price-fraction">
                        {(product.price_cents % 100).toString().padStart(2, "0")}
                      </span>
                    </span>
                    {hasDiscount && (
                      <>
                        <span className="list-price">
                          {formatPrice(product.list_price_cents!)}
                        </span>
                        <span className="discount">
                          -{discountPct(product.list_price_cents!, product.price_cents)}%
                        </span>
                      </>
                    )}
                  </div>
                  {product.prime_eligible && (
                    <div style={{ marginTop: 4 }}>
                      <span className="prime-badge">Prime</span>
                      <span className="delivery-info">
                        {" "}FREE delivery {product.shipping.prime_delivery_days === 1
                          ? "tomorrow"
                          : `in ${product.shipping.prime_delivery_days} days`}
                      </span>
                    </div>
                  )}
                  {!product.prime_eligible && product.shipping.cost_cents > 0 && (
                    <div className="delivery-info">
                      Shipping: {formatPrice(product.shipping.cost_cents)}
                    </div>
                  )}
                  <div style={{ marginTop: 8, fontSize: 13, color: "#565959" }}>
                    {product.features.slice(0, 2).join(" · ")}
                  </div>
                </div>
              </a>
            );
          })}

          {products.length === 0 && (
            <div style={{ padding: 40, textAlign: "center", background: "white" }}>
              <h2>No results found</h2>
              <p>Try different search terms or adjust your filters.</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
