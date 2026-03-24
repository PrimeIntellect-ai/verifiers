import { getEntities, getCategorySlug, formatPrice, starsDisplay } from "../../store";

export const dynamic = "force-dynamic";

interface Props {
  params: { slug: string };
}

export default function CategoryPage({ params }: Props) {
  const entities = getEntities();
  const category = entities.categories.find((c) => getCategorySlug(c) === params.slug);

  if (!category) {
    // Fallback: show first category or generic page
    const fallback = entities.categories[0];
    if (!fallback) {
      return (
        <div style={{ padding: 40, textAlign: "center" }}>
          <h1>Category not found</h1>
          <a href="/">Return to home</a>
        </div>
      );
    }
  }

  const cat = category || entities.categories[0];
  const products = (cat?.products && cat.products.length > 0) ? cat.products : entities.products;

  return (
    <>
      <div className="breadcrumb">
        <a href="/">Amazon</a> &gt; {cat?.name || "All Categories"}
      </div>

      <h1 style={{ fontSize: 24, marginBottom: 16 }}>{cat?.name || "All Categories"}</h1>

      {/* Subcategory chips */}
      {cat?.subcategories && (
        <div className="category-chips">
          {cat.subcategories.map((sub) => (
            <span key={sub} className="category-chip">
              {sub}
            </span>
          ))}
        </div>
      )}

      {/* Product grid */}
      <div style={{ background: "white", borderRadius: 4 }}>
        {products.map((product, index) => (
          <a
            key={index}
            href={`/product/${index}`}
            className="product-card"
          >
            <div className="product-card-image">
              {product.brand} - {product.name}
            </div>
            <div className="product-card-info">
              <div className="product-card-title">{product.name}</div>
              <div style={{ fontSize: 13, color: "#565959" }}>by {product.brand}</div>
              <div>
                <span className="stars">{starsDisplay(product.rating)}</span>
                <span className="rating-count">({product.review_count.toLocaleString()})</span>
              </div>
              <div className="price" style={{ marginTop: 4 }}>
                {formatPrice(product.price_cents)}
              </div>
              {product.prime_eligible && (
                <span className="prime-badge">Prime</span>
              )}
            </div>
          </a>
        ))}
      </div>
    </>
  );
}
