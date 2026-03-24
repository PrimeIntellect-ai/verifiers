import { getEntities, getCart, formatPrice } from "../store";

export const dynamic = "force-dynamic";

export default function CartPage() {
  const entities = getEntities();
  const cartIndices = getCart();
  const cartProducts = cartIndices
    .map((i) => ({ product: entities.products[i], index: i }))
    .filter((item) => item.product);

  const subtotal = cartProducts.reduce((sum, item) => sum + item.product.price_cents, 0);

  return (
    <>
      <div className="breadcrumb">
        <a href="/">Amazon</a> &gt; Shopping Cart
      </div>

      <h1 style={{ fontSize: 24, marginBottom: 16 }}>Shopping Cart</h1>

      {cartProducts.length > 0 ? (
        <div style={{ background: "white", borderRadius: 4 }}>
          {cartProducts.map((item, i) => (
            <div key={i} className="cart-item">
              <div className="cart-item-image">
                {item.product.brand} - {item.product.name}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 500, marginBottom: 4 }}>
                  <a href={`/product/${item.index}`}>{item.product.name}</a>
                </div>
                <div style={{ fontSize: 13, color: "#565959" }}>
                  by {item.product.brand}
                </div>
                <div style={{ fontSize: 12, color: "#565959" }}>
                  ASIN: {item.product.asin}
                </div>
                {item.product.prime_eligible && (
                  <span className="prime-badge">Prime</span>
                )}
                <div style={{ color: "#007600", fontSize: 14, marginTop: 4 }}>
                  In Stock
                </div>
                <div style={{ fontSize: 13, marginTop: 4 }}>Qty: 1</div>
              </div>
              <div className="price">{formatPrice(item.product.price_cents)}</div>
            </div>
          ))}
          <div className="cart-subtotal">
            Subtotal ({cartProducts.length} item{cartProducts.length !== 1 ? "s" : ""}):{" "}
            <strong>{formatPrice(subtotal)}</strong>
          </div>
        </div>
      ) : (
        <div
          style={{
            padding: 40,
            textAlign: "center",
            background: "white",
            borderRadius: 4,
          }}
        >
          <h2>Your Amazon Cart is empty</h2>
          <p style={{ marginTop: 8, color: "#565959" }}>
            Your shopping cart is empty. Browse products and add items to your cart.
          </p>
          <p style={{ marginTop: 16 }}>
            <a href="/">Continue shopping</a>
          </p>
        </div>
      )}
    </>
  );
}
