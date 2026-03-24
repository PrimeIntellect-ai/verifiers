import { NextRequest, NextResponse } from "next/server";
import { addToCart, getCart, getEntities, formatPrice } from "../../store";

export const dynamic = "force-dynamic";

export async function GET() {
  const entities = getEntities();
  const indices = getCart();
  const items = indices.map((i) => entities.products[i]).filter(Boolean);
  const subtotal = items.reduce((s, p) => s + p.price_cents, 0);
  return NextResponse.json({
    items: indices,
    count: items.length,
    subtotal_cents: subtotal,
    subtotal: formatPrice(subtotal),
  });
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const productIndex = body.product_index;
  if (typeof productIndex === "number") {
    addToCart(productIndex);
  }
  return NextResponse.json({ ok: true, cart: getCart() });
}
