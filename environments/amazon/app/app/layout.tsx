import type { Metadata } from "next";
import { getEntities, getCategorySlug } from "./store";
import "./globals.css";

export const metadata: Metadata = {
  title: "Amazon.com",
  description: "Amazon Shopping Simulation",
};

export const dynamic = "force-dynamic";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const entities = getEntities();
  const firstCat = entities.categories?.[0];
  const firstCategorySlug = firstCat ? getCategorySlug(firstCat) : "all";
  return (
    <html lang="en">
      <body>
        <nav className="navbar">
          <a href="/" className="logo" aria-label="Amazon Home">
            amazon
          </a>

          <form className="search-bar" action="/search" method="GET">
            <input
              data-action="search_input"
              name="q"
              type="text"
              placeholder="Search Amazon"
              aria-label="Search Amazon"
              autoComplete="off"
            />
            <button
              data-action="search_submit"
              type="submit"
              aria-label="Search"
            >
              🔍
            </button>
          </form>

          <div className="nav-links">
            <a href="/deals" data-action="deals_link">
              Today&apos;s Deals
            </a>
            <a href="/cart" data-action="cart_icon" aria-label="Shopping Cart">
              🛒 Cart
            </a>
          </div>
        </nav>

        <div className="subnav">
          <a href={`/category/${firstCategorySlug}`} data-action="category_menu">
            All Categories
          </a>
          <a href="/deals">Today&apos;s Deals</a>
          <a href="/">Customer Service</a>
          <a href="/">Best Sellers</a>
        </div>

        <div className="content">{children}</div>
      </body>
    </html>
  );
}
