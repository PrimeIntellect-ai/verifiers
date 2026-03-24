/**
 * Server-side entity store using file-based persistence.
 *
 * In Next.js standalone mode, module-level variables may not share
 * between API routes and server components. We use /tmp for persistence.
 */

import { readFileSync, writeFileSync, existsSync } from "fs";

const STORE_PATH = "/tmp/amazon-sim-store.json";

export interface Product {
  name: string;
  brand: string;
  price_cents: number;
  list_price_cents?: number;
  rating: number;
  review_count: number;
  prime_eligible: boolean;
  features: string[];
  asin: string;
  seller: {
    name: string;
    rating: number;
    total_ratings: number;
    positive_feedback_pct: number;
  };
  shipping: {
    cost_cents: number;
    delivery_days: number;
    prime_delivery_days: number;
  };
  variants?: {
    type: string;
    options: string[];
    price_deltas_cents: number[];
  };
  reviews: {
    reviewer?: string;
    reviewer_name?: string;
    rating: number;
    title: string;
    text: string;
    date: string;
    verified?: boolean;
    verified_purchase?: boolean;
  }[];
  qa_pairs: {
    question: string;
    answer: string;
    votes: number;
  }[];
}

export interface Deal {
  product: Product;
  deal_price_cents: number;
  original_price_cents: number;
  deal_type: string;
  claimed_pct?: number;
}

export interface Category {
  name: string;
  slug?: string;
  category_id?: string;
  subcategories: string[];
  products?: Product[];
  product_count?: number;
}

export interface Entities {
  products: Product[];
  deals: Deal[];
  categories: Category[];
  search_query: string;
  refined_query?: string;
  zip_code: string;
}

export interface StartWorld {
  [path: string]: any;
}

interface StoreData {
  entities: Entities;
  start_world: StartWorld;
  cart: number[];
}

const DEFAULT_STORE: StoreData = {
  entities: {
    products: [],
    deals: [],
    categories: [],
    search_query: "",
    zip_code: "10001",
  },
  start_world: {},
  cart: [],
};

function readStore(): StoreData {
  try {
    if (existsSync(STORE_PATH)) {
      return JSON.parse(readFileSync(STORE_PATH, "utf-8"));
    }
  } catch {}
  return { ...DEFAULT_STORE };
}

function writeStore(data: StoreData) {
  writeFileSync(STORE_PATH, JSON.stringify(data), "utf-8");
}

export function initStore(entities: Entities, startWorld: StartWorld) {
  writeStore({ entities, start_world: startWorld, cart: [] });
}

export function getEntities(): Entities {
  return readStore().entities;
}

export function getStartWorld(): StartWorld {
  return readStore().start_world;
}

export function getCart(): number[] {
  return readStore().cart;
}

export function addToCart(productIndex: number) {
  const data = readStore();
  if (!data.cart.includes(productIndex)) {
    data.cart.push(productIndex);
    writeStore(data);
  }
}

// Category slug helper — entity sampler uses category_id, test data uses slug
export function getCategorySlug(cat: Category): string {
  return cat.slug || cat.category_id || cat.name.toLowerCase().replace(/ & /g, "_").replace(/ /g, "_");
}

// Formatting helpers
export function formatPrice(cents: number): string {
  const dollars = Math.floor(cents / 100);
  const remainder = cents % 100;
  return `$${dollars}.${remainder.toString().padStart(2, "0")}`;
}

export function starsDisplay(rating: number): string {
  const full = Math.floor(rating);
  const half = rating - full >= 0.3;
  return "\u2605".repeat(full) + (half ? "\u00bd" : "") + "\u2606".repeat(5 - full - (half ? 1 : 0));
}

export function discountPct(original: number, current: number): number {
  if (original <= 0) return 0;
  return Math.round(((original - current) / original) * 100);
}
