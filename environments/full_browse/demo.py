"""Interactive demo — test the Amazon shopping Next.js app locally.

Usage:
    python demo.py                              # minimal test task
    python demo.py --task-id surface_plain_d6_000  # specific kernel task
    python demo.py --list                       # list available tasks

Starts the Next.js dev server, seeds it with task entities, opens your
browser, and gives you an interactive REPL to explore.

No sandbox, no CUA server, no verifiers dependency needed — just Node.js.
"""

import json
import os
import subprocess
import time
import webbrowser
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

APP_DIR = Path(__file__).parent / "app"
APP_PORT = 3000
APP_URL = f"http://localhost:{APP_PORT}"


def find_dataset() -> Path | None:
    """Look for a pre-built dataset.json in common locations."""
    candidates = [
        Path.home()
        / "Documents"
        / "depgraph"
        / "domains"
        / "amazon_shopping"
        / "dataset.json",
        Path(__file__).parent / "data" / "dataset.json",
        Path.home()
        / "Documents"
        / "tau2-bench"
        / "data"
        / "tau2"
        / "domains"
        / "amazon_shopping"
        / "dataset.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_tasks(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def minimal_task() -> dict:
    """Inline test task — no external files needed."""
    return {
        "task_id": "demo-test-001",
        "description": (
            "Search for 'wireless headphones' on Amazon. Find the 'Wireless Headphones Pro' "
            "by AudioTech. Report the product name, price, rating, and whether it has Prime."
        ),
        "entities": {
            "products": [
                {
                    "name": "Wireless Headphones Pro",
                    "brand": "AudioTech",
                    "price_cents": 7999,
                    "list_price_cents": 9999,
                    "rating": 4.5,
                    "review_count": 1234,
                    "prime_eligible": True,
                    "features": [
                        "Active noise cancellation",
                        "40hr battery life",
                        "Bluetooth 5.3",
                    ],
                    "asin": "B0DEMO001",
                    "seller": {
                        "name": "AudioTech Official",
                        "rating": 4.8,
                        "total_ratings": 5600,
                        "positive_feedback_pct": 97,
                    },
                    "shipping": {
                        "cost_cents": 0,
                        "delivery_days": 3,
                        "prime_delivery_days": 1,
                    },
                    "variants": {
                        "type": "Color",
                        "options": ["Black", "White", "Navy"],
                        "price_deltas_cents": [0, 0, 500],
                    },
                    "reviews": [
                        {
                            "reviewer": "Jane D.",
                            "rating": 5,
                            "title": "Best headphones I've owned",
                            "text": "Incredible sound quality and the noise cancellation is top notch. "
                            "Comfortable for long flights.",
                            "date": "2025-01-15",
                            "verified": True,
                        },
                        {
                            "reviewer": "Mike R.",
                            "rating": 4,
                            "title": "Great but pricey",
                            "text": "Sound is excellent. Wish the price was a bit lower.",
                            "date": "2025-02-03",
                            "verified": True,
                        },
                        {
                            "reviewer": "Sarah L.",
                            "rating": 3,
                            "title": "Good enough",
                            "text": "Decent headphones for the price. Battery life is as advertised.",
                            "date": "2025-03-10",
                            "verified": False,
                        },
                    ],
                    "qa_pairs": [
                        {
                            "question": "Does this work with iPhone?",
                            "answer": "Yes, it works with all Bluetooth devices including iPhone, Android, and laptops.",
                            "votes": 42,
                        },
                        {
                            "question": "How long does the battery last?",
                            "answer": "Up to 40 hours with ANC off, about 30 hours with ANC on.",
                            "votes": 28,
                        },
                    ],
                },
                {
                    "name": "Budget Wireless Earbuds",
                    "brand": "SoundBasic",
                    "price_cents": 2499,
                    "rating": 3.8,
                    "review_count": 567,
                    "prime_eligible": True,
                    "features": [
                        "Wireless charging case",
                        "IPX5 water resistant",
                        "Touch controls",
                    ],
                    "asin": "B0DEMO002",
                    "seller": {
                        "name": "SoundBasic Store",
                        "rating": 4.2,
                        "total_ratings": 1200,
                        "positive_feedback_pct": 91,
                    },
                    "shipping": {
                        "cost_cents": 0,
                        "delivery_days": 5,
                        "prime_delivery_days": 2,
                    },
                    "reviews": [
                        {
                            "reviewer": "Tom K.",
                            "rating": 4,
                            "title": "Great value",
                            "text": "For the price, these are hard to beat.",
                            "date": "2025-01-20",
                            "verified": True,
                        },
                    ],
                    "qa_pairs": [],
                },
                {
                    "name": "Premium Studio Headphones",
                    "brand": "ProAudio",
                    "price_cents": 34999,
                    "list_price_cents": 39999,
                    "rating": 4.9,
                    "review_count": 89,
                    "prime_eligible": False,
                    "features": [
                        "Studio-grade drivers",
                        "Detachable cable",
                        "Carrying case included",
                    ],
                    "asin": "B0DEMO003",
                    "seller": {
                        "name": "ProAudio Direct",
                        "rating": 4.9,
                        "total_ratings": 340,
                        "positive_feedback_pct": 99,
                    },
                    "shipping": {
                        "cost_cents": 999,
                        "delivery_days": 7,
                        "prime_delivery_days": 3,
                    },
                    "reviews": [],
                    "qa_pairs": [],
                },
            ],
            "deals": [
                {
                    "product": {
                        "name": "Wireless Headphones Pro",
                        "brand": "AudioTech",
                        "price_cents": 7999,
                        "rating": 4.5,
                        "review_count": 1234,
                        "prime_eligible": True,
                        "features": [],
                        "asin": "B0DEMO001",
                        "seller": {
                            "name": "AudioTech",
                            "rating": 4.8,
                            "total_ratings": 5600,
                            "positive_feedback_pct": 97,
                        },
                        "shipping": {
                            "cost_cents": 0,
                            "delivery_days": 3,
                            "prime_delivery_days": 1,
                        },
                        "reviews": [],
                        "qa_pairs": [],
                    },
                    "deal_price_cents": 5999,
                    "original_price_cents": 7999,
                    "deal_type": "lightning",
                    "claimed_pct": 67,
                },
            ],
            "categories": [
                {
                    "name": "Electronics",
                    "slug": "electronics",
                    "subcategories": ["Headphones", "Speakers", "Wearables"],
                    "products": [],
                },
            ],
            "search_query": "wireless headphones",
            "zip_code": "10001",
        },
        "start_world": {
            "page.type": "home",
            "task.entry_point": "search",
            "task.requires_detail": True,
            "task.requires_shipping": True,
            "task.requires_reviews": True,
            "task.requires_variants": True,
            "task.requires_filters": False,
            "task.requires_qa": True,
            "task.requires_cart": False,
            "task.num_products": 3,
        },
        "goal_world": [],
        "required_actions": [],
    }


def div(label: str = ""):
    w = 72
    if label:
        pad = w - len(label) - 4
        print(f"\n{'─' * 2} {label} {'─' * max(pad, 2)}")
    else:
        print(f"{'─' * w}")


def wait_for_server(url: str, timeout: int = 60):
    """Poll until the dev server is up."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urlopen(url, timeout=2)
            return True
        except (URLError, OSError):
            time.sleep(1)
    return False


def call_init(entities: dict, start_world: dict):
    """POST /api/init to seed the app."""
    payload = json.dumps({"entities": entities, "start_world": start_world}).encode()
    req = Request(
        f"{APP_URL}/api/init",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=10)
        return json.loads(resp.read())
    except Exception as e:
        print(f"  Warning: /api/init failed: {e}")
        return None


def main():
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Demo the Amazon shopping Next.js app")
    parser.add_argument("--task-id", help="Load a specific task from dataset.json")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument(
        "--no-open", action="store_true", help="Don't auto-open browser"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use 'npm run dev' instead of 'npm run start'",
    )
    args = parser.parse_args()

    # Load dataset if available
    dataset_path = find_dataset()
    all_tasks = load_tasks(dataset_path) if dataset_path else None

    if args.list:
        if not all_tasks:
            print("No dataset.json found. Only the built-in minimal task is available.")
            return
        from collections import Counter

        profiles = Counter(t.get("terminal_profile_id", "?") for t in all_tasks)
        print(f"Loaded {len(all_tasks)} tasks from {dataset_path}\n")
        print("Terminal profiles:")
        for p, c in profiles.most_common():
            ex = next(t for t in all_tasks if t.get("terminal_profile_id") == p)
            print(f"  {p}: {c} tasks (e.g. {ex['task_id']})")
        print("\nUse --task-id <id> to load a specific task.")
        return

    # Select task
    if args.task_id and all_tasks:
        task = next((t for t in all_tasks if t["task_id"] == args.task_id), None)
        if not task:
            print(f"Task '{args.task_id}' not found")
            return
    elif all_tasks:
        task = random.choice(all_tasks)
    else:
        task = minimal_task()

    div(f"TASK: {task['task_id']}")
    print(f"\n{task['description']}\n")

    if task.get("required_actions"):
        print(f"Required actions: {task['required_actions']}")
    if task.get("goal_world"):
        print(f"Goal predicates: {len(task['goal_world'])}")

    # Start Next.js
    div("STARTING APP")
    cmd = "dev" if args.dev else "start"
    print(f"  Running 'npm run {cmd}' in {APP_DIR}...")

    env = {**os.environ, "PORT": str(APP_PORT)}
    proc = subprocess.Popen(
        ["npm", "run", cmd],
        cwd=str(APP_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print(f"  Waiting for {APP_URL} ...")
    if not wait_for_server(APP_URL):
        print("  ERROR: App didn't start within 60s")
        proc.terminate()
        return

    print(f"  App running at {APP_URL}")

    # Seed with entities
    div("SEEDING ENTITIES")
    result = call_init(task["entities"], task.get("start_world", {}))
    if result:
        print(f"  /api/init → {result}")
    print(f"  Products: {len(task['entities'].get('products', []))}")
    print(f"  Deals: {len(task['entities'].get('deals', []))}")
    print(f"  Categories: {len(task['entities'].get('categories', []))}")
    print(f"  Search query: {task['entities'].get('search_query', '?')}")

    # Open browser
    div("BROWSE")
    q = task["entities"].get("search_query", "")
    print(f"\n  Home:           {APP_URL}/")
    print(f"  Search:         {APP_URL}/search?q={q}")
    print(f"  Product #1:     {APP_URL}/product/0")
    if len(task["entities"].get("products", [])) > 1:
        print(f"  Product #2:     {APP_URL}/product/1")
    if len(task["entities"].get("products", [])) > 2:
        print(f"  Product #3:     {APP_URL}/product/2")
    if task["entities"].get("deals"):
        print(f"  Deals:          {APP_URL}/deals")
    if task["entities"].get("categories"):
        slug = task["entities"]["categories"][0].get("slug", "all")
        print(f"  Category:       {APP_URL}/category/{slug}")

    if not args.no_open:
        webbrowser.open(APP_URL)

    # Interactive
    div("INTERACTIVE MODE")
    print("  The app is running. Browse it in your browser.")
    print("  Commands:")
    print("    open [url]  — open a URL in browser (e.g. 'open /search?q=headphones')")
    print("    entities    — dump entity data")
    print("    reinit      — re-seed the app with current task entities")
    print("    quit        — stop the app and exit")
    print()

    try:
        while True:
            try:
                cmd = input("demo> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "entities":
                print(json.dumps(task["entities"], indent=2, default=str)[:3000])
            elif cmd == "reinit":
                result = call_init(task["entities"], task.get("start_world", {}))
                print(f"  Re-seeded: {result}")
            elif cmd.startswith("open"):
                path = cmd[4:].strip() or "/"
                url = f"{APP_URL}{path}" if path.startswith("/") else path
                webbrowser.open(url)
                print(f"  Opened {url}")
            elif cmd == "help":
                print("  open [path]   — open URL in browser")
                print("  entities      — dump entity data")
                print("  reinit        — re-seed app with entities")
                print("  quit          — stop")
            elif cmd == "":
                continue
            else:
                print("  Unknown command. Type 'help'.")
    finally:
        div("SHUTTING DOWN")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Done.")


if __name__ == "__main__":
    main()
