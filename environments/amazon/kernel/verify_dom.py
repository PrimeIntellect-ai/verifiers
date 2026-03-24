"""Verify the Next.js app implements the kernel's browser_action contract.

Two levels of verification:
  1. Source check — grep app source for data-action attributes, cross-reference
     against the graph contract. No running app needed.
  2. Runtime check — start the app, seed entities, fetch each page, verify
     data-action elements and entity content appear in the rendered HTML.

Usage (from environments/amazon/):
    python -m kernel.verify_dom                        # source check only
    python -m kernel.verify_dom --runtime              # source + runtime checks
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_KERNEL_DIR = Path(__file__).parent
_ENV_DIR = _KERNEL_DIR.parent
APP_DIR = _ENV_DIR / "app"
APP_SRC = APP_DIR / "app"
DEFAULT_CONTRACT = _KERNEL_DIR / "graph_contract.yaml"
APP_PORT = 3098


# ---------------------------------------------------------------------------
# Contract loading
# ---------------------------------------------------------------------------


def load_contract(path: Path) -> dict:
    """Load graph contract and extract browser_action specs."""
    try:
        from .loaders import load_graph_contract

        contract = load_graph_contract(str(path))
    except ImportError:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        return _parse_contract_raw(raw)

    pages: dict[str, list[dict]] = defaultdict(list)
    observations: dict[str, list[str]] = defaultdict(list)
    tool_calls: list[str] = []

    for action in contract.actions:
        ba = action.browser_action
        if ba is None:
            if action.tool_call:
                tool_calls.append(action.action_id)
            continue
        if ba.element:
            pages[ba.page].append(
                {
                    "element": ba.element,
                    "interaction": ba.interaction,
                    "input_value": ba.input_value,
                    "action_id": action.action_id,
                }
            )
        else:
            observations[ba.page].append(action.action_id)

    return {
        "pages": dict(pages),
        "observations": dict(observations),
        "tool_calls": tool_calls,
    }


def _parse_contract_raw(raw: dict) -> dict:
    """Fallback parser for raw YAML when kernel isn't importable."""
    pages: dict[str, list[dict]] = defaultdict(list)
    observations: dict[str, list[str]] = defaultdict(list)
    tool_calls: list[str] = []

    for action in raw.get("actions", []):
        ba = action.get("browser_action")
        tc = action.get("tool_call")
        if ba is None and tc is not None:
            tool_calls.append(action["action_id"])
            continue
        if ba is None:
            continue
        page = ba.get("page", "")
        element = ba.get("element")
        if element:
            pages[page].append(
                {
                    "element": element,
                    "interaction": ba.get("interaction"),
                    "input_value": ba.get("input_value"),
                    "action_id": action["action_id"],
                }
            )
        else:
            observations[page].append(action["action_id"])

    for schema in raw.get("action_schemas", []):
        for variant in schema.get("variants", []):
            ba = variant.get("browser_action")
            if ba is None:
                continue
            page = ba.get("page", "")
            element = ba.get("element")
            aid = (
                variant.get("action_id")
                or f"{schema['schema_id']}_{variant['variant_id']}"
            )
            if element:
                pages[page].append(
                    {
                        "element": element,
                        "interaction": ba.get("interaction"),
                        "input_value": ba.get("input_value"),
                        "action_id": aid,
                    }
                )
            else:
                observations[page].append(aid)

    return {
        "pages": dict(pages),
        "observations": dict(observations),
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Source-level verification
# ---------------------------------------------------------------------------


def check_source(contract: dict) -> list[dict]:
    """Check app source files for data-action attributes."""
    results = []

    source_elements: set[str] = set()
    source_dynamic: list[str] = []

    for tsx_file in APP_SRC.rglob("*.tsx"):
        content = tsx_file.read_text()
        for m in re.finditer(r'data-action="([^"]+)"', content):
            source_elements.add(m.group(1))
        for m in re.finditer(r"data-action=\{([^}]+)\}", content):
            source_dynamic.append(m.group(1))
        for m in re.finditer(r"`([a-z_]+)\$\{[^}]+\}`", content):
            source_dynamic.append(m.group(0))

    for expr in source_dynamic:
        for m in re.finditer(r'"([^"]+)"', expr):
            source_elements.add(m.group(1))
        for m in re.finditer(r"`([^`]*?)\$\{[^}]+\}([^`]*?)`", expr):
            prefix, suffix = m.group(1), m.group(2)
            for i in range(1, 4):
                source_elements.add(f"{prefix}{i}{suffix}")

    contract_elements: set[str] = set()
    for page, elements in contract["pages"].items():
        for el in elements:
            contract_elements.add(el["element"])

    for page, elements in sorted(contract["pages"].items()):
        for el in elements:
            element = el["element"]
            found = element in source_elements
            results.append(
                {
                    "check": "source",
                    "page": page,
                    "element": element,
                    "action_id": el["action_id"],
                    "interaction": el["interaction"],
                    "found": found,
                }
            )

    extra = source_elements - contract_elements
    layout_elements = {
        "search_input",
        "search_submit",
        "deals_link",
        "cart_icon",
        "category_menu",
    }
    truly_extra = extra - layout_elements
    if truly_extra:
        for el in sorted(truly_extra):
            results.append(
                {
                    "check": "source_extra",
                    "element": el,
                    "found": True,
                    "note": "In source but not in contract",
                }
            )

    return results


def check_routes(contract: dict) -> list[dict]:
    """Check that route files exist for each page in the contract."""
    results = []
    page_to_route = {
        "home": "page.tsx",
        "search_results": "search/page.tsx",
        "product_detail": "product/[id]/page.tsx",
        "category": "category/[slug]/page.tsx",
        "deals": "deals/page.tsx",
        "cart": "cart/page.tsx",
    }

    all_pages = set(contract["pages"].keys()) | set(contract["observations"].keys())
    for page in sorted(all_pages):
        route_file = page_to_route.get(page)
        if route_file:
            exists = (APP_SRC / route_file).exists()
            results.append(
                {
                    "check": "route",
                    "page": page,
                    "route": route_file,
                    "found": exists,
                }
            )
        else:
            results.append(
                {
                    "check": "route",
                    "page": page,
                    "route": "???",
                    "found": False,
                    "note": f"No route mapping for page '{page}'",
                }
            )

    return results


# ---------------------------------------------------------------------------
# Runtime verification
# ---------------------------------------------------------------------------

TEST_ENTITIES = {
    "products": [
        {
            "name": "VerifyTest Product Alpha",
            "brand": "TestBrand",
            "price_cents": 12345,
            "list_price_cents": 15999,
            "rating": 4.3,
            "review_count": 789,
            "prime_eligible": True,
            "features": ["Feature Alpha One", "Feature Alpha Two"],
            "asin": "B0VERIFY1",
            "seller": {
                "name": "VerifySeller",
                "rating": 4.6,
                "total_ratings": 2300,
                "positive_feedback_pct": 96,
            },
            "shipping": {"cost_cents": 0, "delivery_days": 3, "prime_delivery_days": 1},
            "variants": {
                "type": "Size",
                "options": ["Small", "Large"],
                "price_deltas_cents": [0, 2000],
            },
            "reviews": [
                {
                    "reviewer": "VerifyReviewer",
                    "rating": 5,
                    "title": "VerifyReviewTitle",
                    "text": "VerifyReviewText content here.",
                    "date": "2025-06-15",
                    "verified": True,
                }
            ],
            "qa_pairs": [
                {
                    "question": "VerifyQuestion here?",
                    "answer": "VerifyAnswer here.",
                    "votes": 17,
                }
            ],
        },
        {
            "name": "VerifyTest Product Beta",
            "brand": "TestBrand2",
            "price_cents": 6789,
            "rating": 3.9,
            "review_count": 234,
            "prime_eligible": False,
            "features": ["Feature Beta One"],
            "asin": "B0VERIFY2",
            "seller": {
                "name": "BetaSeller",
                "rating": 4.1,
                "total_ratings": 500,
                "positive_feedback_pct": 89,
            },
            "shipping": {
                "cost_cents": 599,
                "delivery_days": 7,
                "prime_delivery_days": 3,
            },
            "reviews": [],
            "qa_pairs": [],
        },
        {
            "name": "VerifyTest Product Gamma",
            "brand": "TestBrand3",
            "price_cents": 99999,
            "rating": 4.8,
            "review_count": 56,
            "prime_eligible": True,
            "features": ["Feature Gamma One"],
            "asin": "B0VERIFY3",
            "seller": {
                "name": "GammaSeller",
                "rating": 4.9,
                "total_ratings": 100,
                "positive_feedback_pct": 99,
            },
            "shipping": {"cost_cents": 0, "delivery_days": 2, "prime_delivery_days": 1},
            "reviews": [],
            "qa_pairs": [],
        },
    ],
    "deals": [
        {
            "product": {
                "name": "VerifyTest Product Alpha",
                "brand": "TestBrand",
                "price_cents": 12345,
                "rating": 4.3,
                "review_count": 789,
                "prime_eligible": True,
                "features": [],
                "asin": "B0VERIFY1",
                "seller": {
                    "name": "VerifySeller",
                    "rating": 4.6,
                    "total_ratings": 2300,
                    "positive_feedback_pct": 96,
                },
                "shipping": {
                    "cost_cents": 0,
                    "delivery_days": 3,
                    "prime_delivery_days": 1,
                },
                "reviews": [],
                "qa_pairs": [],
            },
            "deal_price_cents": 9900,
            "original_price_cents": 12345,
            "deal_type": "lightning",
            "claimed_pct": 45,
        },
    ],
    "categories": [
        {
            "name": "VerifyCategory",
            "slug": "verify-cat",
            "subcategories": ["VerifySub1", "VerifySub2"],
            "products": [],
        },
    ],
    "search_query": "verify test query",
    "zip_code": "90210",
}


def fetch_page(url: str) -> str:
    try:
        return urlopen(url, timeout=10).read().decode("utf-8", errors="replace")
    except (URLError, OSError):
        return ""


def check_runtime(contract: dict, base_url: str) -> list[dict]:
    """Verify pages at runtime."""
    results = []

    page_urls = {
        "home": f"{base_url}/",
        "search_results": f"{base_url}/search?q=verify+test+query",
        "product_detail": f"{base_url}/product/0",
        "category": f"{base_url}/category/verify-cat",
        "deals": f"{base_url}/deals",
        "cart": f"{base_url}/cart",
    }

    entities_html = fetch_page(f"{base_url}/api/entities")
    try:
        entities_data = json.loads(entities_html)
        api_ok = len(entities_data.get("products", [])) == 3
    except (json.JSONDecodeError, ValueError):
        api_ok = False

    results.append(
        {
            "check": "runtime_api",
            "endpoint": "/api/entities",
            "found": api_ok,
            "note": "3 products returned" if api_ok else "API returned unexpected data",
        }
    )

    server_rendered_pages = ["home", "search_results", "deals", "category", "cart"]

    for page in server_rendered_pages:
        url = page_urls.get(page)
        if not url:
            continue

        html = fetch_page(url)
        if not html:
            results.append(
                {
                    "check": "runtime_page",
                    "page": page,
                    "url": url,
                    "found": False,
                    "note": "Failed to fetch page",
                }
            )
            continue

        found_elements = set(re.findall(r'data-action="([^"]+)"', html))

        for el_spec in contract["pages"].get(page, []):
            element = el_spec["element"]
            found = element in found_elements
            results.append(
                {
                    "check": "runtime_element",
                    "page": page,
                    "element": element,
                    "action_id": el_spec["action_id"],
                    "found": found,
                }
            )

        content_checks = {
            "home": ["VerifyCategory", "VerifyTest Product Alpha"],
            "search_results": [
                "VerifyTest Product Alpha",
                "VerifyTest Product Beta",
                "VerifyTest Product Gamma",
                "TestBrand",
            ],
            "deals": ["VerifyTest Product Alpha", "deal_card_1"],
            "category": ["VerifyCategory", "VerifySub1"],
            "cart": [],
        }

        for content in content_checks.get(page, []):
            found = content in html
            results.append(
                {
                    "check": "runtime_content",
                    "page": page,
                    "content": content,
                    "found": found,
                }
            )

    client_pages = ["product_detail"]
    for page in client_pages:
        results.append(
            {
                "check": "runtime_note",
                "page": page,
                "found": True,
                "note": "Client-rendered — requires browser (Chrome MCP) for full verification.",
            }
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results(results: list[dict]):
    passed = sum(1 for r in results if r["found"])
    failed = sum(1 for r in results if not r["found"])
    total = len(results)

    print(f"\n{'=' * 70}")
    print(f"  VERIFICATION RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 70}\n")

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["check"]].append(r)

    type_labels = {
        "route": "Route Checks",
        "source": "Source data-action Checks",
        "source_extra": "Extra Source Elements (info only)",
        "runtime_api": "Runtime API Checks",
        "runtime_element": "Runtime DOM Element Checks",
        "runtime_content": "Runtime Content Checks",
        "runtime_page": "Runtime Page Checks",
        "runtime_note": "Runtime Notes",
    }

    for check_type in [
        "route",
        "source",
        "source_extra",
        "runtime_api",
        "runtime_element",
        "runtime_content",
        "runtime_page",
        "runtime_note",
    ]:
        items = by_type.get(check_type, [])
        if not items:
            continue

        label = type_labels.get(check_type, check_type)
        type_passed = sum(1 for r in items if r["found"])

        print(f"--- {label} ({type_passed}/{len(items)}) ---")
        for r in items:
            icon = "\u2713" if r["found"] else "\u2717"
            parts = []
            if "page" in r:
                parts.append(r["page"])
            if "element" in r:
                parts.append(f'data-action="{r["element"]}"')
            if "action_id" in r:
                parts.append(f"({r['action_id']})")
            if "route" in r:
                parts.append(r["route"])
            if "endpoint" in r:
                parts.append(r["endpoint"])
            if "content" in r:
                parts.append(f'"{r["content"]}"')
            if "note" in r:
                parts.append(f"[{r['note']}]")

            line = " ".join(parts)
            print(f"  {icon} {line}")
        print()

    if failed > 0:
        print(f"FAILED: {failed} check(s) did not pass.")
        return False
    else:
        print("ALL CHECKS PASSED.")
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify Next.js app against kernel contract"
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=str(DEFAULT_CONTRACT),
        help="Path to graph_contract.yaml",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Also run runtime checks (starts the app)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=APP_PORT,
        help=f"Port for runtime checks (default: {APP_PORT})",
    )
    args = parser.parse_args()

    contract_path = Path(args.contract)
    if not contract_path.exists():
        print(f"Contract not found: {contract_path}")
        sys.exit(1)

    print(f"Loading contract: {contract_path}")
    contract = load_contract(contract_path)

    total_elements = sum(len(els) for els in contract["pages"].values())
    total_observations = sum(len(obs) for obs in contract["observations"].values())
    total_pages = len(
        set(contract["pages"].keys()) | set(contract["observations"].keys())
    )
    print(
        f"  {total_pages} pages, {total_elements} interactive elements, "
        f"{total_observations} observation actions, {len(contract['tool_calls'])} tool calls"
    )

    print("\nRunning source checks...")
    results = check_routes(contract)
    results.extend(check_source(contract))

    if args.runtime:
        print("\nRunning runtime checks...")
        base_url = f"http://localhost:{args.port}"

        print(f"  Starting app on port {args.port}...")
        env = {**os.environ, "PORT": str(args.port)}
        proc = subprocess.Popen(
            ["npm", "run", "start"],
            cwd=str(APP_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        ready = False
        for _ in range(30):
            try:
                urlopen(base_url, timeout=2)
                ready = True
                break
            except (URLError, OSError):
                time.sleep(1)

        if not ready:
            print("  ERROR: App did not start within 30s")
            proc.terminate()
            sys.exit(1)

        print("  App running. Seeding entities...")

        payload = json.dumps({"entities": TEST_ENTITIES, "start_world": {}}).encode()
        req = Request(
            f"{base_url}/api/init",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(req, timeout=10)
            print("  Entities seeded.")
        except Exception as e:
            print(f"  WARNING: seed failed: {e}")

        results.extend(check_runtime(contract, base_url))

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    all_passed = print_results(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
