"""Generate a browser test manifest from the kernel graph contract.

Produces a JSON manifest with one test case per action in the contract.
Each test case specifies:
  - action_id: which kernel action we're testing
  - page_url: where to navigate before testing
  - element: data-action attribute to find (or null for observations)
  - interaction: click/type/etc
  - input_value: what to type (for type actions)
  - verify: what to check after the interaction
  - setup_note: any precondition setup needed

The manifest is designed to be executed by a browser agent (Claude via
Chrome MCP) or a Playwright script.

Usage:
    python generate_test_manifest.py                     # stdout
    python generate_test_manifest.py -o test_manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_CONTRACT = (
    Path.home()
    / "Documents"
    / "depgraph"
    / "domains"
    / "amazon_shopping"
    / "graph_contract.yaml"
)

# Test entity data with unique strings for verification
SEED_ENTITIES = {
    "products": [
        {
            "name": "TestProduct Alpha X900",
            "brand": "AlphaBrand",
            "price_cents": 12345,
            "list_price_cents": 15999,
            "rating": 4.3,
            "review_count": 789,
            "prime_eligible": True,
            "features": ["UniqueFeatureAlpha1", "UniqueFeatureAlpha2"],
            "asin": "B0ALPHA01",
            "seller": {
                "name": "AlphaSeller Official",
                "rating": 4.6,
                "total_ratings": 2300,
                "positive_feedback_pct": 96,
            },
            "shipping": {
                "cost_cents": 0,
                "delivery_days": 3,
                "prime_delivery_days": 1,
            },
            "variants": {
                "type": "Color",
                "options": ["Midnight Black", "Arctic White"],
                "price_deltas_cents": [0, 500],
            },
            "reviews": [
                {
                    "reviewer": "TestReviewer Jane",
                    "rating": 5,
                    "title": "UniqueReviewTitle Alpha",
                    "text": "UniqueReviewText Alpha content here for verification.",
                    "date": "2025-06-15",
                    "verified": True,
                },
            ],
            "qa_pairs": [
                {
                    "question": "UniqueQuestion Alpha here?",
                    "answer": "UniqueAnswer Alpha content here.",
                    "votes": 17,
                },
            ],
        },
        {
            "name": "TestProduct Beta Z500",
            "brand": "BetaBrand",
            "price_cents": 6789,
            "rating": 3.9,
            "review_count": 234,
            "prime_eligible": False,
            "features": ["UniqueFeatureBeta1"],
            "asin": "B0BETA002",
            "seller": {
                "name": "BetaSeller Store",
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
            "name": "TestProduct Gamma K100",
            "brand": "GammaBrand",
            "price_cents": 99999,
            "rating": 4.8,
            "review_count": 56,
            "prime_eligible": True,
            "features": ["UniqueFeatureGamma1"],
            "asin": "B0GAMMA03",
            "seller": {
                "name": "GammaSeller Direct",
                "rating": 4.9,
                "total_ratings": 100,
                "positive_feedback_pct": 99,
            },
            "shipping": {
                "cost_cents": 0,
                "delivery_days": 2,
                "prime_delivery_days": 1,
            },
            "reviews": [],
            "qa_pairs": [],
        },
    ],
    "deals": [
        {
            "product": {
                "name": "TestProduct Alpha X900",
                "brand": "AlphaBrand",
                "price_cents": 12345,
                "rating": 4.3,
                "review_count": 789,
                "prime_eligible": True,
                "features": [],
                "asin": "B0ALPHA01",
                "seller": {
                    "name": "AlphaSeller Official",
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
            "name": "TestElectronics",
            "slug": "test-electronics",
            "subcategories": ["TestHeadphones", "TestSpeakers"],
            "products": [],
        },
    ],
    "search_query": "test alpha product",
    "zip_code": "90210",
}

# Page type → URL path mapping
PAGE_URLS = {
    "home": "/",
    "search_results": "/search?q=test+alpha+product",
    "product_detail": "/product/{product_index}",
    "category": "/category/test-electronics",
    "deals": "/deals",
}


def load_contract(path: Path) -> list[dict]:
    """Load all actions from the graph contract."""
    try:
        sys.path.insert(0, str(Path.home() / "Documents" / "depgraph"))
        from depgraph.loaders import load_graph_contract

        contract = load_graph_contract(str(path))
        actions = []
        for action in contract.actions:
            entry = {
                "action_id": action.action_id,
                "classification": action.classification,
                "requires_world": [
                    {"path": p.path, "op": p.op, "value": p.value}
                    for p in action.requires_world
                ],
                "effects_world": [
                    {"path": e.path, "set": e.set} for e in action.effects_world
                ],
            }
            if action.browser_action:
                entry["browser_action"] = {
                    "page": action.browser_action.page,
                    "element": action.browser_action.element,
                    "interaction": action.browser_action.interaction,
                    "input_value": action.browser_action.input_value,
                }
            if action.tool_call:
                entry["tool_call"] = {
                    "tool_name": action.tool_call.tool_name,
                }
            actions.append(entry)
        return actions
    except ImportError:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        # Simplified fallback — just return raw actions
        return raw.get("actions", [])


def generate_manifest(actions: list[dict]) -> list[dict]:
    """Generate one test case per action."""
    tests = []

    for action in actions:
        ba = action.get("browser_action")
        tc = action.get("tool_call")
        action_id = action["action_id"]
        requires = action.get("requires_world", [])
        effects = action.get("effects_world", [])

        if tc and not ba:
            # Tool call action (submit_result) — skip browser test
            tests.append(
                {
                    "action_id": action_id,
                    "type": "tool_call",
                    "tool_name": tc["tool_name"],
                    "skip_browser": True,
                    "note": "Tool call action — tested via verifiers environment, not browser.",
                }
            )
            continue

        if not ba:
            continue

        page = ba["page"]
        element = ba.get("element")
        interaction = ba.get("interaction")
        input_value = ba.get("input_value")

        # Determine which product index this action targets
        product_index = 0  # default
        for req in requires:
            if req["path"] == "page.current_product":
                val = str(req["value"])
                if val.isdigit():
                    product_index = int(val) - 1

        # Build the page URL
        page_url = PAGE_URLS.get(page, "/")
        if "{product_index}" in page_url:
            page_url = page_url.replace("{product_index}", str(product_index))

        # Build the test case
        test = {
            "action_id": action_id,
            "type": "observation" if element is None else "interaction",
            "page": page,
            "page_url": page_url,
            "element": element,
            "interaction": interaction,
            "input_value": input_value,
            "requires_world": requires,
            "effects_world": effects,
        }

        # Add specific verification steps
        if element is None:
            # Observation action — verify page content
            content_checks = _content_checks_for_page(page, product_index)
            test["verify"] = {
                "type": "content_visible",
                "checks": content_checks,
                "note": f"Page '{page}' must render entity content visible to agent.",
            }
        elif interaction == "click":
            # Click action — verify element exists and something happens
            expected_effects = _describe_effects(effects)
            test["verify"] = {
                "type": "click_and_check",
                "find": f'[data-action="{element}"]',
                "after": expected_effects,
            }
        elif interaction == "type":
            # Type action — verify input exists and accepts text
            type_value = _resolve_input_value(input_value)
            test["verify"] = {
                "type": "type_and_check",
                "find": f'[data-action="{element}"]',
                "type_text": type_value,
                "after": _describe_effects(effects),
            }

        # Add setup notes for preconditions
        setup = _setup_for_requires(requires, page_url)
        if setup:
            test["setup"] = setup

        tests.append(test)

    return tests


def _content_checks_for_page(page: str, product_index: int) -> list[str]:
    """What content should be visible on this page."""
    products = SEED_ENTITIES["products"]
    checks = {
        "home": ["TestElectronics", "TestProduct Alpha X900"],
        "search_results": [
            "TestProduct Alpha X900",
            "TestProduct Beta Z500",
            "TestProduct Gamma K100",
        ],
        "product_detail": [
            products[product_index]["name"],
            products[product_index]["brand"],
            products[product_index]["asin"],
        ],
        "category": ["TestElectronics", "TestHeadphones"],
        "deals": ["TestProduct Alpha X900"],
    }
    return checks.get(page, [])


def _describe_effects(effects: list[dict]) -> list[dict]:
    """Describe expected effects in human-readable form."""
    result = []
    for e in effects:
        path = e["path"]
        value = e["set"]
        if path == "page.type":
            result.append({"check": "page_changed", "expected_page": value})
        elif path == "page.current_product":
            result.append({"check": "product_changed", "expected_product": value})
        else:
            result.append({"check": "state_change", "path": path, "value": value})
    return result


def _resolve_input_value(input_value: str | None) -> str:
    """Resolve entity references in input_value."""
    if not input_value:
        return "test input"
    if "{entities.search_query}" in input_value:
        return SEED_ENTITIES["search_query"]
    if "{entities.refined_query}" in input_value:
        return "refined test query"
    if "{entities.zip_code}" in input_value:
        return SEED_ENTITIES["zip_code"]
    return input_value


def _setup_for_requires(requires: list[dict], page_url: str) -> list[str]:
    """Describe any setup steps needed before the action can be tested."""
    steps = []
    for req in requires:
        path = req["path"]
        value = req["value"]
        if path.startswith("task."):
            continue  # Task config — handled by entity seeding
        if path == "page.type":
            continue  # Handled by page_url navigation
        if path == "page.current_product":
            continue  # Handled by product_index in URL
        if path == "search.query_entered" and value is True:
            steps.append("Must type a search query first")
        if path == "extract.results_read" and value is True:
            steps.append("Must read/screenshot the results page first")
        if path.endswith(".detail_read") and value is True:
            steps.append(f"Must visit product detail page first ({path})")
        if path == "cart.opened" and value is True:
            steps.append("Must open cart first")
        if path == "deals.read" and value is True:
            steps.append("Must read deals page first")
        if path == "category.read" and value is True:
            steps.append("Must read category page first")
    return steps


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate browser test manifest")
    parser.add_argument("--contract", type=str, default=str(DEFAULT_CONTRACT))
    parser.add_argument(
        "-o", "--output", type=str, help="Output file (default: stdout)"
    )
    args = parser.parse_args()

    contract_path = Path(args.contract)
    if not contract_path.exists():
        print(f"Contract not found: {contract_path}", file=sys.stderr)
        sys.exit(1)

    actions = load_contract(contract_path)
    manifest = generate_manifest(actions)

    # Summary
    interactions = sum(1 for t in manifest if t["type"] == "interaction")
    observations = sum(1 for t in manifest if t["type"] == "observation")
    tool_calls = sum(1 for t in manifest if t["type"] == "tool_call")
    print(
        f"Generated {len(manifest)} tests: {interactions} interactions, "
        f"{observations} observations, {tool_calls} tool calls",
        file=sys.stderr,
    )

    # Unique elements
    elements = {t["element"] for t in manifest if t.get("element")}
    pages = {t["page"] for t in manifest if t.get("page")}
    print(
        f"Covers {len(elements)} unique elements across {len(pages)} pages",
        file=sys.stderr,
    )

    output = json.dumps(manifest, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
