# Amazon Shopping — Browser CUA Environment

Self-contained environment for generating and evaluating browser-based CUA tasks on a simulated Amazon.com.

## Architecture

```
kernel/
  graph_contract.yaml ──► BFS sampler ──► task_specs.sampled.yaml
  sampling_request.yaml ─┘                        │
                                                   ▼
                                entity_sampler.py + task_spec.py
                                                   │
                                                   ▼
                                HF Dataset (prompt, answer_key, entities)
                                                   │
                                                   ▼
                                LocalBrowserEnv + Next.js app in sandbox
                                                   │
                                                   ▼
                                Agent rollout → submit_result → judge → score
```

**Correct-by-construction guarantee:** The kernel proves every task is solvable via BFS. The Next.js app faithfully renders the kernel's world state through `data-action` elements. Entity data is the ground truth — unique, non-guessable values (prices like $123.45, ratings like 4.3) mean the agent can only get the right answer by actually navigating the app.

## File Map

```
amazon/
├── CLAUDE.md              ← you are here
├── amazon.py              ← environment wiring, load_environment() entry point
├── entities.py            ← Pydantic entity models (Product, Seller, Review, etc.)
├── entity_sampler.py      ← deterministic entity generation from task_id
├── task_spec.py           ← unified description + answer key generator
├── task_specs.sampled.yaml← BFS output: abstract tasks
├── pyproject.toml
├── kernel/                ← BFS state-machine engine + domain definition
│   ├── graph_contract.yaml   ← state machine: projection fields, actions, browser_action specs
│   ├── sampling_request.yaml ← BFS config: terminal profiles, seed schemas
│   ├── resample.py           ← re-run BFS → task_specs.sampled.yaml
│   ├── verify_dom.py         ← Phase A: programmatic DOM verification
│   ├── types.py              ← GraphContractSpec, ActionContract, BrowserActionSpec, etc.
│   ├── semantics.py          ← apply_action, is_action_enabled, sync rules
│   ├── preflight.py          ← structural task checks
│   ├── sampler.py            ← BFS fan-out task generator
│   └── loaders.py            ← YAML I/O
└── app/                   ← Next.js Amazon sim
    └── app/               ← routes: page.tsx, search/, product/[id]/, deals/, category/[slug]/, cart/
```

## How the Kernel Works

The kernel is a state machine. The world is a flat dict of `projection_fields` (e.g., `page.type`, `search.executed`, `p1.detail_read`). Actions have `requires_world` preconditions and `effects_world` postconditions. The BFS sampler explores all reachable states from each seed and records paths that reach terminal profiles.

Each action also declares a `browser_action:` block mapping it to the app:
- `page`: which route the action occurs on (home, search_results, product_detail, deals, category, cart)
- `element`: the `data-action` attribute in the DOM (e.g., `search_submit`, `product_card_1`)
- `interaction`: click, type, scroll, etc.

**This is the contract between kernel and app.** Every `browser_action.element` must exist as `data-action="X"` in the app source. Every `browser_action.page` must have a route.

## How to Add a New Action

### 1. Define the action in `kernel/graph_contract.yaml`

Add projection fields if needed, then add the action:

```yaml
projection_fields:
  - ... existing fields ...
  - wishlist.item_added     # NEW

actions:
  - action_id: add_to_wishlist
    requestor: assistant
    classification: causal
    requires_world:
      - path: page.type
        value: product_detail
      - path: p1.detail_read        # must have read product first
        value: true
    effects_world:
      - path: wishlist.item_added
        set: true
    browser_action:
      page: product_detail
      element: wishlist_button       # data-action value in app
      interaction: click
```

Key rules:
- `requires_world` must be satisfiable — actions are only enabled when all predicates hold
- `effects_world` must change the world (otherwise BFS prunes it as a stutter)
- `browser_action.element` becomes the `data-action` attribute you add to the app

### 2. Add the element to the Next.js app

In the correct page component (matching `browser_action.page`), add the DOM element:

```tsx
<button data-action="wishlist_button" onClick={handleAddToWishlist}>
  Add to Wishlist
</button>
```

The `data-action` value **must exactly match** `browser_action.element` in the contract.

### 3. Update entity sampler if needed

If the action needs new entity data (e.g., wishlist items), update `entities.py` with the model and `entity_sampler.py` to generate it. Add a `task.requires_wishlist` flag to `start_world` config.

### 4. Update task_spec.py if the action produces answer key data

Add a section builder that returns `(description_fragment, answer_key_fragment)`:

```python
def _wishlist_section(goal_world, entities, config):
    if not _goal_has(goal_world, "wishlist.item_added", True):
        return None, {}
    product = entities.products[0]
    desc = f"Add {product.name} to your wishlist."
    ak = {"wishlist_item": product.name}
    return desc, ak
```

**Critical rule:** Every field in the answer key must be asked for in the description, and vice versa. Generate both in the same function from the same data.

### 5. Update `kernel/sampling_request.yaml`

Add the new action's world effects to relevant terminal profiles and seed schemas so BFS can reach states where the action fires.

### 6. Re-sample and verify

```bash
python -m kernel.resample
python -m kernel.verify_dom
```

Then run browser verification (see Verification section below).

## How to Add a New Task Type

### 1. Add a terminal profile in `kernel/sampling_request.yaml`

```yaml
terminal_profiles:
  - profile_id: submitted_with_wishlist
    description: "Agent adds item to wishlist and submits"
    requires_world:
      - path: wishlist.item_added
        value: true
      - path: task.submitted
        value: true
```

### 2. Add a seed schema

```yaml
seed_schemas:
  - schema_id: wishlist_tasks
    seed_id_template: "wishlist_{entry_point}"
    allowed_terminal_profiles: [submitted_with_wishlist]
    min_depth: 5
    max_depth: 10
    dimensions:
      - dimension_id: entry_point
        variants:
          - variant_id: search
            start_world:
              - path: task.entry_point
                set: search
```

### 3. Re-sample

```bash
python -m kernel.resample
```

Check the output — you should see tasks with `profile=submitted_with_wishlist`.

## Sync Rules

Sync rules fire automatically after every action to derive computed state:

```yaml
sync_rules:
  - rule_id: product_detail_implies_search
    requires_world:
      - path: p1.detail_read
        value: true
    effects_world:
      - path: search.has_results
        set: true
```

Use sync rules for: page type derivation, computed flags, state that follows from other state. Avoid cycles — the engine caps at 32 iterations.

## Entity Field Name Contract

Entity Pydantic models are serialized to JSON and sent to the Next.js app via `POST /api/init`. The app's TypeScript reads these fields by name.

**If the Pydantic model uses `reviewer_name` but TypeScript expects `reviewer`, the app renders empty values.**

After changing entity models, check every field name against the app's TypeScript. Common mismatches we've hit:

| Python (Pydantic) | TypeScript (App) | Resolution |
|---|---|---|
| `reviewer_name` | `reviewer` | Handle both in app: `review.reviewer_name \|\| review.reviewer` |
| `verified_purchase` | `verified` | Same pattern |
| `category_id` | `slug` | Use helper function in app |

## Verification

Run verification after any change. There are four phases — A is automated, B and C use Chrome MCP, D is a code check.

| Changed | Run |
|---------|-----|
| `kernel/graph_contract.yaml` | `python -m kernel.resample` then Phase A |
| `app/` components | Phase A, then Phase B |
| `entity_sampler.py` or `task_spec.py` | Phase C |
| `entities.py` | Phase D |
| Any of the above | `prime eval run amazon -m anthropic/claude-sonnet-4.6 -a '{"max_tasks": 3}'` |

### Phase A: Programmatic DOM Check

```bash
python -m kernel.verify_dom           # source checks only
python -m kernel.verify_dom --runtime # source + live app checks
```

Verifies every `browser_action.element` from the contract exists as `data-action` in app source and every `browser_action.page` has a route file.

### Phase B: Element-by-Element Browser Verification

**Prerequisites:** Start the Next.js app (`cd app && npm run dev`), seed entities via `POST /api/init`, open Chrome MCP.

For EVERY interactive element in the contract, verify in the live browser:

1. Navigate to the element's page
2. Find the element via `[data-action="element_name"]`
3. Confirm: exists, correct tag type (button/input/link), visible, responds to interaction
4. Record PASS/FAIL with evidence

**Page-by-page checklist** (derive specifics from `kernel/graph_contract.yaml`):

- **HOME** (`/`): `search_input` (input/type), `search_submit` (button/click), `category_menu` (link/click → `/category/`), `deals_link` (link/click → `/deals`)
- **SEARCH RESULTS** (`/search?q=...`): `product_card_1/2/3` (link/click → `/product/N`), `filter_condition_new`, `filter_price_range`, `filter_prime_only` (click → URL params), `sort_price_asc`, `sort_avg_review` (click → reorder), `cart_icon` (link → `/cart`)
- **PRODUCT DETAIL** (`/product/N`): `shipping_zip_input` (input/type), `reviews_section` (click → expand reviews), `seller_link` (click → expand seller info), `qa_section` (click → expand Q&A), `add_to_cart_button` (button/click), `variant_option_a/b` (click → select), `back_to_results` (link → `/search`), `back_to_deals` (link → `/deals`)
- **DEALS** (`/deals`): `deal_card_1` (link/click → `/product/N`)
- **CATEGORY** (`/category/[slug]`): observation only (products render from entities)
- **CART** (`/cart`): observation only (items render after add-to-cart)

**Do not skip any element. One element = one explicit check.**

### Phase C: End-to-End Task Walks

This is the most important phase. Most bugs are only caught here because they require real entity data flowing through the full pipeline.

**Protocol:**

1. **Select tasks.** Use the minimum covering set algorithm to pick the fewest tasks that exercise all kernel actions:

```python
from kernel.loaders import load_graph_contract, load_sampling_request
from kernel.sampler import sample_task_intents

contract = load_graph_contract("kernel/graph_contract.yaml")
request = load_sampling_request("kernel/sampling_request.yaml")
tasks = sample_task_intents(contract, request)

# Greedy minimum covering set
remaining = set()
for st in tasks:
    remaining.update(st.task.required_actions)

selected = []
task_list = [(st.task, set(st.task.required_actions)) for st in tasks]
while remaining:
    best_task, best_actions = max(task_list, key=lambda t: len(t[1] & remaining))
    newly_covered = best_actions & remaining
    if not newly_covered:
        break
    selected.append(best_task)
    remaining -= newly_covered

print(f"Need {len(selected)} tasks to cover all actions")
for t in selected:
    print(f"  {t.task_id}: {t.required_actions}")
```

Also select one task per terminal profile: `submitted_surface`, `submitted_with_detail`, `submitted_comparison`, `submitted_from_category`, `submitted_from_deals`, `submitted_with_cart`.

2. **For each task**, execute in order:
   - Generate entities: `sample_entities(task_id, start_world)`
   - Seed the app: `POST /api/init` with `{entities, start_world}`
   - Walk through `required_actions` one by one in the browser
   - For each action: execute it, verify entity data renders correctly, record PASS/FAIL

3. **Action-type verification rules:**
   - **Navigation** (navigate_to_amazon, navigate_to_category, navigate_to_deals): page loads with entity content
   - **Search** (enter_search_query, submit_search): entity products appear in results
   - **Click-to-navigate** (click_product_1/2/3, click_deal_product): correct product detail page loads
   - **Observation** (read_results_text, read_p1_detail, etc.): entity-specific values visible (name, price, rating, features)
   - **Interaction** (check_p*_shipping, check_p*_reviews, check_p*_seller, check_p*_qa, variants, add-to-cart): element exists, interaction works, entity data appears
   - **Filter/sort** (filter_*, sort_*): URL params change, results update

4. **After all actions**, confirm: all actions executable, task description answerable from rendered data, entity values correct.

**If any step fails:** record the failure with evidence, determine root cause (app bug / entity sampler issue / kernel contract issue), fix, and re-run the entire task from step 1.

### Phase D: Entity Schema Validation

Check Pydantic field names match TypeScript:

```python
from entity_sampler import sample_entities
entities = sample_entities("test_task", {"task.entry_point": "search"})
product = entities.model_dump()["products"][0]
# Compare product.keys() against app's TypeScript interface
```

## Chrome MCP for Dry Runs

To manually test the app with Chrome MCP tools:

1. Start the Next.js app: `cd app && npm run dev`
2. In Claude Code with Chrome MCP enabled, navigate to `http://localhost:3000`
3. Seed entities: POST to `http://localhost:3000/api/init` with entity JSON
4. Walk through actions using `mcp__claude-in-chrome__*` tools

This is how Phase B and Phase C verification work — the agent uses Chrome MCP to interact with the live app and verify each action.

## Running Evals

```bash
# Install the environment
prime env install amazon

# Quick smoke test (3 tasks)
prime eval run amazon -m anthropic/claude-sonnet-4.6 -a '{"max_tasks": 3}'

# Full eval
prime eval run amazon -m anthropic/claude-sonnet-4.6
```

## Common Pitfalls

**Description/answer key drift:** Agent gets correct data but judge scores partial. Cause: description asks for X but answer key doesn't include it (or vice versa). Prevention: use unified `task_spec.py` — never write description and answer key generators separately.

**Star rating precision loss:** Answer key says 4.7, agent reads "4.5 stars" (visual rounding). Prevention: show numeric rating alongside stars on listing pages.

**Missing products in listing answer keys:** Agent correctly reports products from category page, judge scores "no". Cause: answer key only has category metadata, not the products displayed. Prevention: when `category.read=true` in goal_world, include product listings in answer key.

**React controlled component:** CUA agent typing doesn't trigger React `onChange`. The app reads `input.value` directly from DOM for shipping ZIP input as a workaround.

**BFS state explosion:** Full resampling with 64 seeds takes a couple minutes. For quick iteration, use `python -m kernel.resample --max-per-schema 5` or reduce seeds in sampling_request.yaml temporarily.
