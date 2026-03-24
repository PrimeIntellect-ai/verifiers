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
                              amazon.py: LocalBrowserEnv + Next.js app in sandbox
                                                   │
                                                   ▼
                              Agent rollout → submit_result tool → LLM judge → score
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
│   ├── semantics.py          ← apply_action, is_action_enabled
│   ├── preflight.py          ← structural task checks
│   ├── sampler.py            ← BFS fan-out task generator
│   └── loaders.py            ← YAML I/O
└── app/                   ← Next.js Amazon sim
    ├── app/store.ts          ← file-based entity store (/tmp/amazon-sim-store.json)
    └── app/               ← routes: page.tsx, search/, product/[id]/, deals/, category/[slug]/, cart/
```

## How the Kernel Works

The kernel is a state machine. The world is a flat dict of `projection_fields` (e.g., `page.type`, `search.executed`, `p1.detail_read`). Actions have `requires_world` preconditions and `effects_world` postconditions. The BFS sampler explores all reachable states from each seed and records paths that reach terminal profiles.

### Action Types

There are three kinds of actions in the contract:

**Browser actions with elements** — the agent clicks/types a specific DOM element:
```yaml
- action_id: check_p1_reviews
  browser_action:
    page: product_detail
    element: reviews_section      # data-action="reviews_section" in app
    interaction: click            # agent clicks to expand reviews panel
```

**Browser observation actions** — the agent just needs the page to render content (no element, no click). These include `read_*` actions (agent extracts info from the page) and `compile_results` (agent organizes findings before submitting — a kernel bookkeeping step with no browser interaction):
```yaml
- action_id: read_p1_detail
  browser_action:
    page: product_detail          # no element — agent reads from page/screenshot
```

**Tool call actions** — the agent calls a verifiers tool, not a browser element:
```yaml
- action_id: submit_results
  tool_call:
    tool_name: submit_result      # called via the submit_result tool, ends rollout
```

`submit_results` is the only tool_call action. It maps to the `_submit_result` tool in `amazon.py` which stores data in `state["submitted_result"]` and triggers the `@vf.stop` decorator to end the rollout.

## How amazon.py Works

`amazon.py` wires the kernel output to the verifiers framework:

**`_build_dataset()`** loads `task_specs.sampled.yaml`, runs each task through `entity_sampler.sample_entities()` and `task_spec.generate_task_spec()`, and produces an HF Dataset with fields: `prompt` (task description), `answer` (answer key JSON), `task_id`, and `info` (entities dict, start_world, etc.).

**`LocalBrowserEnv`** extends `vf.StatefulToolEnv`. On setup, it creates a sandbox with the Next.js app via `FullBrowseMode`, then seeds the app by POSTing entities to `http://localhost:3000/api/init` inside the sandbox. The app stores entities in `/tmp/amazon-sim-store.json` (file-based, one task per sandbox — not concurrent-safe).

**Agent tools** match production CUA traces:
- `computer(actions)` — batched browser actions (click, type, scroll, etc.), returns screenshot
- `get_page_text()` — full page text
- `read_page(filter)` — accessibility tree with `[ref=ref_42]` refs
- `find(query)` — element search returning `[ref_42]` format
- `form_input(ref, value)` — fill form field by ref
- `submit_result(data)` — submit findings as JSON string, ends rollout

**Judge scoring:** `judge_reward_func` reads `state["submitted_result"]`, sends it with the answer key to an LLM judge (Claude Sonnet via Prime Inference). The judge compares field by field and responds with `yes` (1.0), `partial` (0.5), or `no` (0.0).

## start_world Config Flags

The `start_world` dict in each sampled task controls what entities get generated and what the task asks the agent to do. These flags are set by the seed schemas in `sampling_request.yaml` and read by `entity_sampler.py`:

| Flag | Values | Effect |
|------|--------|--------|
| `task.entry_point` | `"search"`, `"category"`, `"deals"` | Where the agent starts navigating |
| `task.num_products` | 1-3 | How many products the entity sampler generates |
| `task.requires_reviews` | `true`/`false` | Whether products get review data |
| `task.requires_qa` | `true`/`false` | Whether products get Q&A pairs |
| `task.requires_variants` | `true`/`false` | Whether products get variant options (size, color) |

When adding new entity types, add a corresponding `task.requires_*` flag and read it in `entity_sampler.py`.

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
- For tool_call actions (rare — only `submit_results` uses this), use `tool_call:` instead of `browser_action:`

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
python -m kernel.resample --max-per-schema 5  # faster iteration
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

There are two levels: an automated source check, and browser dry-runs of real tasks.

| Changed | Run |
|---------|-----|
| `kernel/graph_contract.yaml` | `python -m kernel.resample` then `python -m kernel.verify_dom` |
| `app/` components | `python -m kernel.verify_dom` then browser verification |
| `entity_sampler.py` or `task_spec.py` | Browser verification |
| `entities.py` | Check field names match app TypeScript (see Entity Field Name Contract above) |
| Any of the above | `prime eval run amazon -m anthropic/claude-sonnet-4.6 -a '{"max_tasks": 3}'` |

### Automated: `verify_dom.py`

```bash
python -m kernel.verify_dom           # source checks only
python -m kernel.verify_dom --runtime # also starts app, fetches rendered pages, checks data-action attrs in live HTML
```

Verifies every `browser_action.element` from the contract exists as `data-action` in app source and every `browser_action.page` has a route file. Fast, catches typos and missing elements. Run after any contract or app change.

### Browser Verification

Walk through real sampled tasks end-to-end in the browser using Chrome MCP. This is the most important check — most bugs only surface when real entity data flows through the full pipeline.

**Setup:**
1. Start the Next.js app: `cd app && npm run dev`
2. In Claude Code with Chrome MCP enabled, navigate to `http://localhost:3000`

**Select tasks** using the minimum covering set — the fewest tasks that exercise all kernel actions:

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

Also pick one task per terminal profile (`submitted_surface`, `submitted_with_detail`, `submitted_comparison`, `submitted_from_category`, `submitted_from_deals`, `submitted_with_cart`).

**For each task:**
1. Generate entities: `sample_entities(task_id, start_world)`
2. Seed the app: `POST /api/init` with `{entities, start_world}`
3. Walk through `required_actions` one by one in the browser, verifying each:
   - **Element actions** (anything with `browser_action.element`): find `[data-action="X"]`, confirm it exists, is visible, and responds correctly to the interaction (click expands a panel, type fills an input, etc.)
   - **Observation actions** (`read_results_text`, `read_p1_detail`, `compile_results`, etc.): no element to click — just verify entity-specific values are visible on the page (product name, price, rating, features from the generated entities)
   - **`submit_results`**: tool_call, not browser — skip
4. After all actions: confirm the task description is answerable from the rendered entity data

**If any step fails:** record the failure, determine root cause (app bug / entity sampler / contract issue), fix, re-run the full task.

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

**React controlled component:** CUA agent typing doesn't trigger React `onChange`. The app reads `input.value` directly from DOM for shipping ZIP input as a workaround. Note: the `form_input` tool (which sets values by ref) has the same issue — the app's shipping check button reads the DOM value directly rather than relying on React state.

**Stateful app store:** The Next.js app persists entities in `/tmp/amazon-sim-store.json`. This means one sandbox = one task. The app is not safe for concurrent tasks in the same sandbox (each `POST /api/init` overwrites the store).

**BFS state explosion:** Full resampling with 64 seeds takes a couple minutes. For quick iteration, use `python -m kernel.resample --max-per-schema 5` or reduce seeds in sampling_request.yaml temporarily.
