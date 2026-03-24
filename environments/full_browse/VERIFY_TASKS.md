# End-to-End Task Verification Runbook (Phase C)

## Purpose

Verify that complete kernel-sampled tasks are solvable in the browser sim. Each test walks through a real task's `required_actions` sequence with real entity data, confirming the agent could navigate the app and extract the information the task asks for.

This is the strongest verification of correct-by-construction: if the kernel's BFS plan is executable in the browser and the entity data is extractable at each step, the task is solvable.

## Prerequisites

1. depgraph installed with amazon_shopping domain
2. Next.js app built and running (port 3099)
3. Chrome MCP tools available
4. Entity sampler functional (`depgraph.domain_runtime.amazon_shopping.entity_sampler`)

## Task Selection Protocol

**You MUST test tasks that collectively cover ALL exercised kernel actions.** Do not hand-pick tasks. Run this script to find the minimum covering set:

```python
from depgraph.loaders import load_graph_contract, load_sampling_request
from depgraph.kernel.sampler import sample_task_intents
from collections import Counter

contract = load_graph_contract("domains/amazon_shopping/graph_contract.yaml")
request = load_sampling_request("domains/amazon_shopping/sampling_request.yaml")
tasks = sample_task_intents(contract, request)

# Greedy minimum covering set
covered = set()
all_actions = set()
for st in tasks:
    for a in st.task.required_actions:
        all_actions.add(a)

remaining = set(all_actions)
selected = []
task_list = [(st.task, set(st.task.required_actions)) for st in tasks]

while remaining:
    best_task, best_actions = max(task_list, key=lambda t: len(t[1] & remaining))
    newly_covered = best_actions & remaining
    if not newly_covered:
        break
    selected.append(best_task)
    remaining -= newly_covered

print(f"Need {len(selected)} tasks to cover {len(all_actions)} actions")
for t in selected:
    print(f"  {t.task_id} (profile={t.terminal_profile_id}, actions={t.required_actions})")
```

Additionally, select one task per terminal profile to ensure all storyline types are covered:
- `submitted_surface` — search and report
- `submitted_with_detail` — single product deep dive
- `submitted_comparison` — multi-product comparison
- `submitted_from_category` — category browsing
- `submitted_from_deals` — deal hunting
- `submitted_with_cart` — cart verification

## Per-Task Verification Protocol

For EACH selected task, execute these steps IN ORDER. Do not skip any step.

### Step 1: Load task and generate entities

```python
from depgraph.domain_runtime.amazon_shopping.dataset import build_task
from depgraph.domain_runtime.amazon_shopping.entity_sampler import sample_entities

task = <load the specific task from sampled tasks>
task_data = build_task(task)
entities = task_data["entities"]
start_world = task_data["start_world"]
description = task_data["description"]
required_actions = task_data["required_actions"]
```

Record:
- Task ID
- Terminal profile
- Required actions list
- Task description (what the agent is asked to do)
- Key entity values (product names, prices, specific data points the description asks for)

### Step 2: Seed the app

```
POST /api/init with {entities, start_world}
```

Verify `/api/entities` returns the correct number of products.

### Step 3: Walk through required_actions in order

For each action in `required_actions`, execute it in the browser and verify.

**For navigation actions** (navigate_to_amazon, navigate_to_category, navigate_to_deals):
- Navigate to the target page URL
- Verify the page loads with entity content
- Record: URL landed on, key content visible

**For search actions** (enter_search_query, submit_search, search_within_category, refine_search):
- Type the search query from entities into the search input
- Submit the form
- Verify search results page loads with entity products
- Record: URL, products visible, count

**For click-to-navigate actions** (click_product_1/2/3, click_deal_product):
- Find the element on the current page
- Click it
- Verify navigation to the correct product page
- Verify the correct product's entity data renders
- Record: source page, target page, product name visible

**For observation actions** (read_results_text, read_p1_detail, read_category_listings, etc.):
- On the current page, verify the information the task description asks about is visible
- Check specific entity values: product name, price, rating, features
- Record: which entity fields are visible, any missing

**For interaction actions** (check_p*_shipping, check_p*_reviews, check_p*_seller, check_p*_qa, select_p1_variant_a/b, add_p*_to_cart):
- Find the element
- Execute the interaction (click to expand, type ZIP, etc.)
- Verify the result shows entity-specific data
- Record: element found, interaction result, entity data visible

**For filter/sort actions** (filter_*, sort_*):
- Find the element on search results page
- Click it
- Verify URL changes and results update
- Record: URL params, product order change

**For compile_results and submit_results**:
- compile_results: verify entity data is still visible for extraction
- submit_results: this is a tool_call, not browser — record as PASS

### Step 4: Verify task solvability

After walking through all required_actions, answer:

1. **Were all actions executable?** Every element found, every interaction worked?
2. **Was the task description answerable?** Could an agent reading the pages extract the specific information the description asks for?
3. **Were entity values correct?** Did the app render the entity sampler's data faithfully (prices, names, ratings, shipping, reviews, etc.)?

Record PASS or FAIL with specific evidence for each question.

## Recording Format

For each task, produce this record:

```
=== TASK: {task_id} ===
Profile: {terminal_profile_id}
Description: {task_description}
Required actions: {list}
Key entity values: {product names, prices, specific data points}

Step-by-step execution:
  1. {action_id}: {what you did} → {what happened} → PASS/FAIL
  2. {action_id}: {what you did} → {what happened} → PASS/FAIL
  ...

Solvability:
  All actions executable: YES/NO (if NO, which failed?)
  Task answerable: YES/NO (if NO, what information is missing?)
  Entity values correct: YES/NO (if NO, which values are wrong?)

RESULT: PASS / FAIL
```

## Failure Handling

If ANY step fails:
1. Record the exact failure with evidence (screenshot, DOM state, error)
2. Determine root cause: app bug? entity sampler issue? kernel contract issue?
3. Fix the issue
4. RE-RUN the entire task from Step 1 (do not resume mid-task)

## Completion Criteria

Phase C is complete when:
- [ ] All selected covering-set tasks pass
- [ ] At least one task per terminal profile passes
- [ ] Zero actions in the covering set failed
- [ ] Every task's description is answerable from the rendered entity data

## Known Constraints

- Product detail page is client-rendered — wait 3 seconds after navigation
- Shipping ZIP input requires native value setting (React controlled component workaround)
- QA section and variant options only render when entity data includes them — this is correct, not a bug
- The entity sampler only generates QA/variants/reviews when the task's `start_world` flags require them
