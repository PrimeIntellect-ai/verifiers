"""
Amazon Shopping Environment — kernel-sampled CUA tasks.

Subclasses FullBrowseEnv to add:
- Kernel-sampled dataset loading (from depgraph BFS)
- Entity generation per task (from depgraph entity_sampler)
- POST /api/init to seed the Next.js app with task entities
- submit_result tool for structured output
- Scoring against ground truth entities

Usage:
    prime eval run full-browse -m openai/gpt-4.1-mini \\
        -b https://api.openai.com/v1 -k OPENAI_API_KEY \\
        -a '{"domain": "amazon_shopping"}'
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

from full_browse import FullBrowseEnv, FULL_BROWSE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Path to the Next.js app (relative to this file)
_APP_PATH = Path(__file__).parent / "app"

# System prompt tailored for Amazon shopping tasks
AMAZON_SYSTEM_PROMPT = (
    FULL_BROWSE_SYSTEM_PROMPT
    + """

## Task Context
You are browsing a simulated Amazon.com to complete a shopping research task.
The browser starts on the Amazon home page. Use the search bar, navigate
categories, and explore product listings to find the information requested.

When you have gathered all the required information, call the submit_result
tool with a JSON object containing your findings. The exact fields depend
on the task — read the task instructions carefully.
"""
)

# Non-monotonic goal paths to skip in scoring (navigation artifacts)
SKIP_GOAL_PATHS = {"page.type", "page.current_product"}


def _build_dataset_from_kernel(
    tasks_path: str | Path | None = None,
    max_tasks: int | None = None,
) -> Dataset:
    """Load kernel-sampled tasks and generate entities for each.

    Imports from depgraph's domain_runtime. Falls back to a minimal
    inline dataset if depgraph is not installed.
    """
    try:
        from depgraph.domain_runtime.amazon_shopping.dataset import build_dataset
    except ImportError:
        logger.warning(
            "depgraph not installed — using minimal inline dataset. "
            "Install depgraph for kernel-sampled tasks."
        )
        return _minimal_dataset()

    raw_tasks = build_dataset(tasks_path=tasks_path, max_tasks=max_tasks)

    return Dataset.from_list(
        [
            {
                "prompt": task["prompt"],
                "answer": json.dumps(
                    task.get("answer_key", task["entities"]), default=str
                ),
                "task_id": task["task_id"],
                "info": {
                    "entities": task["entities"],
                    "answer_key": task.get("answer_key", {}),
                    "start_world": task["start_world"],
                    "goal_world": task.get("goal_world", []),
                    "description": task["description"],
                    "required_actions": task.get("required_actions", []),
                },
            }
            for task in raw_tasks
        ]
    )


def _minimal_dataset() -> Dataset:
    """Fallback dataset for testing without depgraph."""
    entities = {
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
                    "40hr battery",
                    "Bluetooth 5.3",
                ],
                "asin": "B0TEST001",
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
                        "text": "Incredible sound quality and the noise cancellation is top notch.",
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
                ],
                "qa_pairs": [
                    {
                        "question": "Does this work with iPhone?",
                        "answer": "Yes, works with all Bluetooth devices.",
                        "votes": 42,
                    },
                ],
            },
        ],
        "deals": [],
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
    }

    description = (
        "Search for 'wireless headphones' on Amazon. Find the 'Wireless Headphones Pro' "
        "by AudioTech. Report the product name, price, rating, and whether it has Prime shipping."
    )

    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": description}]],
            "answer": [
                json.dumps(
                    {
                        "name": "Wireless Headphones Pro",
                        "price": "$79.99",
                        "rating": "4.5",
                        "prime": True,
                    }
                )
            ],
            "task_id": ["amazon-test-001"],
            "info": [
                {
                    "entities": entities,
                    "start_world": {
                        "page.type": "home",
                        "task.entry_point": "search",
                        "task.requires_detail": True,
                        "task.requires_shipping": False,
                        "task.requires_reviews": False,
                        "task.requires_variants": False,
                        "task.requires_filters": False,
                        "task.requires_qa": False,
                        "task.requires_cart": False,
                        "task.num_products": 1,
                    },
                    "goal_world": [],
                    "description": description,
                    "required_actions": [],
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating a browser automation agent that was given a shopping research task on a simulated Amazon website.

## Task Given to the Agent
```
{question}
```

## Answer Key (ground truth — the correct values the agent should have found)
```
{answer}
```

## Agent's Submitted Result
```
{response}
```

## Evaluation Instructions

Compare the agent's submission against the answer key field by field:

1. **Product names**: Must match exactly (minor formatting differences OK).
2. **Prices**: Must match the correct dollar amount (e.g., "$123.45"). Small rounding differences are OK.
3. **Ratings**: Must match the numeric rating (e.g., 4.3 or 4.3/5).
4. **Boolean fields** (prime_eligible, verified, etc.): Must be correct.
5. **Shipping/delivery**: Must include correct delivery time and cost for the correct ZIP code.
6. **Reviews**: If reviews were requested, agent should mention key review details (reviewer sentiment, ratings).
7. **Seller info**: If seller was requested, agent should include seller name and feedback.

The agent's response may be formatted differently, use different field names, or include extra information. That's fine. What matters is whether the core facts match the answer key.

Score the submission:
- "yes" if the agent correctly reported ALL the key facts from the answer key
- "partial" if the agent got SOME facts right but missed or got wrong others
- "no" if the agent failed to find the correct information or reported wrong values

Respond with ONLY one of: "yes", "partial", or "no"."""


async def judge_submission(
    judge,
    prompt: str | list,
    completion: str | list,
    answer: str,
    state: vf.State,
) -> float:
    """Score agent's submit_result output against focused answer key."""
    judge_response = await judge(prompt, completion, answer, state)
    response_lower = judge_response.lower().strip()
    if "yes" in response_lower:
        return 1.0
    elif "partial" in response_lower:
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AmazonShoppingEnv(FullBrowseEnv):
    """Amazon Shopping environment with kernel-sampled tasks.

    Extends FullBrowseEnv with:
    - Entity seeding via POST /api/init after sandbox startup
    - submit_result tool for structured output
    """

    def __init__(self, dataset: Dataset, rubric: vf.Rubric, **kwargs):
        # Default to our Next.js app
        kwargs.setdefault("app_path", str(_APP_PATH))
        super().__init__(dataset=dataset, rubric=rubric, **kwargs)

        # Register the submit_result tool
        self.add_tool(self._submit_result, args_to_skip=["state"])

    async def _submit_result(self, state: vf.State, **result: Any) -> str:
        """Submit structured results for the task.

        Call this when you have gathered all the requested information.
        Pass the results as keyword arguments matching the task requirements.
        """
        state["submitted_result"] = result
        return json.dumps({"submitted": True, "result": result})

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Initialize sandbox, then seed the app with task entities."""
        state = await super().setup_state(state, **kwargs)

        # Seed the Next.js app with this task's entities
        info = state.get("info", {})
        entities = info.get("entities", {})
        start_world = info.get("start_world", {})

        if entities:
            sandbox_id = state.get("cua_sandbox_id", "")
            if sandbox_id:
                payload = json.dumps({"entities": entities, "start_world": start_world})
                # Use curl inside the sandbox to call the app's init endpoint
                cmd = (
                    f"curl -s -X POST http://localhost:3000/api/init "
                    f"-H 'Content-Type: application/json' "
                    f"-d '{payload}'"
                )
                try:
                    result = await self._full_browse_mode._execute_sandbox_command(
                        sandbox_id, cmd, timeout=30
                    )
                    if self.logger:
                        self.logger.info(f"App init result: {result[:200]}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to seed app: {e}")

        return state


# ---------------------------------------------------------------------------
# load_environment — entry point for `prime eval run`
# ---------------------------------------------------------------------------


def load_environment(
    max_turns: int = 25,
    judge_model: str = "gpt-4o-mini",
    system_prompt: str = AMAZON_SYSTEM_PROMPT,
    # Dataset configuration
    tasks_path: str | None = None,
    max_tasks: int | None = None,
    # All other kwargs forwarded to AmazonShoppingEnv / FullBrowseEnv
    **kwargs,
) -> vf.Environment:
    """Load the Amazon Shopping browser environment.

    Uses kernel-sampled tasks from depgraph if available, otherwise
    falls back to a minimal inline dataset for testing.

    Args:
        max_turns: Maximum agent turns per task.
        judge_model: Model for scoring agent output.
        system_prompt: System prompt for the agent.
        tasks_path: Path to task_specs.sampled.yaml (optional).
        max_tasks: Limit number of tasks (optional).
        **kwargs: Forwarded to AmazonShoppingEnv.

    Returns:
        Configured AmazonShoppingEnv.
    """
    dataset = _build_dataset_from_kernel(
        tasks_path=tasks_path,
        max_tasks=max_tasks,
    )

    rubric = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )
    rubric.add_reward_func(judge_submission, weight=1.0)

    return AmazonShoppingEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        system_prompt=system_prompt,
        **kwargs,
    )
