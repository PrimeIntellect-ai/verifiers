"""
Amazon Shopping — kernel-sampled CUA browser tasks.

Uses the full_browse environment's FullBrowseMode for browser automation,
with a Next.js Amazon clone seeded per-task with entity data from the
depgraph kernel's BFS sampler.

Usage:
    prime env install amazon-shopping
    prime eval run amazon-shopping -m anthropic/claude-opus-4-6

    # With custom model/backend:
    prime eval run amazon-shopping -m openai/gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import verifiers as vf
from verifiers.envs.integrations.browser_env.modes.full_browse_mode import (
    FullBrowseMode,
)
from datasets import Dataset
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# The Next.js app lives in the full_browse environment directory
_FULL_BROWSE_DIR = Path(__file__).parent.parent / "full_browse"
_APP_PATH = _FULL_BROWSE_DIR / "app"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a browser automation agent that can interact with web pages using a rich set of tools.

## Tools

### computer
Execute low-level browser actions. Pass a JSON string containing a list of action objects.
Each action has an "action" key. Supported actions:
- left_click: {"action": "left_click", "coordinate": [x, y]}
- type: {"action": "type", "text": "hello"}
- key: {"action": "key", "key": "Enter"}
- scroll: {"action": "scroll", "coordinate": [x, y], "direction": "up"|"down"}
- wait: {"action": "wait", "duration": 2}
- screenshot: {"action": "screenshot"}
- double_click, triple_click, right_click, back, forward

Multiple actions can be chained in one call. A screenshot is always returned.

### get_page_text
Extract the full text content of the current page.

### read_page
Get the page's element tree with refs and coordinates. Use filter="interactive" to see only buttons, links, inputs. Elements get refs (e.g. ref_42) usable with form_input.

### find
Search for elements matching a query. Returns refs and coordinates.

### form_input
Set a form field value using its ref: form_input(ref="ref_42", value="hello")

### submit_result
Submit your findings as a JSON string when you have completed the task.

IMPORTANT: There is no 'goto' tool. The browser starts at the application's home page.

## Task Context
You are browsing a simulated Amazon.com to complete a shopping research task.
The browser starts on the Amazon home page.

## Strategy
1. Start by taking a screenshot to see the page.
2. Use read_page with filter="interactive" to find clickable elements.
3. Use computer for click/type/scroll actions.
4. Use get_page_text to extract text from pages.
5. Use find to locate specific elements.
6. Use form_input for filling forms precisely.
7. When done, call submit_result with the requested information as a JSON string.
"""

# ---------------------------------------------------------------------------
# Judge prompt
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _build_dataset(
    tasks_path: str | Path | None = None,
    max_tasks: int | None = None,
) -> Dataset:
    """Load kernel-sampled tasks with entities and answer keys."""
    try:
        from domain_runtime.amazon_shopping.dataset import build_dataset
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
                    "description": task["description"],
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
                    "options": ["Black", "White"],
                    "price_deltas_cents": [0, 0],
                },
                "reviews": [
                    {
                        "reviewer_name": "Jane D.",
                        "rating": 5,
                        "title": "Excellent",
                        "text": "Best headphones.",
                        "date": "2025-01-15",
                        "verified_purchase": True,
                    }
                ],
                "qa_pairs": [
                    {"question": "Works with iPhone?", "answer": "Yes.", "votes": 42}
                ],
            },
        ],
        "deals": [],
        "categories": [
            {
                "category_id": "electronics",
                "name": "Electronics",
                "subcategories": ["Headphones"],
                "product_count": 100,
            }
        ],
        "search_query": "wireless headphones",
        "zip_code": "10001",
    }

    answer_key = {
        "task_type": "search",
        "product_1": {
            "name": "Wireless Headphones Pro",
            "brand": "AudioTech",
            "price": "$79.99",
            "rating": 4.5,
            "prime_eligible": True,
        },
    }

    description = (
        "Search for 'wireless headphones' on Amazon. Find the 'Wireless Headphones Pro' "
        "by AudioTech. Report the product name, price, rating, and whether it has Prime shipping."
    )

    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": description}]],
            "answer": [json.dumps(answer_key)],
            "task_id": ["amazon-test-001"],
            "info": [
                {
                    "entities": entities,
                    "answer_key": answer_key,
                    "start_world": {"page.type": "home", "task.entry_point": "search"},
                    "description": description,
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def _extract_submitted_data(completion) -> str:
    """Extract the data from the _submit_result tool call in the completion.

    Handles both dict-based messages (from saved results.jsonl) and
    Pydantic model messages (from live scoring in verifiers framework).
    """
    if not isinstance(completion, list):
        return str(completion)

    def _get_attr(obj, key, default=""):
        """Get attribute from dict or Pydantic model."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    for msg in completion:
        role = _get_attr(msg, "role", "")

        # Check tool_calls on assistant messages
        tool_calls = _get_attr(msg, "tool_calls", []) or []
        for tc in tool_calls:
            # tc could be a string (saved JSON), dict, or Pydantic ToolCall
            if isinstance(tc, str):
                try:
                    tc = json.loads(tc)
                except json.JSONDecodeError:
                    continue

            name = _get_attr(tc, "name", "")
            if not name:
                func = _get_attr(tc, "function", None)
                if func:
                    name = _get_attr(func, "name", "")

            if name == "_submit_result":
                args_str = _get_attr(tc, "arguments", "")
                if not args_str:
                    func = _get_attr(tc, "function", None)
                    if func:
                        args_str = _get_attr(func, "arguments", "")

                try:
                    args = (
                        json.loads(args_str) if isinstance(args_str, str) else args_str
                    )
                    if isinstance(args, dict):
                        return args.get("data", json.dumps(args))
                    return str(args)
                except (json.JSONDecodeError, TypeError):
                    return str(args_str)

        # Check content blocks (Anthropic format)
        content = _get_attr(msg, "content", "")
        if isinstance(content, list):
            for block in content:
                block_type = _get_attr(block, "type", "")
                block_name = _get_attr(block, "name", "")
                if block_name == "_submit_result" or (
                    block_type == "tool_use" and block_name == "_submit_result"
                ):
                    inp = _get_attr(block, "input", _get_attr(block, "arguments", "{}"))
                    try:
                        args = json.loads(inp) if isinstance(inp, str) else inp
                        if isinstance(args, dict):
                            return args.get("data", json.dumps(args))
                        return str(args)
                    except (json.JSONDecodeError, TypeError):
                        return str(inp)

    # Fallback: last assistant message text
    for msg in reversed(completion):
        role = _get_attr(msg, "role", "")
        if role == "assistant":
            content = _get_attr(msg, "content", "")
            if isinstance(content, str) and content:
                return content
    return ""


async def judge_submission(
    judge,
    prompt: str | list,
    completion: str | list,
    answer: str,
    state: vf.State,
) -> float:
    """Score agent's submit_result output against focused answer key."""
    submitted = _extract_submitted_data(completion)

    # Replace completion with just the submitted data so the judge sees it
    patched_completion = [
        {"role": "assistant", "content": submitted or "(no submission found)"}
    ]

    with open("/tmp/judge_debug.log", "a") as dbg:
        dbg.write(f"submitted: {submitted[:200]}\n")
        dbg.write(f"answer: {answer[:200]}\n")
        dbg.flush()
    try:
        judge_response = await judge(prompt, patched_completion, answer, state)
    except Exception as e:
        with open("/tmp/judge_debug.log", "a") as dbg:
            dbg.write(f"JUDGE ERROR: {e}\n")
        return 0.0
    response_lower = judge_response.lower().strip() if judge_response else ""
    with open("/tmp/judge_debug.log", "a") as dbg:
        dbg.write(f"judge says: '{response_lower}'\n\n")
    if "yes" in response_lower:
        return 1.0
    elif "partial" in response_lower:
        return 0.5
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class AmazonShoppingEnv(vf.StatefulToolEnv):
    """Amazon Shopping environment with kernel-sampled CUA tasks."""

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_turns: int = 40,
        system_prompt: str = SYSTEM_PROMPT,
        app_path: str | Path | None = None,
        app_port: int = 3000,
        app_start_command: str = "npm run start",
        app_build_command: str | None = "npm install && npm run build",
        cua_server_port: int = 3001,
        viewport_width: int = 1024,
        viewport_height: int = 768,
        save_screenshots: bool = False,
        keep_recent_screenshots: int | None = 2,
        cpu_cores: int = 2,
        memory_gb: int = 4,
        use_prebuilt_image: bool = True,
        prebuilt_image: str = "team-cmlr3u2er002zhr01tj8f48ts/localbrowserapp:v1.0.1",
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            max_turns=max_turns,
            stop_errors=[vf.SandboxError],
            **kwargs,
        )

        self._browse_mode = FullBrowseMode(
            app_path=app_path or str(_APP_PATH),
            app_port=app_port,
            app_start_command=app_start_command,
            app_build_command=app_build_command,
            cua_server_port=cua_server_port,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            save_screenshots=save_screenshots,
            keep_recent_screenshots=keep_recent_screenshots,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            use_prebuilt_image=use_prebuilt_image,
            prebuilt_image=prebuilt_image,
        )

        self._rubric = rubric
        self._browse_mode.register_tools(self)

        # Register submit_result tool
        self.add_tool(self._submit_result, args_to_skip=["state"])

    async def _submit_result(self, state: vf.State, data: str = "") -> str:
        """Submit your findings for the task.

        Call this when you have gathered all the requested information.
        Pass a JSON string with the results matching the task requirements.
        For example: submit_result(data='{"product_name": "...", "price": "$79.99", "rating": 4.5}')
        """
        state["submitted_result"] = data
        return json.dumps({"submitted": True})

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Initialize sandbox, then seed the app with task entities."""
        state = await self._browse_mode.setup_state(state, **kwargs)
        state = await super().setup_state(state, **kwargs)

        # Seed the Next.js app with this task's entities
        info = state.get("info", {})
        entities = info.get("entities", {})
        start_world = info.get("start_world", {})

        if entities:
            sandbox_id = state.get("cua_sandbox_id", "")
            if sandbox_id:
                payload = json.dumps({"entities": entities, "start_world": start_world})
                # Escape single quotes in payload for shell
                payload_escaped = payload.replace("'", "'\\''")
                cmd = (
                    f"curl -s -X POST http://localhost:3000/api/init "
                    f"-H 'Content-Type: application/json' "
                    f"-d '{payload_escaped}'"
                )
                try:
                    result = await self._browse_mode._execute_sandbox_command(
                        sandbox_id, cmd, timeout=30
                    )
                    if self.logger:
                        self.logger.info(f"App init: {result[:200]}")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to seed app: {e}")

        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Inject session and sandbox IDs into tool calls."""
        return self._browse_mode.update_tool_args(
            tool_name, tool_args, messages, state, **kwargs
        )

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        """Filter old screenshots from context."""
        messages = await super().get_prompt_messages(state)
        return self._browse_mode.filter_screenshots_in_messages(list(messages))

    @vf.cleanup
    async def cleanup_session(self, state: vf.State) -> None:
        """Clean up browser session and sandbox."""
        await self._browse_mode.cleanup_session(state)

    @vf.teardown
    async def teardown(self) -> None:
        """Clean up all resources."""
        if hasattr(self, "_browse_mode") and self._browse_mode is not None:
            await self._browse_mode.teardown()


# ---------------------------------------------------------------------------
# load_environment — entry point for `prime eval run`
# ---------------------------------------------------------------------------


def load_environment(
    max_turns: int = 40,
    judge_model: str = "anthropic/claude-sonnet-4.6",
    system_prompt: str = SYSTEM_PROMPT,
    tasks_path: str | None = None,
    max_tasks: int | None = None,
    **kwargs,
) -> vf.Environment:
    """Load the Amazon Shopping browser environment.

    Args:
        max_turns: Maximum agent turns per task.
        judge_model: Model for scoring agent output.
        system_prompt: System prompt for the agent.
        tasks_path: Path to task_specs.sampled.yaml (optional).
        max_tasks: Limit number of tasks (optional).
        **kwargs: Forwarded to AmazonShoppingEnv.
    """
    dataset = _build_dataset(tasks_path=tasks_path, max_tasks=max_tasks)

    _judge_client = AsyncOpenAI(
        base_url="https://api.pinference.ai/api/v1",
        api_key=os.environ.get("PRIME_API_KEY", ""),
    )
    _judge_model = judge_model

    async def _reward_func(prompt, completion, answer, state, **kwargs) -> float:
        """Reward function that calls the LLM judge directly."""
        with open("/tmp/judge_debug.log", "a") as dbg:
            dbg.write(
                f"REWARD FUNC CALLED! comp type={type(completion).__name__} len={len(completion) if isinstance(completion, list) else '?'}\n"
            )
            # Dump first and last few messages to understand format
            if isinstance(completion, list):
                for j in [0, 1, len(completion) - 2, len(completion) - 1]:
                    if 0 <= j < len(completion):
                        msg = completion[j]
                        if isinstance(msg, dict):
                            dbg.write(
                                f"  comp[{j}] role={msg.get('role')} keys={list(msg.keys())} content_type={type(msg.get('content')).__name__}\n"
                            )
                            # Check for tool use in content blocks
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                for k, block in enumerate(content):
                                    if isinstance(block, dict):
                                        dbg.write(
                                            f"    block[{k}] type={block.get('type')} name={block.get('name', '')}\n"
                                        )
                        else:
                            dbg.write(
                                f"  comp[{j}] type={type(msg).__name__} attrs={[a for a in dir(msg) if not a.startswith('_')][:10]}\n"
                            )
                            if hasattr(msg, "tool_calls"):
                                tcs = msg.tool_calls
                                if tcs:
                                    dbg.write(
                                        f"    tool_calls: {len(tcs)}, first={type(tcs[0]).__name__}\n"
                                    )
                                    tc0 = tcs[0]
                                    if hasattr(tc0, "name"):
                                        dbg.write(f"    tc0.name={tc0.name}\n")
                                    if hasattr(tc0, "function"):
                                        dbg.write(
                                            f"    tc0.function.name={tc0.function.name}\n"
                                        )
        submitted = _extract_submitted_data(completion)
        question = ""
        if isinstance(prompt, list) and prompt:
            last = prompt[-1]
            question = last.get("content", "") if isinstance(last, dict) else str(last)

        judge_prompt_filled = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            response=submitted or "(no submission found)",
        )

        with open("/tmp/judge_debug.log", "a") as dbg:
            dbg.write(f"Calling judge: model={_judge_model}\n")
            dbg.write(f"  submitted: {submitted[:200]}\n")
            dbg.write(f"  answer: {answer[:200]}\n")

        try:
            resp = await _judge_client.chat.completions.create(
                model=_judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": judge_prompt_filled
                        + "\n\nIMPORTANT: Your response MUST start with exactly one word: yes, partial, or no. Then optionally explain.",
                    },
                ],
                max_tokens=50,
            )
            judge_response = resp.choices[0].message.content or ""
        except Exception as e:
            with open("/tmp/judge_debug.log", "a") as dbg:
                dbg.write(f"  JUDGE ERROR: {e}\n")
            return 0.0

        response_lower = judge_response.lower().strip()
        with open("/tmp/judge_debug.log", "a") as dbg:
            dbg.write(f"  judge says (full): '{response_lower}'\n\n")

        # Check first and last words for clean verdict
        words = response_lower.split()
        first_word = words[0] if words else ""
        last_word = words[-1].rstrip(".!") if words else ""
        # Also check last line
        last_line = (
            response_lower.strip().split("\n")[-1].strip() if response_lower else ""
        )

        for token in [first_word, last_word, last_line]:
            if token == "yes":
                return 1.0
            elif token == "partial":
                return 0.5
            elif token == "no":
                return 0.0

        # Verbose response — look for sentiment keywords
        if (
            "correctly reported all" in response_lower
            or "all the key facts" in response_lower
            or "correctly found" in response_lower
            or "correctly reported" in response_lower
            or "match" in last_line
        ):
            return 1.0
        elif (
            "some facts right" in response_lower
            or "missed" in response_lower
            or "partially" in response_lower
            or "partial" in last_line
        ):
            return 0.5
        elif (
            "failed" in response_lower
            or "wrong" in response_lower
            or "incorrect" in response_lower
        ):
            return 0.0
        else:
            with open("/tmp/judge_debug.log", "a") as dbg:
                dbg.write("  UNPARSEABLE — defaulting to 0.0\n")
            return 0.0

    env = AmazonShoppingEnv(
        dataset=dataset,
        rubric=vf.Rubric(),
        max_turns=max_turns,
        system_prompt=system_prompt,
        **kwargs,
    )

    # Add reward function AFTER construction — the init chain replaces
    # the rubric object, so we add to the final rubric directly.
    if hasattr(env.rubric, "rubrics"):
        # RubricGroup — add to the first sub-rubric (ours)
        env.rubric.rubrics[0].add_reward_func(_reward_func, weight=1.0)
    else:
        env.rubric.add_reward_func(_reward_func, weight=1.0)
    with open("/tmp/judge_debug.log", "a") as dbg:
        dbg.write(f"ENV INIT: rubric={type(env.rubric).__name__}\n")
        dbg.write(f"  score_rollouts={env.score_rollouts}\n")
        dbg.write(
            f"  top-level reward_funcs={env.rubric._get_individual_reward_func_names()}\n"
        )
        if hasattr(env.rubric, "rubrics"):
            for i, r in enumerate(env.rubric.rubrics):
                dbg.write(
                    f"  sub-rubric[{i}]: {type(r).__name__} id={id(r)} funcs={r._get_individual_reward_func_names()}\n"
                )
        dbg.write(f"  class_objects keys={list(env.rubric.class_objects.keys())}\n")
    return env
