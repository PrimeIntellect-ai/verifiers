"""
Amazon Shopping — CUA browser environment with kernel-sampled tasks.

Usage:
    prime env install amazon
    prime eval run amazon -m anthropic/claude-sonnet-4.6
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import yaml
import verifiers as vf
from verifiers.envs.integrations.browser_env.modes.full_browse_mode import (
    FullBrowseMode,
)
from verifiers.rubrics.judge_rubric import JudgeRubric
from datasets import Dataset
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_DIR = Path(__file__).parent
_APP_PATH = _DIR / "app"
_SAMPLED_TASKS_PATH = _DIR / "task_specs.sampled.yaml"


# ═══════════════════════════════════════════════════════════════════════
# LocalBrowserEnv — reusable base for any local-browser CUA environment
# ═══════════════════════════════════════════════════════════════════════

BROWSER_SYSTEM_PROMPT = """You are a browser automation agent that can interact with web pages using a rich set of tools.

## Tools

### computer
Execute low-level browser actions. Pass a list of action objects.
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
Get the page's element tree with refs and coordinates. Use filter="interactive" to see only buttons, links, inputs.

### find
Search for elements matching a query. Returns refs and coordinates.

### form_input
Set a form field value using its ref: form_input(ref="ref_42", value="hello")

### submit_result
Submit your findings as a JSON string when you have completed the task.

IMPORTANT: There is no 'goto' tool. The browser starts at the application's home page.
After each action you will receive updated page state. Analyze it to determine your next action.
"""


class LocalBrowserEnv(vf.StatefulToolEnv):
    """Base environment for local-browser CUA tasks.

    Handles all browser plumbing — sandbox, CUA server, tool registration,
    screenshot filtering, session lifecycle. Subclass or instantiate with
    a dataset, rubric, app_path, and system_prompt.
    """

    def __init__(
        self,
        dataset: Dataset,
        rubric: vf.Rubric,
        system_prompt: str = BROWSER_SYSTEM_PROMPT,
        app_path: str | Path = _APP_PATH,
        max_turns: int = 40,
        # Sandbox defaults — rarely need changing
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
            app_path=str(app_path),
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

        self._browse_mode.register_tools(self)
        self.add_tool(self._submit_result, args_to_skip=["state"])

    async def _submit_result(self, state: vf.State, data: str = "") -> str:
        """Submit your findings for the task.

        Call this when you have gathered all the requested information.
        Pass a JSON string with the results.
        """
        state["submitted_result"] = data
        return json.dumps({"submitted": True})

    @vf.stop
    async def result_submitted(self, state: vf.State) -> bool:
        """Stop the rollout once the agent has submitted results."""
        return bool(state.get("submitted_result"))

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

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        if tool_name == "_submit_result":
            # Only inject state, don't add browse-mode args (session_id etc.)
            tool_args["state"] = state
            return tool_args
        return self._browse_mode.update_tool_args(
            tool_name, tool_args, messages, state, **kwargs
        )

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        messages = await super().get_prompt_messages(state)
        return self._browse_mode.filter_screenshots_in_messages(list(messages))

    @vf.cleanup
    async def cleanup_session(self, state: vf.State) -> None:
        await self._browse_mode.cleanup_session(state)

    @vf.teardown
    async def teardown(self) -> None:
        if hasattr(self, "_browse_mode") and self._browse_mode is not None:
            await self._browse_mode.teardown()


# ═══════════════════════════════════════════════════════════════════════
# Amazon-specific: dataset, prompts, answer keys, reward
# ═══════════════════════════════════════════════════════════════════════

AMAZON_SYSTEM_PROMPT = (
    BROWSER_SYSTEM_PROMPT
    + """
## Task Context
You are browsing a simulated Amazon.com to complete a shopping research task.
The browser starts on the Amazon home page. Use the search bar, navigate
categories, and explore product listings to find the information requested.
When done, call submit_result with a JSON string containing your findings.
"""
)

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
2. **Prices**: Must match the correct dollar amount. Small rounding differences OK.
3. **Ratings**: Must match the numeric rating (e.g., 4.3 or 4.3/5).
4. **Boolean fields** (prime_eligible, verified, etc.): Must be correct.
5. **Shipping/delivery**: Must include correct delivery time and cost for the correct ZIP code.
6. **Reviews**: If requested, agent should mention key review details.
7. **Seller info**: If requested, agent should include seller name and feedback.

The agent's response may be formatted differently or include extra information. What matters is whether the core facts match the answer key.

Score the submission:
- "yes" if the agent correctly reported ALL the key facts from the answer key
- "partial" if the agent got SOME facts right but missed or got wrong others
- "no" if the agent failed to find the correct information or reported wrong values

IMPORTANT: Your response MUST start with exactly one word: yes, partial, or no. Then optionally explain."""


def _load_sampled_tasks(path: Path | None = None) -> list[dict]:
    """Load BFS-sampled tasks from YAML."""
    path = path or _SAMPLED_TASKS_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("tasks", [])


def _build_dataset(max_tasks: int | None = None) -> Dataset:
    """Build HF dataset from sampled tasks + entity generation."""
    from entity_sampler import sample_entities
    from task_spec import generate_task_spec

    sampled = _load_sampled_tasks()
    if max_tasks:
        sampled = sampled[:max_tasks]

    rows = []
    for task in sampled:
        task_id = task["task_id"]
        start_world = {e["path"]: e["set"] for e in task.get("start_world", [])}
        goal_world = [
            g
            for g in task.get("goal_world", [])
            if g["path"] not in {"page.type", "page.current_product"}
        ]

        entities = sample_entities(task_id, start_world)
        description, answer_key = generate_task_spec(
            task_id, start_world, goal_world, entities
        )

        rows.append(
            {
                "prompt": [{"role": "user", "content": description}],
                "answer": json.dumps(answer_key, default=str),
                "task_id": task_id,
                "info": {
                    "entities": entities.model_dump(),
                    "answer_key": answer_key,
                    "start_world": start_world,
                    "description": description,
                },
            }
        )

    return Dataset.from_list(rows)


# ═══════════════════════════════════════════════════════════════════════
# load_environment — entry point for `prime eval run amazon`
# ═══════════════════════════════════════════════════════════════════════


def load_environment(
    max_turns: int = 40,
    max_tasks: int | None = None,
    judge_model: str = "anthropic/claude-sonnet-4.6",
    judge_base_url: str = "https://api.pinference.ai/api/v1",
    judge_api_key_var: str = "PRIME_API_KEY",
    **kwargs,
) -> vf.Environment:
    dataset = _build_dataset(max_tasks=max_tasks)

    judge_client = AsyncOpenAI(
        base_url=judge_base_url,
        api_key=os.environ.get(judge_api_key_var, ""),
    )

    judge_rubric = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )

    async def judge_reward_func(judge, prompt, completion, answer, state) -> float:
        submitted = state.get("submitted_result", "") if state else ""

        if not submitted:
            return 0.0

        # Inject submitted data into completion so the judge's parser sees it
        patched_completion = [{"role": "assistant", "content": submitted}]
        judge_response = await judge(prompt, patched_completion, answer, state)

        response_lower = judge_response.lower().strip()
        first_word = response_lower.split()[0] if response_lower.split() else ""

        if first_word == "yes":
            return 1.0
        elif first_word == "partial":
            return 0.5
        elif first_word == "no":
            return 0.0
        elif (
            "correctly reported" in response_lower
            or "all the key facts" in response_lower
        ):
            return 1.0
        elif "missed" in response_lower or "partially" in response_lower:
            return 0.5
        return 0.0

    env = LocalBrowserEnv(
        dataset=dataset,
        rubric=judge_rubric,
        system_prompt=AMAZON_SYSTEM_PROMPT,
        app_path=_APP_PATH,
        max_turns=max_turns,
        **kwargs,
    )

    # Add reward func after construction — StatefulToolEnv's init chain
    # wraps the rubric in a RubricGroup, so we add to the first sub-rubric
    if hasattr(env.rubric, "rubrics"):
        env.rubric.rubrics[0] = judge_rubric
        judge_rubric.add_reward_func(judge_reward_func, weight=1.0)
    else:
        env.rubric.add_reward_func(judge_reward_func, weight=1.0)

    return env
