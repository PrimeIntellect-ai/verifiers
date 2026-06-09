from typing import Literal

import verifiers as vf
from verifiers.v1.toolsets.browser import (
    BrowserbaseBackend,
    CDPBackend,
    Mode,
    browser_toolset,
)
from verifiers.v1.utils.judge_utils import clamp_float, parse_judge_json

Backend = Literal["browserbase", "cdp"]

SYSTEM_PROMPT = """You are an autonomous web-browsing agent operating a real Chrome browser.

You interact with the page only through the provided browser tools. After each
action you receive a screenshot of the current page — always base your next
action on the most recent screenshot. Coordinates are pixels within the viewport.

Workflow:
- Use `navigate` to open the start URL given in the task.
- Use the click / type / key / scroll actions (or the `computer` tool) to
  interact with the page.
- When you have completed the task, reply with your final answer as plain text
  and DO NOT call any tool. That plain-text reply is your submission.

Be efficient and use the minimum number of actions necessary."""

JUDGE_SYSTEM_PROMPT = """You grade a web agent's answer to a browsing task.

Respond with compact JSON only: {"score": 0.0-1.0, "reason": "..."}
- 1.0: the answer correctly and completely accomplishes the task.
- 0.5: partially correct or plausible but incomplete / unverified.
- 0.0: wrong, empty, or unrelated to the task."""


TASKS: list[vf.ConfigData] = [
    {
        "task_id": "example-domain-heading",
        "start_url": "https://example.com",
        "question": "Open the page and report the exact text of the page's main heading.",
        "answer": "Example Domain",
    },
    {
        "task_id": "wolfram-derivative",
        "start_url": "https://www.wolframalpha.com",
        "question": (
            "Use Wolfram Alpha to compute the derivative of x^2 evaluated at "
            "x = 5.6, and report the numeric result."
        ),
        "answer": "11.2",
    },
    {
        "task_id": "arxiv-latest-quantum",
        "start_url": "https://arxiv.org",
        "question": (
            "Search arXiv for the latest preprints about 'quantum computing' "
            "(sorted newest first) and report the title of the top result."
        ),
        "answer": (
            "Criteria: any plausible, real arXiv paper title related to quantum "
            "computing that appears at the top of a newest-first search."
        ),
    },
]


class BrowserToolsetTasksetConfig(vf.TasksetConfig):
    rewards: list[str] = ["task_success"]
    system_prompt: str = SYSTEM_PROMPT
    num_examples: int = -1


class BrowserToolsetHarnessConfig(vf.HarnessConfig):
    max_turns: int = 15
    # Backend: "browserbase" (managed, isolated session per rollout) or "cdp"
    # (connect to any browser exposing a CDP endpoint).
    backend: Backend = "browserbase"
    cdp_url: str | None = None
    proxies: bool = False
    # Tool surface: "computer" | "decomposed" | "both".
    mode: Mode = "both"
    viewport_width: int = 1280
    viewport_height: int = 800


class BrowserToolsetEnvConfig(vf.EnvConfig):
    taskset: BrowserToolsetTasksetConfig = BrowserToolsetTasksetConfig()
    harness: BrowserToolsetHarnessConfig = BrowserToolsetHarnessConfig()


def _score_prompt(task: vf.Task, answer: str) -> str:
    return (
        f"Task: {task['question']}\n"
        f"Start URL: {task['start_url']}\n"
        f"Reference answer / criteria: {task['answer']}\n\n"
        f"Agent's final answer:\n{answer}\n\n"
        "Score how well the agent accomplished the task as compact JSON."
    )


@vf.reward(weight=1.0)
async def task_success(task, state) -> float:
    messages = vf.get_messages(state.get("completion") or [], role="assistant")
    answer = str(messages[-1].content or "") if messages else ""
    if not answer:
        return 0.0
    judge_task = vf.Task(
        {
            "prompt": [{"role": "user", "content": _score_prompt(task, answer)}],
            "max_turns": 1,
        }
    ).freeze()
    judge_state = state.for_task(judge_task, borrow="model")
    judge_state = await vf.Harness(
        config=vf.HarnessConfig(system_prompt=JUDGE_SYSTEM_PROMPT, max_turns=1)
    ).run(judge_task, judge_state)
    judge_messages = vf.get_messages(
        judge_state.get("completion") or [], role="assistant"
    )
    judge_text = str(judge_messages[-1].content or "") if judge_messages else ""
    parsed = parse_judge_json(judge_text)
    score = clamp_float(parsed.get("score", 0.0))
    state["judge"] = {"score": score, "reason": str(parsed.get("reason", ""))}
    return score


def load_browser_toolset(
    backend: Backend = "browserbase",
    cdp_url: str | None = None,
    proxies: bool = False,
    mode: Mode = "both",
    viewport_width: int = 1280,
    viewport_height: int = 800,
) -> vf.Toolset:
    if backend == "browserbase":
        browser_backend = BrowserbaseBackend(proxies=proxies)
    elif backend == "cdp":
        browser_backend = CDPBackend(cdp_url=cdp_url)
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'browserbase' or 'cdp'.")
    return browser_toolset(
        backend=browser_backend,
        mode=mode,
        width=viewport_width,
        height=viewport_height,
    )


def load_tasks(num_examples: int = -1):
    rows = TASKS if num_examples < 0 else TASKS[:num_examples]
    for index, row in enumerate(rows):
        yield {
            **row,
            "example_id": index,
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{row['question']}\n\nStart by navigating to: {row['start_url']}"
                    ),
                }
            ],
        }


class BrowserToolsetTaskset(vf.Taskset[BrowserToolsetTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(num_examples=self.config.num_examples)


class BrowserToolsetHarness(vf.Harness[BrowserToolsetHarnessConfig]):
    pass


def load_environment(config: BrowserToolsetEnvConfig) -> vf.Env:
    if config.harness.backend == "browserbase":
        vf.ensure_keys(["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"])
    harness = BrowserToolsetHarness(config=config.harness)
    if "toolsets" not in config.harness.model_fields_set:
        harness.add_toolset(
            {
                "browser": load_browser_toolset(
                    backend=config.harness.backend,
                    cdp_url=config.harness.cdp_url,
                    proxies=config.harness.proxies,
                    mode=config.harness.mode,
                    viewport_width=config.harness.viewport_width,
                    viewport_height=config.harness.viewport_height,
                )
            }
        )
    return vf.Env(
        taskset=BrowserToolsetTaskset(config=config.taskset),
        harness=harness,
    )
