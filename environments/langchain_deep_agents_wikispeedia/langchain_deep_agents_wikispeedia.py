import os
from collections.abc import Awaitable, Callable, Iterator, Mapping

from datasets import Dataset

import verifiers as vf
from harnesses import DeepAgents, DeepAgentsConfig
from verifiers.v1.utils.prompt_utils import normalize_system_prompt

if __package__:
    from .wiki_graph import WikiGraph, WikiPair, load_wiki_graph
else:
    from wiki_graph import WikiGraph, WikiPair, load_wiki_graph


def system_prompt(allow_go_back: bool = True) -> str:
    backtracking = (
        "Use `go_back` to undo your last click."
        if allow_go_back
        else "Backtracking is disabled, so choose each link carefully."
    )
    return f"""\
This game is easy and fun:

You are given two Wikipedia articles. Starting from the first article, your goal \
is to reach the second one, exclusively by following links in the articles you \
encounter. (For the articles you are given this is always possible.)

Each article ends with a list of `Available links: ...` — those are the only \
links you can follow. Use the `click_link` tool to navigate to one. \
{backtracking}

Try to be quick — think about which broader concepts connect the source to \
the target, and aim for the article that most likely lists your destination \
among its links.

When you reach the target the system will say `TARGET REACHED`. Stop calling \
tools at that point and reply with a brief confirmation."""


NAVIGATION_TOOL_CALLS_KEY = "navigation_tool_calls"


class WikispeediaTasksetConfig(vf.TasksetConfig):
    cache_dir: str | None = None
    min_path_length: int = 3
    max_path_length: int = 6
    train_size: int = 50_000
    eval_size: int = 1_000
    eval_target_fraction: float = 0.1
    split_seed: int = 0
    links_only: bool = False
    allow_go_back: bool = True
    max_turns: int = 50
    efficiency_weight: float = 0.0
    stratify_path_length: bool = True


class WikispeediaTaskset(vf.Taskset[WikispeediaTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(self.config, split=split)

    @vf.setup
    async def init_navigation_state(self, task: vf.Task, state: vf.State) -> None:
        state["current_article"] = state["info"]["source"]
        state["path"] = [state["info"]["source"]]
        state["reached_target"] = False
        state["links_only"] = bool(task.get("links_only", False))
        state[NAVIGATION_TOOL_CALLS_KEY] = []


def format_article(wiki: WikiGraph, article: str, links_only: bool = False) -> str:
    links = wiki.get_links(article)
    links_str = ", ".join(links) if links else "(no outgoing links)"
    if links_only:
        return f"# {article}\n\nAvailable links: {links_str}"
    text = wiki.get_text(article)
    return f"# {article}\n\n{text}\n\n---\nAvailable links: {links_str}"


def record_navigation_tool_call(state: vf.State, name: str, valid: bool) -> None:
    calls = state.get(NAVIGATION_TOOL_CALLS_KEY)
    if not isinstance(calls, list):
        calls = []
        state[NAVIGATION_TOOL_CALLS_KEY] = calls
    calls.append({"name": name, "valid": valid})


async def click_link(article: str, wiki: WikiGraph, state: vf.State) -> str:
    """Navigate to a linked Wikipedia article."""
    links_only = bool(state.get("links_only", False))
    current = state["current_article"]
    available = wiki.get_links(current)
    normalized = wiki.normalize_name(article)
    if normalized is None or normalized not in available:
        record_navigation_tool_call(state, "click_link", valid=False)
        avail_str = ", ".join(available) if available else "(none)"
        return (
            f"'{article}' is not a valid link from '{current}'.\n"
            f"Available links: {avail_str}"
        )
    record_navigation_tool_call(state, "click_link", valid=True)
    state["current_article"] = normalized
    state["path"].append(normalized)
    if normalized == state["info"]["target"]:
        state["reached_target"] = True
        state.stop("target_reached")
        return (
            f"TARGET REACHED: {normalized}\n\n"
            "You successfully navigated to the target. Stop calling tools "
            "and reply briefly to confirm."
        )
    return format_article(wiki, normalized, links_only=links_only)


async def go_back(wiki: WikiGraph, state: vf.State) -> str:
    """Undo the last click_link and return to the previous article."""
    path = state["path"]
    if len(path) <= 1:
        record_navigation_tool_call(state, "go_back", valid=False)
        return "You are already at the starting article. Cannot go back."
    record_navigation_tool_call(state, "go_back", valid=True)
    path.pop()
    state["current_article"] = path[-1]
    return format_article(
        wiki, path[-1], links_only=bool(state.get("links_only", False))
    )


DEEP_AGENT_TOOLS = {
    "write_todos",
    "write_file",
    "read_file",
    "ls",
    "edit_file",
    "grep",
    "task",
}
WIKISPEEDIA_TOOLS = {"click_link", "go_back"}


async def reached_target(task: vf.Task, state: vf.State) -> float:
    return 1.0 if state.get("reached_target", False) else 0.0


async def path_efficiency(task: vf.Task, state: vf.State) -> float:
    if not state.get("reached_target", False):
        return 0.0
    shortest = float(state["info"]["shortest_path"])
    actual = max(len(state.get("path", [])) - 1, 1)
    return min(1.0, shortest / actual)


async def path_length(task: vf.Task, state: vf.State) -> float:
    return float(max(len(state.get("path", [])) - 1, 0))


async def shortest_path(task: vf.Task, state: vf.State) -> float:
    return float(state.get("info", {}).get("shortest_path", 0))


def has_navigation_tool_log(state: vf.State) -> bool:
    return isinstance(state.get(NAVIGATION_TOOL_CALLS_KEY), list)


def iter_navigation_tool_calls(state: vf.State) -> Iterator[vf.JsonData]:
    calls = state.get(NAVIGATION_TOOL_CALLS_KEY)
    if not isinstance(calls, list):
        return
    for call in calls:
        if isinstance(call, Mapping):
            yield call


def iter_completion_tool_calls(state: vf.State) -> Iterator[str]:
    completion = state.get("completion") or []
    messages = (
        vf.get_messages(completion, role="assistant")
        if isinstance(completion, list)
        else []
    )
    for msg in messages:
        tool_calls = msg.tool_calls
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            yield tool_call.name


def count_tool_calls(state: vf.State, name: str | None = None) -> int:
    if has_navigation_tool_log(state):
        nav_count = sum(
            1
            for call in iter_navigation_tool_calls(state)
            if name is None or call.get("name") == name
        )
        if name in WIKISPEEDIA_TOOLS:
            return nav_count
        completion_count = sum(
            1
            for tool_name in iter_completion_tool_calls(state)
            if tool_name not in WIKISPEEDIA_TOOLS
            and (name is None or tool_name == name)
        )
        return nav_count + completion_count
    if name is None:
        return sum(1 for _ in iter_completion_tool_calls(state))
    return sum(
        1 for tool_name in iter_completion_tool_calls(state) if tool_name == name
    )


def make_tool_count_metric(
    name: str,
) -> Callable[[vf.Task, vf.State], Awaitable[float]]:
    async def metric(task: vf.Task, state: vf.State) -> float:
        return float(count_tool_calls(state, name))

    metric.__name__ = f"calls_{name}"
    return metric


def load_toolset(
    cache_dir: str | None = None,
    allow_go_back: bool = True,
    config: vf.ToolsetConfig | None = None,
) -> vf.Toolset:
    wiki_graph: WikiGraph | None = None

    def wiki() -> WikiGraph:
        nonlocal wiki_graph
        if wiki_graph is None:
            wiki_graph = load_wiki_graph(cache_dir)
        return wiki_graph

    async def click_link_tool(article: str, state: vf.State) -> str:
        return await click_link(article, wiki(), state)

    click_link_tool.__name__ = "click_link"
    click_link_tool.__doc__ = click_link.__doc__

    tools: list[vf.Handler] = [click_link_tool]
    if allow_go_back:

        async def go_back_tool(state: vf.State) -> str:
            return await go_back(wiki(), state)

        go_back_tool.__name__ = "go_back"
        go_back_tool.__doc__ = go_back.__doc__
        tools.append(go_back_tool)
    return vf.Toolset(
        tools=tools,
        config=config,
    )


async def total_tool_calls(task: vf.Task, state: vf.State) -> float:
    return float(count_tool_calls(state))


async def assistant_turns(task: vf.Task, state: vf.State) -> float:
    completion = state.get("completion") or []
    return float(
        len(vf.get_messages(completion, role="assistant"))
        if isinstance(completion, list)
        else 0
    )


async def invalid_link_rate(task: vf.Task, state: vf.State) -> float:
    if has_navigation_tool_log(state):
        click_calls = [
            call
            for call in iter_navigation_tool_calls(state)
            if call.get("name") == "click_link"
        ]
        invalid = sum(1 for call in click_calls if call.get("valid") is False)
        return float(invalid / len(click_calls)) if click_calls else 0.0

    clicks = 0
    invalid = 0
    completion = state.get("completion") or []
    if not isinstance(completion, list):
        return 0.0

    transcript = vf.get_messages(completion)
    id_to_name: dict[str, str] = {}
    for msg in transcript:
        if msg.role == "assistant":
            tool_calls = msg.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    id_to_name[tc.id] = tc.name

    for msg in transcript:
        if msg.role != "tool":
            continue
        tool_name = id_to_name.get(msg.tool_call_id)
        if tool_name is None:
            extra = msg.get("name")
            tool_name = extra if isinstance(extra, str) else None
        if tool_name != "click_link":
            continue
        clicks += 1
        content = msg.content
        if isinstance(content, str) and "is not a valid link" in content:
            invalid += 1
    return float(invalid / clicks) if clicks else 0.0


def build_dataset(
    wiki: WikiGraph,
    pairs: list[WikiPair],
    links_only: bool,
    max_turns: int,
) -> Dataset:
    records = []
    for source, target, dist in pairs:
        starting = format_article(wiki, source, links_only=links_only)
        prompt_text = (
            f"Your mission: {source} >> {target}\n\n"
            f"Here is the starting article:\n\n{starting}"
        )
        info: vf.JsonData = {
            "source": source,
            "target": target,
            "shortest_path": dist,
        }
        human = wiki.get_human_stats(source, target)
        if human is not None:
            info.update(human)
        records.append(
            {
                "task_id": f"{source}->{target}",
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": target,
                "info": info,
                "links_only": links_only,
                "max_turns": max_turns,
            }
        )
    return Dataset.from_list(records)


def split_pairs(
    config: WikispeediaTasksetConfig,
) -> tuple[list[WikiPair], list[WikiPair]]:
    return load_wiki_graph(config.cache_dir).split_pairs(
        train_size=config.train_size,
        eval_size=config.eval_size,
        min_dist=config.min_path_length,
        max_dist=config.max_path_length,
        eval_target_fraction=config.eval_target_fraction,
        seed=config.split_seed,
        stratify=config.stratify_path_length,
    )


def load_tasks(
    config: WikispeediaTasksetConfig, split: vf.TaskSplit = "train"
) -> Dataset:
    train, eval_ = split_pairs(config)
    return build_dataset(
        load_wiki_graph(config.cache_dir),
        train if split == "train" else eval_,
        links_only=config.links_only,
        max_turns=config.max_turns,
    )


def load_taskset(
    config: WikispeediaTasksetConfig,
) -> WikispeediaTaskset:
    rewards = [reached_target]
    metrics = [
        path_length,
        shortest_path,
        total_tool_calls,
        assistant_turns,
        invalid_link_rate,
        *[
            make_tool_count_metric(name)
            for name in sorted(DEEP_AGENT_TOOLS | WIKISPEEDIA_TOOLS)
        ],
    ]
    if config.efficiency_weight > 0:

        async def weighted_path_efficiency(task: vf.Task, state: vf.State) -> float:
            return await path_efficiency(task, state)

        weighted_path_efficiency.__name__ = "path_efficiency"
        rewards.append(
            vf.reward(weight=config.efficiency_weight)(weighted_path_efficiency)
        )
    else:
        metrics.insert(0, path_efficiency)

    taskset = WikispeediaTaskset(config=config)
    taskset.taskset_id = "langchain-deep-agents-wikispeedia"
    taskset.system_prompt = normalize_system_prompt(
        system_prompt(allow_go_back=config.allow_go_back),
        field_name="taskset.system_prompt",
    )
    taskset.add_toolset(
        load_toolset(
            cache_dir=config.cache_dir,
            allow_go_back=config.allow_go_back,
        )
    )
    for reward in rewards:
        taskset.add_reward(reward)
    for metric in metrics:
        taskset.add_metric(metric)
    return taskset


def load_harness(config: DeepAgentsConfig) -> DeepAgents:
    return DeepAgents(config=config)


class WikispeediaEnvConfig(vf.EnvConfig):
    taskset: WikispeediaTasksetConfig = WikispeediaTasksetConfig()
    harness: DeepAgentsConfig = DeepAgentsConfig(
        agent_name="wikispeedia-navigator",
        max_turns=50,
        timeout_seconds=1200.0,
        system_prompt_strategy="TH",
    )


def load_environment(config: WikispeediaEnvConfig) -> vf.Env:
    """Load the v1 Wikispeedia taskset with a LangChain Deep Agents harness."""
    if os.environ.get("LANGSMITH_TRACING") == "true":
        vf.ensure_keys(["LANGSMITH_API_KEY"])

    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
