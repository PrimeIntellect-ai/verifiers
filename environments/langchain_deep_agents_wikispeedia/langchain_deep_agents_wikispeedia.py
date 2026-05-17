import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from typing import Protocol, cast

from datasets import Dataset
import verifiers as vf
from wiki_graph import WikiGraph, WikiPair, load_wiki_graph


class AgentMessage(Protocol):
    role: str
    content: object


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

You also have access to deep-agent scaffolding tools (`write_todos`, \
`write_file`, `read_file`, `ls`, `edit_file`, `task`). Use them when they help: \
sketch a plan with `write_todos`, jot promising bridge concepts or dead-ends \
in a file, and call `task` to spawn a focused sub-agent for a sub-search. They \
are entirely optional.

Try to be quick — think about which broader concepts connect the source to \
the target, and aim for the article that most likely lists your destination \
among its links.

When you reach the target the system will say `TARGET REACHED`. Stop calling \
tools at that point and reply with a brief confirmation."""


SYSTEM_PROMPT = system_prompt()


def format_article(wiki: WikiGraph, article: str, links_only: bool = False) -> str:
    links = wiki.get_links(article)
    links_str = ", ".join(links) if links else "(no outgoing links)"
    if links_only:
        return f"# {article}\n\nAvailable links: {links_str}"
    text = wiki.get_text(article)
    return f"# {article}\n\n{text}\n\n---\nAvailable links: {links_str}"


async def click_link(article: str, wiki: WikiGraph, state: vf.State) -> str:
    """Navigate to a linked Wikipedia article."""
    links_only = bool(state.get("links_only", False))
    current = state["current_article"]
    available = wiki.get_links(current)
    normalized = wiki.normalize_name(article)
    if normalized is None or normalized not in available:
        avail_str = ", ".join(available) if available else "(none)"
        return (
            f"'{article}' is not a valid link from '{current}'.\n"
            f"Available links: {avail_str}"
        )
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
        return "You are already at the starting article. Cannot go back."
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


class WikispeediaHarnessConfig(vf.HarnessConfig):
    turn_limit: int = 50
    timeout_seconds: float = 1200.0


class WikispeediaEnvConfig(vf.EnvConfig):
    taskset: WikispeediaTasksetConfig
    harness: WikispeediaHarnessConfig


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


async def agent_timeout(task: vf.Task, state: vf.State) -> float:
    return 1.0 if state.get("agent_timeout", False) else 0.0


def iter_tool_calls(state: vf.State) -> Iterator[str]:
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
    if name is None:
        return sum(1 for _ in iter_tool_calls(state))
    return sum(1 for tool_name in iter_tool_calls(state) if tool_name == name)


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
    tools = [click_link]
    bindings = {
        "click_link.wiki": "objects.wiki",
    }
    if allow_go_back:
        tools.append(go_back)
        bindings.update(
            {
                "go_back.wiki": "objects.wiki",
            }
        )
    return WikispeediaToolset(
        tools=tools,
        bindings=bindings,
        cache_dir=cache_dir,
        config=config,
    )


class WikispeediaToolset(vf.Toolset):
    def __init__(
        self,
        *,
        tools: list[vf.Handler],
        bindings: vf.ConfigData,
        cache_dir: str | None,
        config: vf.ToolsetConfig | None = None,
    ):
        super().__init__(
            tools=tools,
            objects={"wiki": lambda: load_wiki_graph(cache_dir)},
            bindings=bindings,
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
    clicks = 0
    invalid = 0
    completion = state.get("completion") or []
    messages = (
        vf.get_messages(completion, role="tool") if isinstance(completion, list) else []
    )
    for msg in messages:
        if getattr(msg, "name", None) != "click_link":
            continue
        clicks += 1
        content = msg.content
        if isinstance(content, str) and "is not a valid link" in content:
            invalid += 1
    return float(invalid / clicks) if clicks else 0.0


@vf.update(priority=-200)
async def restore_agent_completion(task: vf.Task, state: vf.State) -> None:
    agent_completion = state.get("agent_completion")
    if isinstance(agent_completion, list):
        state["completion"] = agent_completion


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
        info: vf.ConfigData = {
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


def serialize_agent_completion(
    messages: Sequence[AgentMessage | vf.ConfigMap],
) -> list[vf.ConfigData]:
    role_aliases = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "system": "system",
    }
    call_names: dict[str, str] = {}
    serialized: list[vf.ConfigData] = []
    for message in messages:
        if isinstance(message, Mapping):
            payload = dict(message)
        else:
            model_dump = getattr(message, "model_dump", None)
            payload = (
                model_dump(mode="json", exclude_none=True)
                if callable(model_dump)
                else {
                    "role": getattr(message, "role", None)
                    or getattr(message, "type", "assistant"),
                    "content": getattr(message, "content", str(message)),
                    "name": getattr(message, "name", None),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                    "tool_calls": getattr(message, "tool_calls", None),
                }
            )
        raw_role = payload.get("role") or payload.get("type") or "assistant"
        role = role_aliases.get(str(raw_role), str(raw_role))
        item: vf.ConfigData = {
            "role": role,
            "content": payload.get("content", ""),
        }
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                tool_call_payload = dict(tool_call)
                name = tool_call_payload.get("name")
                tool_id = tool_call_payload.get("id") or tool_call_payload.get(
                    "tool_call_id"
                )
                if isinstance(tool_id, str) and isinstance(name, str):
                    call_names[tool_id] = name
                arguments = tool_call_payload.get("arguments")
                if not isinstance(arguments, str):
                    args = tool_call_payload.get("args", {})
                    try:
                        arguments = json.dumps(args if args is not None else {})
                    except (TypeError, ValueError):
                        arguments = str(args)
                    tool_call_payload["arguments"] = arguments
                normalized_tool_calls.append(tool_call_payload)
            item["tool_calls"] = normalized_tool_calls
        name = payload.get("name")
        if isinstance(name, str):
            item["name"] = name
        tool_call_id = payload.get("tool_call_id")
        if isinstance(tool_call_id, str):
            item["tool_call_id"] = tool_call_id
            if item["role"] == "tool" and "name" not in item:
                name = call_names.get(tool_call_id)
                if name is not None:
                    item["name"] = name
        serialized.append(item)
    if serialized and serialized[0].get("role") == "user":
        return serialized[1:]
    return serialized


def langchain_navigation_tools(runtime_tools):
    from langchain_core.tools import tool

    nav_tools = []
    if "click_link" in runtime_tools:
        click_link_tool = runtime_tools["click_link"]

        @tool
        async def click_link(article: str) -> str:
            """Navigate to a linked Wikipedia article."""
            return str(await click_link_tool(article=article))

        nav_tools.append(click_link)
    if "go_back" in runtime_tools:
        go_back_tool = runtime_tools["go_back"]

        @tool
        async def go_back() -> str:
            """Undo the last click_link and return to the previous article."""
            return str(await go_back_tool())

        nav_tools.append(go_back)
    return nav_tools


def make_langchain_deep_agents_program(
    max_turns: int,
    timeout_seconds: float,
) -> Callable[[vf.Task, vf.State], Awaitable[vf.State]]:
    async def run_langchain_deep_agents_wikispeedia_program(
        task: vf.Task, state: vf.State
    ) -> vf.State:
        from deepagents import create_deep_agent
        from langchain_openai import ChatOpenAI
        from langgraph.errors import GraphRecursionError

        state["current_article"] = state["info"]["source"]
        state["path"] = [state["info"]["source"]]
        state["reached_target"] = False
        state["agent_timeout"] = False
        state["links_only"] = bool(task.get("links_only", False))

        endpoint_config = state.get_endpoint_config(api="chat")
        model = ChatOpenAI(
            model=endpoint_config["model"],
            base_url=endpoint_config["api_base"],
            api_key=endpoint_config["api_key"],
        )
        runtime_tools = state.get_tools()
        nav_tools = langchain_navigation_tools(runtime_tools)
        state_system_prompt = ""
        system_prompt_messages = state.get("system_prompt")
        if isinstance(system_prompt_messages, list):
            state_system_prompt = "\n\n".join(
                str(message.content or "")
                for message in vf.get_messages(system_prompt_messages)
            )
        agent = create_deep_agent(
            model=model,
            tools=nav_tools,
            system_prompt=state_system_prompt or SYSTEM_PROMPT,
        )
        prompt = str(cast(list[vf.ConfigData], state["prompt"])[-1]["content"])
        recursion_limit = state.get_max_turns(max_turns)
        invoke_config = (
            {"recursion_limit": recursion_limit} if recursion_limit > 0 else None
        )
        invoke = agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=invoke_config,
        )
        try:
            result = await asyncio.wait_for(invoke, timeout=timeout_seconds)
        except (TimeoutError, GraphRecursionError) as exc:
            state["agent_timeout"] = True
            state.stop(
                "agent_timeout"
                if isinstance(exc, TimeoutError)
                else "agent_recursion_limit"
            )
            state.setdefault("agent_completion", [])
            return state

        messages = result.get("messages", []) if isinstance(result, Mapping) else []
        completion = serialize_agent_completion(messages)
        state["agent_completion"] = completion
        state["completion"] = completion
        if completion:
            state["agent_result"] = str(completion[-1].get("content") or "")
        return state

    return run_langchain_deep_agents_wikispeedia_program


def load_taskset(config: WikispeediaTasksetConfig) -> vf.Taskset:
    pair_cache: dict[str, tuple[list[WikiPair], list[WikiPair]]] = {}

    def pairs() -> tuple[list[WikiPair], list[WikiPair]]:
        if "pairs" not in pair_cache:
            pair_cache["pairs"] = load_wiki_graph(config.cache_dir).split_pairs(
                train_size=config.train_size,
                eval_size=config.eval_size,
                min_dist=config.min_path_length,
                max_dist=config.max_path_length,
                eval_target_fraction=config.eval_target_fraction,
                seed=config.split_seed,
                stratify=config.stratify_path_length,
            )
        return pair_cache["pairs"]

    def build_train() -> Dataset:
        train, _ = pairs()
        return build_dataset(
            load_wiki_graph(config.cache_dir),
            train,
            links_only=config.links_only,
            max_turns=config.max_turns,
        )

    def build_eval() -> Dataset:
        _, eval_ = pairs()
        return build_dataset(
            load_wiki_graph(config.cache_dir),
            eval_,
            links_only=config.links_only,
            max_turns=config.max_turns,
        )

    rewards = [reached_target]
    metrics = [
        path_length,
        shortest_path,
        agent_timeout,
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

    return vf.Taskset(
        source=build_train,
        eval_source=build_eval,
        taskset_id="langchain-deep-agents-wikispeedia",
        system_prompt=system_prompt(allow_go_back=config.allow_go_back),
        toolsets=[
            load_toolset(cache_dir=config.cache_dir, allow_go_back=config.allow_go_back)
        ],
        rewards=rewards,
        metrics=metrics,
        config=config,
    )


class WikispeediaHarness(vf.Harness):
    def __init__(self, config: WikispeediaHarnessConfig | None = None):
        config = WikispeediaHarnessConfig(config)
        super().__init__(
            program=make_langchain_deep_agents_program(
                max_turns=config.turn_limit,
                timeout_seconds=config.timeout_seconds,
            ),
            max_turns=config.turn_limit,
            updates=[restore_agent_completion],
            config=config,
        )


def load_harness(config: WikispeediaHarnessConfig) -> WikispeediaHarness:
    return WikispeediaHarness(config=config)


def load_environment(config: WikispeediaEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
