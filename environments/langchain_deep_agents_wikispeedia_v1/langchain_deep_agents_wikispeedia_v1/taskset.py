import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from datasets import Dataset

import verifiers.v1 as vf

from .wiki_graph import WikiGraph, WikiPair, load_wiki_graph


class AgentMessage(Protocol):
    role: str
    content: object


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
`write_file`, `read_file`, `ls`, `edit_file`, `task`). Use them when they help.

When you reach the target the system will say `TARGET REACHED`. Stop calling \
tools at that point and reply with a brief confirmation."""


class WikispeediaTasksetConfig(vf.TasksetConfig):
    id: str = "langchain-deep-agents-wikispeedia"
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
    max_turns: int = 50
    timeout_seconds: float = 1200.0


class WikispeediaTask(vf.Task):
    answer: str
    source: str
    target: str
    shortest_path: int
    cache_dir: str | None = None
    links_only: bool = False
    allow_go_back: bool = True


class WikispeediaTaskset(vf.Taskset[WikispeediaTasksetConfig]):
    task_type = WikispeediaTask

    def load_system_prompt(self, config: WikispeediaTasksetConfig) -> vf.SystemPrompt:
        return system_prompt(allow_go_back=config.allow_go_back)

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(self.config, split=split)

    @vf.reward(weight=1.0)
    async def reached_target(self, state: vf.State) -> float:
        return float(bool(state.extras.get("reached_target", False)))

    @vf.reward(weight=1.0)
    async def path_efficiency_reward(self, state: vf.State) -> float:
        if self.config.efficiency_weight <= 0:
            return 0.0
        return self.config.efficiency_weight * path_efficiency(state)

    @vf.metric
    async def path_efficiency(self, state: vf.State) -> float:
        return path_efficiency(state)

    @vf.metric
    async def path_length(self, state: vf.State) -> float:
        return float(max(len(path(state)) - 1, 0))

    @vf.metric
    async def shortest_path(self, state: vf.State) -> float:
        value = state.extras.get("shortest_path", 0)
        return float(value) if isinstance(value, int | float) else 0.0

    @vf.metric
    async def agent_timeout(self, state: vf.State) -> float:
        return float(bool(state.extras.get("agent_timeout", False)))

    @vf.metric
    async def total_tool_calls(self, state: vf.State) -> float:
        return float(count_tool_calls(state))

    @vf.metric
    async def assistant_turns(self, state: vf.State) -> float:
        return float(len(state.transcript))

    @vf.metric
    async def invalid_link_rate(self, state: vf.State) -> float:
        clicks = 0
        invalid = 0
        id_to_name = {
            tool_call.id: tool_call.name
            for turn in state.transcript
            for tool_call in turn.tool_calls
        }
        for turn in state.transcript:
            for result in turn.tool_results:
                if id_to_name.get(result.tool_call_id) != "click_link":
                    continue
                clicks += 1
                if (
                    isinstance(result.content, str)
                    and "is not a valid link" in result.content
                ):
                    invalid += 1
        return float(invalid / clicks) if clicks else 0.0


class WikispeediaHarness(vf.Harness[WikispeediaHarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = WikispeediaTask.model_validate(context.task.model_dump())
        state = context.state
        runtime = context.runtime
        if runtime is None:
            raise ValueError("WikispeediaHarness requires a runtime.")
        from deepagents import create_deep_agent
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.errors import GraphRecursionError

        wiki = load_wiki_graph(cache_dir(task))
        init_navigation_state(task, state)
        prompt = self.initial_messages(task)

        @tool
        async def click_link(article: str) -> str:
            """Navigate to a linked Wikipedia article."""
            return click_link_result(article, wiki, state)

        nav_tools = [click_link]
        if allow_go_back(task):

            @tool
            async def go_back() -> str:
                """Undo the last click_link and return to the previous article."""
                return go_back_result(wiki, state)

            nav_tools.append(go_back)

        async def stop_check() -> str | None:
            if await self.is_completed(context):
                return state.stop_condition or "stop"
            return None

        async with vf.InterceptionServer(
            context,
            task,
            state,
            protocols=self.protocols,
            stop_check=stop_check,
        ) as endpoint:
            endpoint_url = await runtime.expose(endpoint.port)
            endpoint_env = endpoint.env(base_url=endpoint_url, model=context.model)
            system_messages = [
                message for message in prompt if message.role == "system"
            ]
            user_messages = [message for message in prompt if message.role != "system"]
            model = ChatOpenAI(
                model=endpoint_env["OPENAI_MODEL"],
                base_url=endpoint_env["OPENAI_BASE_URL"],
                api_key=endpoint_env["OPENAI_API_KEY"],
            )
            agent = create_deep_agent(
                model=model,
                tools=nav_tools,
                system_prompt="\n\n".join(
                    str(message.content or "") for message in system_messages
                ),
            )
            invoke_config = (
                {"recursion_limit": self.config.max_turns}
                if self.config.max_turns > 0
                else None
            )
            invoke = agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "\n\n".join(
                                str(message.content or "") for message in user_messages
                            ),
                        }
                    ]
                },
                config=invoke_config,
            )
            try:
                result = await asyncio.wait_for(
                    invoke, timeout=self.config.timeout_seconds
                )
            except (TimeoutError, GraphRecursionError) as exc:
                state.extras["agent_timeout"] = True
                state.stop(
                    "agent_timeout"
                    if isinstance(exc, TimeoutError)
                    else "agent_recursion_limit"
                )
                return

        messages = result.get("messages", []) if isinstance(result, Mapping) else []
        completion = serialize_agent_completion(messages)
        if completion:
            final = completion[-1]
            content = final.content if hasattr(final, "content") else ""
            state.artifacts["agent_result"] = str(content or "")
        if not state.transcript:
            state.add_turn(vf.Turn(prompt=prompt, completion=completion))
        if not state.is_completed:
            state.stop("agent_completed")


def format_article(wiki: WikiGraph, article: str, links_only: bool = False) -> str:
    links = wiki.get_links(article)
    links_str = ", ".join(links) if links else "(no outgoing links)"
    if links_only:
        return f"# {article}\n\nAvailable links: {links_str}"
    text = wiki.get_text(article)
    return f"# {article}\n\n{text}\n\n---\nAvailable links: {links_str}"


def build_dataset(
    wiki: WikiGraph,
    pairs: list[WikiPair],
    cache_dir: str | None,
    links_only: bool,
    allow_go_back: bool,
    max_turns: int,
) -> Dataset:
    records: list[vf.JsonData] = []
    for source, target, dist in pairs:
        starting = format_article(wiki, source, links_only=links_only)
        prompt_text = (
            f"Your mission: {source} >> {target}\n\n"
            f"Here is the starting article:\n\n{starting}"
        )
        _ = wiki.get_human_stats(source, target)
        records.append(
            {
                "task_id": f"{source}->{target}",
                "prompt": [{"role": "user", "content": prompt_text}],
                "answer": target,
                "source": source,
                "target": target,
                "shortest_path": dist,
                "cache_dir": cache_dir,
                "links_only": links_only,
                "allow_go_back": allow_go_back,
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
        cache_dir=config.cache_dir,
        links_only=config.links_only,
        allow_go_back=config.allow_go_back,
        max_turns=config.max_turns,
    )


def init_navigation_state(task: WikispeediaTask, state: vf.State) -> None:
    state.extras["current_article"] = task.source
    state.extras["path"] = [task.source]
    state.extras["target"] = task.target
    state.extras["shortest_path"] = task.shortest_path
    state.extras["reached_target"] = False
    state.extras["agent_timeout"] = False
    state.extras["links_only"] = task.links_only


def cache_dir(task: WikispeediaTask) -> str | None:
    return task.cache_dir


def allow_go_back(task: WikispeediaTask) -> bool:
    return task.allow_go_back


def current_article(state: vf.State) -> str:
    value = state.extras.get("current_article")
    if not isinstance(value, str):
        raise RuntimeError("Wikispeedia current article is not initialized.")
    return value


def target_article(state: vf.State) -> str:
    value = state.extras.get("target")
    if not isinstance(value, str):
        raise RuntimeError("Wikispeedia target article is not initialized.")
    return value


def path(state: vf.State) -> list[str]:
    value = state.extras.get("path")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def set_path(state: vf.State, articles: list[str]) -> None:
    state.extras["path"] = articles


def click_link_result(article: str, wiki: WikiGraph, state: vf.State) -> str:
    links_only = bool(state.extras.get("links_only", False))
    current = current_article(state)
    available = wiki.get_links(current)
    normalized = wiki.normalize_name(article)
    if normalized is None or normalized not in available:
        avail_str = ", ".join(available) if available else "(none)"
        return (
            f"'{article}' is not a valid link from '{current}'.\n"
            f"Available links: {avail_str}"
        )
    route = path(state)
    route.append(normalized)
    set_path(state, route)
    state.extras["current_article"] = normalized
    if normalized == target_article(state):
        state.extras["reached_target"] = True
        state.stop("target_reached")
        return (
            f"TARGET REACHED: {normalized}\n\n"
            "You successfully navigated to the target. Stop calling tools and reply briefly."
        )
    return format_article(wiki, normalized, links_only=links_only)


def go_back_result(wiki: WikiGraph, state: vf.State) -> str:
    route = path(state)
    if len(route) <= 1:
        return "You are already at the starting article. Cannot go back."
    route.pop()
    set_path(state, route)
    state.extras["current_article"] = route[-1]
    return format_article(
        wiki, route[-1], links_only=bool(state.extras.get("links_only", False))
    )


def path_efficiency(state: vf.State) -> float:
    if not bool(state.extras.get("reached_target", False)):
        return 0.0
    shortest = 0.0
    raw_shortest = state.extras.get("shortest_path")
    if isinstance(raw_shortest, int | float) and not isinstance(raw_shortest, bool):
        shortest = float(raw_shortest)
    actual = max(len(path(state)) - 1, 1)
    return min(1.0, shortest / actual) if shortest > 0 else 0.0


def count_tool_calls(state: vf.State, name: str | None = None) -> int:
    names = [
        tool_call.name for turn in state.transcript for tool_call in turn.tool_calls
    ]
    if name is None:
        return len(names)
    return sum(1 for tool_name in names if tool_name == name)


def serialize_agent_completion(
    messages: Sequence[AgentMessage | vf.JsonData],
) -> vf.Messages:
    role_aliases = {
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
        "system": "system",
    }
    call_names: dict[str, str] = {}
    serialized: list[vf.JsonData] = []
    for message in messages:
        if isinstance(message, Mapping):
            payload = cast(vf.JsonData, dict(message))
        else:
            model_dump = getattr(message, "model_dump", None)
            payload = cast(
                vf.JsonData,
                model_dump(mode="json", exclude_none=True)
                if callable(model_dump)
                else {
                    "role": getattr(message, "role", None)
                    or getattr(message, "type", "assistant"),
                    "content": getattr(message, "content", str(message)),
                    "name": getattr(message, "name", None),
                    "tool_call_id": getattr(message, "tool_call_id", None),
                    "tool_calls": getattr(message, "tool_calls", None),
                },
            )
        raw_role = payload.get("role") or payload.get("type") or "assistant"
        role = role_aliases.get(str(raw_role), str(raw_role))
        item: vf.JsonData = {"role": role, "content": payload.get("content", "")}
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_tool_calls: list[vf.JsonData] = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                tool_call_payload = cast(vf.JsonData, dict(tool_call))
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
                tool_name = call_names.get(tool_call_id)
                if tool_name is not None:
                    item["name"] = tool_name
        serialized.append(item)
    if serialized and serialized[0].get("role") == "user":
        serialized = serialized[1:]
    return vf.get_messages(serialized)


def load_taskset(config: WikispeediaTasksetConfig) -> WikispeediaTaskset:
    return WikispeediaTaskset(config=config)


def load_harness(config: WikispeediaHarnessConfig) -> WikispeediaHarness:
    return WikispeediaHarness(config=config)
