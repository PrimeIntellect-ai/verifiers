import asyncio
import uuid
from typing import Any, cast

import verifiers as vf

from .utils.deep_agents_utils import (
    DEEP_AGENTS_SCAFFOLDING_PROMPT,
    deep_agent_system_prompt,
    langchain_tools_from_state,
    serialize_agent_completion,
)


class DeepAgentsConfig(vf.HarnessConfig):
    system_prompt: vf.SystemPrompt = DEEP_AGENTS_SCAFFOLDING_PROMPT
    program: vf.ProgramConfig = vf.ProgramConfig(
        fn="harnesses.deep_agents:run_deep_agent"
    )
    max_turns: int = 50
    timeout_seconds: float = 1200.0
    agent_name: str = "deep-agent"


def deep_agent_invoke_config(
    task: vf.Task, state: vf.State, recursion_limit: int
) -> dict[str, object]:
    runtime = state.get("runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    trajectory_id = str(state["trajectory_id"])
    run_id = uuid.UUID(hex=trajectory_id)
    state["langsmith_run_id"] = str(run_id)
    env_id = str(task.get("taskset_id") or "")
    task_id = str(task.get("task_id", ""))
    run_name = f"{env_id}:{task_id}" if env_id and task_id else env_id or task_id
    tags = ["verifiers", "vf-v1"]
    if env_id:
        tags.append(env_id)
    invoke_config: dict[str, object] = {
        "run_name": run_name,
        "run_id": run_id,
        "configurable": {"thread_id": trajectory_id},
        "metadata": {
            "vf_env": env_id,
            "vf_task_id": task_id,
            "vf_trajectory_id": trajectory_id,
            "vf_group_key": str(runtime.get("group_key", "")),
        },
        "tags": tags,
    }
    if recursion_limit > 0:
        invoke_config["recursion_limit"] = recursion_limit
    return invoke_config


async def run_deep_agent(
    task: vf.Task, state: vf.State, harness: "DeepAgents"
) -> vf.State:
    try:
        from deepagents import create_deep_agent  # ty: ignore[unresolved-import]
        from langchain_openai import ChatOpenAI  # ty: ignore[unresolved-import]
        from langgraph.errors import (  # ty: ignore[unresolved-import]
            GraphRecursionError,
        )
    except ModuleNotFoundError as exc:
        raise ImportError(
            "DeepAgents harness requires deepagents. "
            "Install as `verifiers[deepagents]`."
        ) from exc
    from openai import OpenAI

    config = harness.config
    endpoint_config = state.get_endpoint_config(api="chat")
    endpoint_client = cast(OpenAI, state.get_client(api="chat", sync=True))
    endpoint_api_key = endpoint_client.api_key
    endpoint_client.close()
    model = ChatOpenAI(
        model=endpoint_config.model,
        base_url=endpoint_config.base_url,
        api_key=endpoint_api_key,
    )
    tools = langchain_tools_from_state(state, harness.runtime)
    agent: Any = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=deep_agent_system_prompt(state),
        name=config.agent_name,
    )
    prompt = str(cast(list[vf.JsonData], state["prompt"])[-1]["content"])
    recursion_limit = state.get_max_turns(config.max_turns)
    invoke = agent.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=deep_agent_invoke_config(task, state, recursion_limit),
    )
    try:
        result = await asyncio.wait_for(invoke, timeout=config.timeout_seconds)
    except (TimeoutError, GraphRecursionError) as exc:
        reason = (
            "agent_timeout"
            if isinstance(exc, TimeoutError)
            else "agent_recursion_limit"
        )
        state[reason] = True
        state.stop(reason)
        state.setdefault("agent_completion", [])
        return state

    messages = result.get("messages", []) if isinstance(result, dict) else []
    completion = serialize_agent_completion(messages)
    state["agent_completion"] = completion
    state["completion"] = completion
    if completion:
        state["agent_result"] = str(completion[-1].get("content") or "")
    return state


class DeepAgents(vf.Harness[DeepAgentsConfig]):
    config: DeepAgentsConfig

    @vf.update(priority=-200)
    async def restore_agent_completion(self, state: vf.State) -> None:
        agent_completion = state.get("agent_completion")
        if isinstance(agent_completion, list):
            state["completion"] = agent_completion

    @vf.metric
    async def agent_timeout(self, state: vf.State) -> float:
        return 1.0 if state.get("agent_timeout", False) else 0.0

    @vf.metric
    async def agent_recursion_limit(self, state: vf.State) -> float:
        return 1.0 if state.get("agent_recursion_limit", False) else 0.0


def load_harness(config: DeepAgentsConfig) -> DeepAgents:
    return DeepAgents(config=config)
