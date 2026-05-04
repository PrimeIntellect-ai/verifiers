from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import verifiers as core_vf
import verifiers.v1 as vf
from verifiers.types import Tool

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgent
from tau2.agent.llm_agent import is_valid_agent_history_message
from tau2.config import DEFAULT_LLM_ARGS_AGENT, DEFAULT_LLM_ARGS_USER
from tau2.config import DEFAULT_MAX_ERRORS, DEFAULT_MAX_STEPS
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage
from tau2.data_model.message import ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task as TauTask
from tau2.environment.environment import Environment as TauEnvironment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE, Role
from tau2.registry import registry
from tau2.run import load_tasks
from tau2.user.user_simulator import UserSimulator, is_valid_user_history_message
from tau2.utils.utils import DATA_DIR, format_time, get_now

DEFAULT_USER_MODEL = "gpt-4.1"
DEFAULT_USER_BASE_URL = "https://api.openai.com/v1"
DEFAULT_USER_API_KEY_VAR = "OPENAI_API_KEY"


def download_tau2_data() -> None:
    if os.path.exists(DATA_DIR) and os.path.exists(DATA_DIR / "tau2" / "domains"):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    temp_dir = Path("/tmp/tau2_bench")
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/sierra-research/tau2-bench.git",
                temp_dir,
            ],
            check=True,
            capture_output=True,
        )
        source_data = temp_dir / "data"
        if source_data.exists():
            shutil.copytree(source_data, DATA_DIR, dirs_exist_ok=True)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def tau_msg_to_vf_dict(message: Message) -> dict[str, object]:
    if isinstance(message, AssistantMessage):
        if message.tool_calls:
            return core_vf.AssistantMessage(
                content=message.content,
                tool_calls=[
                    core_vf.ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=json.dumps(tool_call.arguments),
                    )
                    for tool_call in message.tool_calls
                ],
            ).model_dump(exclude_none=True)
        return core_vf.AssistantMessage(content=message.content).model_dump(
            exclude_none=True
        )
    if isinstance(message, UserMessage):
        return core_vf.UserMessage(content=message.content or "").model_dump(
            exclude_none=True
        )
    if isinstance(message, ToolMessage):
        return core_vf.ToolMessage(
            tool_call_id=message.id,
            content=message.content or "",
        ).model_dump(exclude_none=True)
    raise ValueError(f"Unknown tau2 message type: {type(message)}")


def dump_tau_message(message: Message) -> dict[str, object]:
    return cast(dict[str, object], message.model_dump(mode="json", exclude_none=True))


def load_tau_message(payload: Mapping[str, object]) -> Message:
    role = payload.get("role")
    if role == "assistant":
        return AssistantMessage.model_validate(payload)
    if role == "user":
        return UserMessage.model_validate(payload)
    if role == "tool":
        if "tool_messages" in payload:
            return MultiToolMessage.model_validate(payload)
        return ToolMessage.model_validate(payload)
    raise ValueError(f"Unknown tau2 message role: {role!r}")


class Tau2Session:
    def __init__(
        self,
        domain: str,
        task_payload: Mapping[str, object],
        user_model: str,
        user_args: Mapping[str, object],
        max_steps: int,
        max_errors: int,
    ):
        self.domain = domain
        self.task_payload = dict(task_payload)
        self.user_model = user_model
        self.user_args = dict(user_args)
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.ready = False
        self.initial_prompt_messages: list[dict[str, object]] = []
        self.recorded_assistant_messages = 0
        self.pending_agent_tool_calls: list[ToolCall] = []
        self.num_assistant_tool_calls = 0
        self.num_user_tool_calls = 0

    async def initialize(self, state: vf.State) -> None:
        if self.ready:
            return
        self.task = TauTask.model_validate(self.task_payload)
        environment_constructor = registry.get_env_constructor(self.domain)
        self.environment = await asyncio.to_thread(environment_constructor)
        self.agent = LLMAgent(
            tools=self.environment.get_tools(),
            domain_policy=self.environment.get_policy(),
            llm=str(state.get("runtime", {}).get("model") or ""),
            llm_args=dict(state.get("runtime", {}).get("sampling_args") or {}),
        )
        self.user = UserSimulator(
            tools=self.user_tools(),
            instructions=str(self.task.user_scenario),
            llm=self.user_model,
            llm_args=self.user_args,
        )
        self.init_tau2_state()
        self.environment.sync_tools()
        self.ready = True
        self.initial_prompt_messages = await self.advance_until_agent(state)
        self.render_state(state)

    def user_tools(self) -> object:
        try:
            return self.environment.get_user_tools()
        except Exception:
            return None

    def init_tau2_state(self) -> None:
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        for message in message_history:
            message.turn_idx = None
        message_history = add_timestamps(message_history)
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        self.done = False
        self.termination_reason: TerminationReason | None = None
        if message_history:
            self.init_from_history(message_history)
        else:
            self.user_state = self.user.get_init_state()
            first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
            first_message.timestamp = get_now()
            self.agent_state = self.agent.get_init_state(
                message_history=[first_message]
            )
            self.trajectory: list[Message] = [first_message]
            self.message = first_message
            self.from_role = Role.AGENT
            self.to_role = Role.USER
        self.step_count = 0
        self.num_errors = 0

    def init_from_history(self, message_history: list[Message]) -> None:
        last_message = message_history[-1]
        self.trajectory = message_history
        self.message = last_message
        if isinstance(last_message, AssistantMessage):
            self.from_role = Role.AGENT
            self.to_role = Role.ENV if last_message.is_tool_call() else Role.USER
            self.agent_state = self.agent.get_init_state(
                message_history=[
                    msg
                    for msg in message_history
                    if is_valid_agent_history_message(msg)
                ]
            )
            self.user_state = self.user.get_init_state(
                message_history=[
                    msg
                    for msg in message_history[:-1]
                    if is_valid_user_history_message(msg)
                ]
            )
            if self.agent.is_stop(last_message):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            return
        if isinstance(last_message, UserMessage):
            self.from_role = Role.USER
            self.to_role = Role.ENV if last_message.is_tool_call() else Role.AGENT
            self.user_state = self.user.get_init_state(
                message_history=[
                    msg for msg in message_history if is_valid_user_history_message(msg)
                ]
            )
            self.agent_state = self.agent.get_init_state(
                message_history=[
                    msg
                    for msg in message_history[:-1]
                    if is_valid_agent_history_message(msg)
                ]
            )
            self.done = UserSimulator.is_stop(last_message)
            if self.done:
                self.termination_reason = TerminationReason.USER_STOP
            return
        if isinstance(last_message, ToolMessage):
            self.from_role = Role.ENV
            self.to_role = (
                Role.AGENT if last_message.requestor == "assistant" else Role.USER
            )
            self.agent_state = self.agent.get_init_state(
                message_history=[
                    msg
                    for msg in message_history
                    if is_valid_agent_history_message(msg)
                ]
            )
            self.user_state = self.user.get_init_state(
                message_history=[
                    msg for msg in message_history if is_valid_user_history_message(msg)
                ]
            )
            return
        raise ValueError(f"Unsupported tau2 message type: {type(last_message)}")

    async def record_assistant_from_state(self, state: vf.State) -> None:
        completion = state.get("completion") or []
        assistant_messages = [
            message
            for message in completion
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        ]
        for message in assistant_messages[self.recorded_assistant_messages :]:
            assistant_message = assistant_from_openai_message(message)
            self.agent_state.messages.append(assistant_message)
            try:
                assistant_message.validate()
            except ValueError:
                self.done = True
                self.termination_reason = TerminationReason.AGENT_ERROR
                self.trajectory.append(assistant_message)
                continue
            if self.agent.is_stop(assistant_message):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            self.trajectory.append(assistant_message)
            self.pending_agent_tool_calls.extend(assistant_message.tool_calls or [])
            self.num_assistant_tool_calls += len(assistant_message.tool_calls or [])
            self.message = assistant_message
            self.from_role = Role.AGENT
            self.to_role = Role.ENV if assistant_message.tool_calls else Role.USER
            self.step_count += 1
            self.environment.sync_tools()
            self.check_limits()
        self.recorded_assistant_messages = len(assistant_messages)
        self.render_state(state)

    async def call_agent_tool(
        self, name: str, arguments: Mapping[str, object], state: vf.State
    ) -> str:
        await self.record_assistant_from_state(state)
        tool_call = self.pop_pending_tool_call(name, arguments)
        tool_message = await asyncio.to_thread(self.environment.get_response, tool_call)
        if tool_message.error:
            self.num_errors += 1
        self.trajectory.append(tool_message)
        self.message = tool_message
        self.from_role = Role.ENV
        self.to_role = Role.AGENT
        self.environment.sync_tools()
        self.check_limits()
        self.render_state(state)
        return tool_message.content or ""

    def pop_pending_tool_call(
        self, name: str, arguments: Mapping[str, object]
    ) -> ToolCall:
        for index, tool_call in enumerate(self.pending_agent_tool_calls):
            if tool_call.name == name:
                return self.pending_agent_tool_calls.pop(index)
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=dict(arguments),
            requestor="assistant",
        )

    async def user_messages(self, state: vf.State) -> list[dict[str, object]]:
        await self.record_assistant_from_state(state)
        if self.done:
            self.render_state(state)
            return []
        messages = await self.advance_until_agent(state)
        self.render_state(state)
        return messages

    async def advance_until_agent(self, state: vf.State) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        while not (self.done or self.to_role == Role.AGENT):
            if self.to_role == Role.USER:
                user_message, self.user_state = await asyncio.to_thread(
                    self.user.generate_next_message,
                    self.message,
                    self.user_state,
                )
                try:
                    user_message.validate()
                except ValueError:
                    self.done = True
                    self.termination_reason = TerminationReason.USER_ERROR
                    self.trajectory.append(user_message)
                    break
                if UserSimulator.is_stop(user_message):
                    self.done = True
                    self.termination_reason = TerminationReason.USER_STOP
                self.num_user_tool_calls += len(user_message.tool_calls or [])
                self.trajectory.append(user_message)
                self.message = user_message
                self.from_role = Role.USER
                self.to_role = Role.ENV if user_message.is_tool_call() else Role.AGENT
                self.step_count += 1
                self.check_limits()
                if not user_message.is_tool_call():
                    messages.append(tau_msg_to_vf_dict(user_message))
                    break
                continue
            if self.to_role == Role.ENV:
                await self.execute_user_tools()
                self.check_limits()
                continue
            raise ValueError(
                f"Invalid tau2 role transition: {self.from_role} -> {self.to_role}"
            )
        return messages

    async def execute_user_tools(self) -> None:
        tool_calls = list(getattr(self.message, "tool_calls", []) or [])
        tool_messages = []
        for tool_call in tool_calls:
            tool_call.requestor = "user"
            tool_message = await asyncio.to_thread(
                self.environment.get_response, tool_call
            )
            if tool_message.error:
                self.num_errors += 1
            tool_messages.append(tool_message)
        self.trajectory.extend(tool_messages)
        if len(tool_messages) > 1:
            self.message = MultiToolMessage(
                role="tool",
                tool_messages=tool_messages,
            )
        elif tool_messages:
            self.message = tool_messages[0]
        self.from_role = Role.ENV
        self.to_role = Role.USER
        self.step_count += 1
        self.environment.sync_tools()

    def check_limits(self) -> None:
        if self.step_count >= self.max_steps and self.to_role != Role.ENV:
            self.done = True
            self.termination_reason = TerminationReason.MAX_STEPS
        if self.num_errors >= self.max_errors:
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS

    def render_state(self, state: vf.State) -> None:
        state["tau2"] = {
            "task_id": self.task.id,
            "done": self.done,
            "termination_reason": (
                self.termination_reason.value if self.termination_reason else None
            ),
            "step_count": self.step_count,
            "num_errors": self.num_errors,
            "num_assistant_tool_calls": self.num_assistant_tool_calls,
            "num_user_tool_calls": self.num_user_tool_calls,
            "messages": [dump_tau_message(message) for message in self.trajectory],
        }
        state["num_assistant_tool_calls"] = self.num_assistant_tool_calls
        state["num_user_tool_calls"] = self.num_user_tool_calls
        if self.done and self.termination_reason is not None:
            state["stop_condition"] = self.termination_reason.value


class Tau2Harness(vf.Harness):
    async def setup_state(self, task: vf.Task, state: vf.State) -> vf.State:
        state = await super().setup_state(task, state)
        sampling_args = dict(DEFAULT_LLM_ARGS_AGENT)
        sampling_args.update(dict(state.get("runtime", {}).get("sampling_args") or {}))
        state["runtime"]["sampling_args"] = sampling_args
        session = cast(
            Tau2Session, await self.runtime.resolve_object("tau2", task, state)
        )
        await session.initialize(state)
        state.setdefault("prompt", [])
        if not state.get("tau2_prompt_initialized"):
            state["prompt"].extend(session.initial_prompt_messages)
            state["tau2_prompt_initialized"] = True
        return state


def assistant_from_openai_message(message: Mapping[str, object]) -> AssistantMessage:
    tool_calls = []
    for raw_tool_call in cast(list[object], message.get("tool_calls") or []):
        if isinstance(raw_tool_call, str):
            raw_tool_call = json.loads(raw_tool_call)
        if not isinstance(raw_tool_call, Mapping):
            raise TypeError("tau2 tool calls must be mappings.")
        function = raw_tool_call.get("function") or {}
        if function is not None and not isinstance(function, Mapping):
            raise TypeError("tau2 tool call function payload must be a mapping.")
        function = cast(Mapping[str, object], function)
        arguments = function.get("arguments", raw_tool_call.get("arguments", "{}"))
        if isinstance(arguments, str):
            parsed_arguments = json.loads(arguments or "{}")
        else:
            parsed_arguments = arguments
        name = function.get("name") or raw_tool_call.get("name") or ""
        tool_calls.append(
            ToolCall(
                id=str(raw_tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                name=str(name),
                arguments=cast(dict[str, object], parsed_arguments),
                requestor="assistant",
            )
        )
    content = message.get("content")
    return AssistantMessage(
        role="assistant",
        content=content if isinstance(content, str) and content else None,
        tool_calls=tool_calls or None,
        raw_data=dict(message),
    )


def add_timestamps(message_history: list[Message]) -> list[Message]:
    time_offset = datetime.now() - timedelta(seconds=len(message_history))
    for index, message in enumerate(message_history):
        message.timestamp = format_time(time_offset + timedelta(seconds=index))
    return message_history


def source(domain: str):
    download_tau2_data()
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    system_prompt = SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=environment.policy,
    )
    for index, task in enumerate(
        load_tasks(task_set_name=domain, task_split_name="base")
    ):
        yield {
            "example_id": index,
            "taskset_id": f"tau2_{domain}",
            "task_id": task.id,
            "domain": domain,
            "prompt": [{"role": "system", "content": system_prompt}],
            "info": task.model_dump_json(exclude_none=True),
        }


def load_session_factory(
    domain: str,
    user_model: str,
    user_args: Mapping[str, object],
    max_steps: int,
    max_errors: int,
) -> Callable[..., Tau2Session]:
    def load_session(task, state) -> Tau2Session:
        _ = state
        task_info = task["info"]
        if isinstance(task_info, str):
            task_info = json.loads(task_info)
        return Tau2Session(
            domain=domain,
            task_payload=cast(Mapping[str, object], task_info),
            user_model=user_model,
            user_args=user_args,
            max_steps=max_steps,
            max_errors=max_errors,
        )

    return load_session


def make_tau2_tool(name: str, schema: Mapping[str, object]) -> Callable[..., object]:
    async def tool(session: Tau2Session, state, **arguments) -> str:
        return await session.call_agent_tool(name, arguments, state)

    function_schema = cast(Mapping[str, object], schema["function"])
    tool.__name__ = name
    tool.__doc__ = str(function_schema.get("description") or "")
    tool.tool_def = Tool(
        name=name,
        description=str(function_schema.get("description") or ""),
        parameters=cast(dict[str, Any], function_schema.get("parameters") or {}),
        strict=False,
    )
    return tool


def load_toolset(
    domain: str = "telecom",
    user_model: str = DEFAULT_USER_MODEL,
    user_args: Mapping[str, object] = DEFAULT_LLM_ARGS_USER,
    user_base_url: str = DEFAULT_USER_BASE_URL,
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
) -> vf.Toolset:
    download_tau2_data()
    environment_constructor = registry.get_env_constructor(domain)
    environment = cast(TauEnvironment, environment_constructor())
    schemas = [tool.openai_schema for tool in environment.get_tools()]
    tools = [
        make_tau2_tool(
            str(cast(Mapping[str, object], schema["function"])["name"]),
            schema,
        )
        for schema in schemas
    ]
    resolved_user_args = {
        **dict(user_args),
        "api_base": user_base_url,
        "api_key": os.getenv(user_api_key_var),
    }
    return vf.Toolset(
        tools=tools,
        bindings={f"{tool.__name__}.session": "objects.tau2" for tool in tools},
        objects={
            "tau2": load_session_factory(
                domain=domain,
                user_model=user_model,
                user_args=resolved_user_args,
                max_steps=max_steps,
                max_errors=max_errors,
            )
        },
        write=True,
        scope="rollout",
    )


async def tau2_user(session: Tau2Session, state, transcript) -> list[dict[str, object]]:
    _ = transcript
    return await session.user_messages(state)


@vf.reward(weight=1.0)
async def tau2_reward(task, state) -> float:
    tau2_state = cast(Mapping[str, object], state["tau2"])
    messages = [
        load_tau_message(cast(Mapping[str, object], message))
        for message in cast(list[object], tau2_state["messages"])
    ]
    termination = tau2_state.get("termination_reason")
    if isinstance(termination, str):
        termination_reason = TerminationReason(termination)
    elif state.get("stop_condition") == "max_turns_reached":
        termination_reason = TerminationReason.MAX_STEPS
    else:
        termination_reason = TerminationReason.AGENT_ERROR
    state["tau2"]["termination_reason"] = termination_reason.value
    task_info = task["info"]
    if isinstance(task_info, str):
        task_info = json.loads(task_info)
    tau_task = TauTask.model_validate(task_info)
    simulation = SimulationRun(
        id=f"{task['taskset_id']}_{task['task_id']}_{datetime.now().isoformat()}",
        task_id=tau_task.id,
        messages=messages,
        termination_reason=termination_reason,
        timestamp=datetime.now().isoformat(),
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat(),
        duration=0.0,
        agent_cost=0.0,
        user_cost=0.0,
    )
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=tau_task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=str(task["domain"]),
    )
    state["tau2"]["evaluation"] = reward_info.model_dump(mode="json")
    return float(reward_info.reward)


@vf.metric
async def tau2_num_errors(task, state) -> float:
    _ = task
    return float(state.get("tau2", {}).get("num_errors", 0.0))


@vf.metric
async def tau2_num_steps(task, state) -> float:
    _ = task
    return float(state.get("tau2", {}).get("step_count", 0.0))


@vf.metric
async def tau2_num_assistant_tool_calls(task, state) -> float:
    _ = task
    return float(state.get("num_assistant_tool_calls", 0.0))


@vf.metric
async def tau2_num_user_tool_calls(task, state) -> float:
    _ = task
    return float(state.get("num_user_tool_calls", 0.0))


def load_taskset(
    domain: str = "telecom",
    user_model: str = DEFAULT_USER_MODEL,
    user_args: Mapping[str, object] = DEFAULT_LLM_ARGS_USER,
    user_base_url: str = DEFAULT_USER_BASE_URL,
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    config=None,
) -> vf.Taskset:
    toolset = load_toolset(
        domain=domain,
        user_model=user_model,
        user_args=user_args,
        user_base_url=user_base_url,
        user_api_key_var=user_api_key_var,
        max_steps=max_steps,
        max_errors=max_errors,
    )
    return vf.Taskset(
        source=lambda: source(domain),
        taskset_id=f"tau2_{domain}",
        rewards=[tau2_reward],
        metrics=[
            tau2_num_errors,
            tau2_num_steps,
            tau2_num_assistant_tool_calls,
            tau2_num_user_tool_calls,
        ],
        toolsets=[toolset],
        user=vf.User(tau2_user, bindings={"session": "objects.tau2"}),
        config=config,
    )


def load_harness(max_turns: int = DEFAULT_MAX_STEPS, config=None) -> Tau2Harness:
    return Tau2Harness(max_turns=max_turns, config=config)


def load_v1_environment(
    domain: str = "telecom",
    user_model: str = DEFAULT_USER_MODEL,
    user_args: Mapping[str, object] = DEFAULT_LLM_ARGS_USER,
    user_base_url: str = DEFAULT_USER_BASE_URL,
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_errors: int = DEFAULT_MAX_ERRORS,
    max_turns: int = DEFAULT_MAX_STEPS,
    **kwargs,
) -> vf.Env:
    if kwargs:
        raise TypeError(f"Unsupported v1 args: {sorted(kwargs)}")
    taskset = load_taskset(
        domain=domain,
        user_model=user_model,
        user_args=user_args,
        user_base_url=user_base_url,
        user_api_key_var=user_api_key_var,
        max_steps=max_steps,
        max_errors=max_errors,
    )
    harness = load_harness(max_turns=max_turns)
    return vf.Env(taskset=taskset, harness=harness)


def load_environment(**kwargs) -> vf.Env:
    return load_v1_environment(**kwargs)
