import asyncio
import json
import os
import shutil
import subprocess
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import verifiers as core_vf
import verifiers as vf
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
from tau2.run import load_tasks as load_tau2_tasks
from tau2.user.user_simulator import UserSimulator, is_valid_user_history_message
from tau2.utils.utils import DATA_DIR, format_time, get_now

DEFAULT_USER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_USER_BASE_URL = "https://api.pinference.ai/api/v1"
DEFAULT_USER_API_KEY_VAR = "PRIME_API_KEY"


def download_tau2_data() -> None:
    if os.path.exists(DATA_DIR) and os.path.exists(DATA_DIR / "tau2" / "domains"):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    temp_dir = Path("/tmp/tau2_bench_v1")
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


def tau_msg_to_vf_dict(message: Message) -> vf.ConfigData:
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


def dump_tau_message(message: Message) -> vf.ConfigData:
    return cast(vf.ConfigData, message.model_dump(mode="json", exclude_none=True))


def load_tau_message(payload: vf.JsonData) -> Message:
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
        task_payload: vf.JsonData,
        user_model: str,
        user_args: vf.JsonData,
        user_base_url: str,
        user_api_key_var: str,
        max_steps: int,
        max_errors: int,
    ):
        self.domain = domain
        self.task_payload = dict(task_payload)
        self.user_model = user_model
        self.user_args = dict(user_args)
        self.user_base_url = user_base_url
        self.user_api_key_var = user_api_key_var
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.ready = False
        self.initial_prompt_messages: list[vf.ConfigData] = []
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
        user_args = dict(self.user_args)
        if self.user_base_url == DEFAULT_USER_BASE_URL:
            custom_provider = user_args.get("custom_llm_provider")
            assert custom_provider in (None, "custom_openai")
            user_args["custom_llm_provider"] = "custom_openai"
        if self.user_api_key_var == "PRIME_API_KEY":
            team_id = os.getenv("PRIME_TEAM_ID")
            if team_id:
                extra_headers = user_args.get("extra_headers") or {}
                assert isinstance(extra_headers, dict)
                user_args["extra_headers"] = {
                    **extra_headers,
                    "X-Prime-Team-ID": team_id,
                }
        user_args["api_base"] = self.user_base_url
        user_args["api_key"] = os.getenv(self.user_api_key_var)
        self.user = UserSimulator(
            tools=self.user_tools(),
            instructions=str(self.task.user_scenario),
            llm=self.user_model,
            llm_args=user_args,
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
        assistant_messages = (
            vf.get_messages(completion, role="assistant")
            if isinstance(completion, list)
            else []
        )
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
        self, name: str, arguments: vf.JsonData, state: vf.State
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

    def pop_pending_tool_call(self, name: str, arguments: vf.JsonData) -> ToolCall:
        for index, tool_call in enumerate(self.pending_agent_tool_calls):
            if tool_call.name == name:
                return self.pending_agent_tool_calls.pop(index)
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=dict(arguments),
            requestor="assistant",
        )

    async def user_messages(self, state: vf.State) -> list[vf.ConfigData]:
        await self.record_assistant_from_state(state)
        if self.done:
            self.render_state(state)
            return []
        messages = await self.advance_until_agent(state)
        self.render_state(state)
        return messages

    async def advance_until_agent(self, state: vf.State) -> list[vf.ConfigData]:
        messages: list[vf.ConfigData] = []
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
            state.stop(self.termination_reason.value)


def assistant_from_openai_message(message: vf.AssistantMessage) -> AssistantMessage:
    tool_calls = []
    for raw_tool_call in message.tool_calls or []:
        arguments = raw_tool_call.arguments
        if isinstance(arguments, str):
            parsed_arguments = json.loads(arguments or "{}")
        else:
            parsed_arguments = arguments
        tool_calls.append(
            ToolCall(
                id=raw_tool_call.id or f"call_{uuid.uuid4().hex[:8]}",
                name=raw_tool_call.name,
                arguments=cast(vf.ConfigData, parsed_arguments),
                requestor="assistant",
            )
        )
    content = message.content
    return AssistantMessage(
        role="assistant",
        content=content if isinstance(content, str) and content else None,
        tool_calls=tool_calls or None,
        raw_data=message.model_dump(exclude_none=True),
    )


def add_timestamps(message_history: list[Message]) -> list[Message]:
    time_offset = datetime.now() - timedelta(seconds=len(message_history))
    for index, message in enumerate(message_history):
        message.timestamp = format_time(time_offset + timedelta(seconds=index))
    return message_history


def load_tasks(domain: str, max_turns: int):
    download_tau2_data()
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    system_prompt = SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=environment.policy,
    )
    for index, task in enumerate(
        load_tau2_tasks(task_set_name=domain, task_split_name="base")
    ):
        yield {
            "example_id": index,
            "taskset_id": f"tau2_{domain}",
            "task_id": task.id,
            "domain": domain,
            "system_prompt": system_prompt,
            "max_turns": max_turns,
            "prompt": [],
            "info": task.model_dump_json(exclude_none=True),
        }


def make_tau2_tool(name: str, schema: vf.JsonData) -> vf.Handler:
    async def tool(task, state, **arguments) -> str:
        _ = task
        session = cast(
            Tau2Session,
            state["tau2_session"],
        )
        return await session.call_agent_tool(name, arguments, state)

    function_schema = cast(vf.JsonData, schema["function"])
    tool.__name__ = name
    tool.__doc__ = str(function_schema.get("description") or "")
    tool.tool_def = Tool(
        name=name,
        description=str(function_schema.get("description") or ""),
        parameters=cast(vf.ConfigData, function_schema.get("parameters") or {}),
        strict=False,
    )
    return tool


def load_toolset(
    domain: str = "telecom",
) -> vf.Toolset:
    download_tau2_data()
    environment_constructor = registry.get_env_constructor(domain)
    environment = cast(TauEnvironment, environment_constructor())
    schemas = [tool.openai_schema for tool in environment.get_tools()]
    tools = [
        make_tau2_tool(
            str(cast(vf.JsonData, schema["function"])["name"]),
            schema,
        )
        for schema in schemas
    ]
    return vf.Toolset(
        tools=tools,
        write=True,
        scope="rollout",
    )


class Tau2UserConfig(vf.UserConfig):
    pass


class Tau2User(vf.User[Tau2UserConfig]):
    async def get_response(self, task, state) -> list[vf.ConfigData]:
        _ = task
        session = cast(
            Tau2Session,
            state["tau2_session"],
        )
        return await session.user_messages(state)


class Tau2TasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "tau2_telecom"
    user: Tau2UserConfig | None = Tau2UserConfig()
    domain: str = "telecom"
    user_model: str = DEFAULT_USER_MODEL
    user_args: vf.ConfigData | None = None
    user_base_url: str = DEFAULT_USER_BASE_URL
    user_api_key_var: str = DEFAULT_USER_API_KEY_VAR
    max_steps: int = DEFAULT_MAX_STEPS
    max_errors: int = DEFAULT_MAX_ERRORS
    max_turns: int = DEFAULT_MAX_STEPS


class Tau2Taskset(vf.Taskset[Tau2TasksetConfig]):
    def load_toolsets(self, config: Tau2TasksetConfig) -> vf.Toolsets:
        if "toolsets" in self.config.model_fields_set:
            return None
        return load_toolset(domain=config.domain)

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(domain=self.config.domain, max_turns=self.config.max_turns)

    @vf.setup(priority=100)
    async def tau2_setup(self, task: vf.Task, state: vf.State) -> None:
        if "setups" in self.config.model_fields_set:
            return
        runtime = state.runtime_state()
        sampling_args = dict(DEFAULT_LLM_ARGS_AGENT)
        sampling_args.update(dict(runtime.get("sampling_args") or {}))
        runtime["sampling_args"] = sampling_args
        task_info = task["info"]
        if isinstance(task_info, str):
            task_info = json.loads(task_info)
        user_args = (
            DEFAULT_LLM_ARGS_USER
            if self.config.user_args is None
            else self.config.user_args
        )
        session = Tau2Session(
            domain=self.config.domain,
            task_payload=cast(vf.JsonData, task_info),
            user_model=self.config.user_model,
            user_args=cast(vf.JsonData, user_args),
            user_base_url=self.config.user_base_url,
            user_api_key_var=self.config.user_api_key_var,
            max_steps=self.config.max_steps,
            max_errors=self.config.max_errors,
        )
        state["tau2_session"] = session
        await session.initialize(state)
        state.setdefault("prompt", [])
        if not state.get("tau2_prompt_initialized"):
            state["prompt"].extend(session.initial_prompt_messages)
            state["tau2_prompt_initialized"] = True

    @vf.cleanup
    async def tau2_cleanup(self, state: vf.State) -> None:
        state.pop("tau2_session", None)

    @vf.reward(weight=1.0)
    async def tau2_reward(self, task: vf.Task, state: vf.State) -> float:
        tau2_state = cast(vf.JsonData, state["tau2"])
        messages = [
            load_tau_message(cast(vf.JsonData, message))
            for message in cast(list[vf.JsonData], tau2_state["messages"])
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
    async def tau2_num_errors(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("tau2", {}).get("num_errors", 0.0))

    @vf.metric
    async def tau2_num_steps(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("tau2", {}).get("step_count", 0.0))

    @vf.metric
    async def tau2_num_assistant_tool_calls(
        self, task: vf.Task, state: vf.State
    ) -> float:
        _ = task
        return float(state.get("num_assistant_tool_calls", 0.0))

    @vf.metric
    async def tau2_num_user_tool_calls(self, task: vf.Task, state: vf.State) -> float:
        _ = task
        return float(state.get("num_user_tool_calls", 0.0))


class Tau2EnvConfig(vf.EnvConfig):
    taskset: Tau2TasksetConfig = Tau2TasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_environment(config: Tau2EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=Tau2Taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
