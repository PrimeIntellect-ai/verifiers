from __future__ import annotations

import os
from copy import deepcopy
from typing import Protocol

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task as TauTask
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.registry import registry
from tau2.user.base import UserState, is_valid_user_history_message
from tau2.user.user_simulator import UserSimulator
from tau2.utils.utils import get_now

import verifiers.v1 as vf

from .config import UserConfig

UserInputMessage = AssistantMessage | ToolMessage | MultiToolMessage


class TauUserTool(Protocol):
    pass


class User(vf.User[UserConfig]):
    environment: Environment | None
    messages: list[vf.JsonData]
    user_simulator: UserSimulator | None
    user_state: UserState | None
    pending_user_input: UserInputMessage | None
    bootstrap_user_message: UserMessage | None
    num_errors: int
    task_id: str
    max_errors: int

    def start(self) -> None:
        self.environment = None
        self.messages = []
        self.user_simulator = None
        self.user_state = None
        self.pending_user_input = None
        self.bootstrap_user_message = None
        self.num_errors = 0
        self.task_id = ""
        self.max_errors = 0

    @vf.tool(
        hidden=True,
        args={
            "domain": "task.domain",
            "tau2_task": "task.tau2_task",
            "tau2_user": "task.tau2_user",
        },
        sets={
            "tau2": "state.extras.tau2",
        },
    )
    def setup(self, domain: str, tau2_task: dict, tau2_user: dict) -> dict:
        tau_task = TauTask.model_validate(tau2_task)
        environment = registry.get_env_constructor(domain)()
        self.environment = environment
        self.task_id = tau_task.id
        self.max_errors = int(tau2_user.get("max_errors") or 0)
        initial_messages = self.initialize_environment(environment, tau_task)
        user_model, user_args = self.user_model_config(tau2_user)
        self.user_simulator = UserSimulator(
            tools=self.user_tools(environment),
            instructions=tau_task.user_scenario,
            llm=user_model,
            llm_args=user_args,
        )
        self.user_state = self.user_simulator.get_init_state(
            message_history=[
                message
                for message in initial_messages
                if is_valid_user_history_message(message)
            ]
        )
        self.messages = [self.message_data(message) for message in initial_messages]
        self.bootstrap_user_message = self.initial_user_message(initial_messages)
        self.pending_user_input = self.initial_user_input(initial_messages)
        return {
            "tau2": self.tau2_state(),
            "tools": self.tau2_tool_defs(environment),
        }

    @vf.user(
        args={
            "tau2_task": "task.tau2_task",
            "completion": "state.completion",
        },
        sets={
            "tau2": "state.extras.tau2",
            "stop_condition": "state.stop_condition",
        },
    )
    def respond(self, tau2_task: dict, completion: list[dict]) -> dict:
        _ = TauTask.model_validate(tau2_task)
        if self.user_simulator is None or self.user_state is None:
            raise RuntimeError("Tau2 user simulator has not started.")
        if not completion:
            if self.bootstrap_user_message is not None:
                user_message = self.bootstrap_user_message
                self.bootstrap_user_message = None
                return {
                    "messages": [self.v1_user_message(user_message)],
                    "tau2": self.tau2_state(),
                }
            if self.pending_user_input is None:
                return {"messages": [], "tau2": self.tau2_state()}
            return self.generate_user_response(self.pending_user_input)
        text = self.latest_assistant_text(completion)
        if text:
            assistant_message = AssistantMessage(role="assistant", content=text)
            return self.generate_user_response(assistant_message)
        return {"messages": [], "tau2": self.tau2_state()}

    @vf.tool(
        hidden=True,
        sets={
            "tau2": "state.extras.tau2",
            "finished": "state.is_completed",
            "stop_condition": "state.stop_condition",
        },
    )
    def call_tool(self, name: str, input: vf.JsonData) -> dict:
        if self.environment is None:
            raise RuntimeError("Tau2 environment has not started.")
        tool_call = ToolCall(
            id=name,
            name=name,
            arguments=input,
            requestor="assistant",
        )
        assistant_message = AssistantMessage(role="assistant", tool_calls=[tool_call])
        tool_message = self.environment.get_response(tool_call)
        self.record_assistant_tool_exchange(assistant_message, [tool_message])
        payload: dict[str, object] = {
            "content": tool_message.content or "",
            "tau2": self.tau2_state(),
        }
        if self.too_many_errors():
            payload["finished"] = True
            payload["stop_condition"] = "tau2_too_many_errors"
        return payload

    def tau2_state(self) -> vf.JsonData:
        return {
            "task_id": self.task_id,
            "step_count": len(self.messages),
            "num_errors": self.num_errors,
            "reward": 0.0,
            "messages": list(self.messages),
        }

    def initialize_environment(
        self, environment: Environment, task: TauTask
    ) -> list[Message]:
        initial_state = task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history or [])
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        for message in message_history:
            message.turn_idx = None
        environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        environment.sync_tools()
        return message_history

    def initial_user_message(
        self, message_history: list[Message]
    ) -> UserMessage | None:
        if not message_history:
            return None
        last_message = message_history[-1]
        if isinstance(last_message, UserMessage) and not last_message.is_tool_call():
            return last_message
        return None

    def initial_user_input(
        self, message_history: list[Message]
    ) -> UserInputMessage | None:
        if message_history:
            last_message = message_history[-1]
            if (
                isinstance(last_message, UserMessage)
                and not last_message.is_tool_call()
            ):
                return None
            if (
                isinstance(last_message, AssistantMessage)
                and not last_message.is_tool_call()
            ):
                return last_message
            if (
                isinstance(last_message, ToolMessage)
                and last_message.requestor == "user"
            ):
                return last_message
            raise ValueError(
                "Tau2 initial message history must end with a user message, "
                "assistant text message, or user tool result."
            )
        first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        first_message.timestamp = get_now()
        self.messages.append(self.message_data(first_message))
        return first_message

    def user_model_config(self, data: dict) -> tuple[str, dict]:
        model = data.get("model")
        if not isinstance(model, str) or not model:
            raise TypeError("Tau2 user model must be a non-empty string.")
        args = data.get("args")
        llm_args = dict(args) if isinstance(args, dict) else {}
        base_url = data.get("base_url")
        if isinstance(base_url, str) and base_url:
            llm_args.setdefault("api_base", base_url)
        api_key_var = data.get("api_key_var")
        if isinstance(api_key_var, str) and api_key_var:
            api_key = os.environ.get(api_key_var)
            if api_key is not None:
                llm_args.setdefault("api_key", api_key)
        return model, llm_args

    def user_tools(self, environment: Environment) -> list[TauUserTool] | None:
        try:
            return list(environment.get_user_tools())
        except ValueError:
            return None

    def generate_user_response(self, message: UserInputMessage) -> dict:
        if self.environment is None:
            raise RuntimeError("Tau2 environment has not started.")
        if self.user_simulator is None or self.user_state is None:
            raise RuntimeError("Tau2 user simulator has not started.")
        self.pending_user_input = None
        current = message
        while True:
            user_message, self.user_state = self.user_simulator.generate_next_message(
                current, self.user_state
            )
            self.messages.append(self.message_data(user_message))
            if self.user_simulator.is_stop(user_message):
                return {
                    "messages": [],
                    "tau2": self.tau2_state(),
                    "stop_condition": "tau2_user_done",
                }
            if not user_message.is_tool_call():
                return {
                    "messages": [self.v1_user_message(user_message)],
                    "tau2": self.tau2_state(),
                }
            tool_messages = [
                self.environment.get_response(tool_call)
                for tool_call in user_message.tool_calls or []
            ]
            self.record_tool_results(tool_messages)
            if self.too_many_errors():
                return {
                    "messages": [],
                    "tau2": self.tau2_state(),
                    "stop_condition": "tau2_too_many_errors",
                }
            current = (
                MultiToolMessage(role="tool", tool_messages=tool_messages)
                if len(tool_messages) > 1
                else tool_messages[0]
            )

    def record_assistant_tool_exchange(
        self, assistant_message: AssistantMessage, tool_messages: list[ToolMessage]
    ) -> None:
        self.messages.append(self.message_data(assistant_message))
        self.record_tool_results(tool_messages)

    def record_tool_results(self, tool_messages: list[ToolMessage]) -> None:
        for tool_message in tool_messages:
            if tool_message.error:
                self.num_errors += 1
            self.messages.append(self.message_data(tool_message))

    def too_many_errors(self) -> bool:
        return self.max_errors > 0 and self.num_errors >= self.max_errors

    @staticmethod
    def message_data(message: Message | ToolMessage) -> vf.JsonData:
        return message.model_dump(mode="json", exclude_none=True)

    @staticmethod
    def v1_user_message(message: UserMessage) -> vf.JsonData:
        return {"role": "user", "content": message.content or ""}

    @staticmethod
    def tau2_tool_defs(environment: Environment) -> list[vf.JsonData]:
        tool_defs: list[vf.JsonData] = []
        for tool in environment.get_tools():
            schema = tool.openai_schema
            function = schema.get("function") if isinstance(schema, dict) else None
            if not isinstance(function, dict):
                raise TypeError(
                    f"Tau2 tool {tool.name!r} did not expose OpenAI schema."
                )
            parameters = function.get("parameters")
            tool_defs.append(
                {
                    "name": str(function.get("name") or tool.name),
                    "description": str(function.get("description") or ""),
                    "parameters": parameters
                    if isinstance(parameters, dict)
                    else {"type": "object", "properties": {}},
                }
            )
        return tool_defs

    @staticmethod
    def latest_assistant_text(completion: list[dict]) -> str:
        for message in reversed(completion):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts)
        return ""
