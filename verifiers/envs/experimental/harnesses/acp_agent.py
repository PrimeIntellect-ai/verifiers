from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any, cast

import verifiers as vf
from acp import (
    PROTOCOL_VERSION,
    Agent as ACPAgent,
    Client as ACPClient,
    spawn_agent_process,
    text_block,
)
from acp.schema import AgentMessageChunk, TextContentBlock

from verifiers.envs.experimental.harnesses.base import Harness
from verifiers.types import (
    FinishReason,
    Messages,
    Response,
    ResponseMessage,
    State,
    TextContentPart,
    Tool,
)


def _user_prompt_text(prompt: Messages | str) -> str:
    if isinstance(prompt, str):
        return prompt.strip()

    for message in reversed(prompt):
        role = message.get("role") if isinstance(message, dict) else message.role
        if role != "user":
            continue

        content = (
            message.get("content") if isinstance(message, dict) else message.content
        )
        if isinstance(content, str):
            text = content.strip()
        elif not isinstance(content, list):
            text = ""
        else:
            text_parts: list[str] = []
            for part in content:
                part_type = part.get("type") if isinstance(part, dict) else part.type
                if part_type != "text":
                    continue
                if isinstance(part, dict):
                    text_value = part.get("text")
                elif isinstance(part, TextContentPart):
                    text_value = part.text
                else:
                    continue
                if isinstance(text_value, str):
                    text_parts.append(text_value)
            text = "".join(text_parts).strip()

        if text:
            return text

    raise vf.InfraError("Could not find a user message in the rollout prompt.")


class ACPCollector:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def on_connect(self, conn: Any) -> None:
        pass

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        if not isinstance(update, AgentMessageChunk):
            return
        if not isinstance(update.content, TextContentBlock):
            return
        self.parts.append(update.content.text)

    @property
    def message(self) -> str:
        return "".join(self.parts).strip()


class ACPHarness(Harness):
    """Barebones local ACP harness with minimal state parity with CliHarness."""

    DEFAULT_COMMAND = ("opencode", "acp")

    def __init__(
        self,
        command: Sequence[str] = DEFAULT_COMMAND,
        cwd: str | None = None,
        session_model_id: str | None = None,
        mcp_servers: list[Any] | None = None,
        system_prompt: str = "",
        timeout_seconds: float = 900.0,
        tools: list[str] | None = None,
    ):
        super().__init__(
            tools=tools,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds,
        )
        self.command = tuple(command)
        self.cwd = cwd
        self.session_model_id = session_model_id
        self.mcp_servers = list(mcp_servers or [])

    async def setup(self, env: Any, state: State) -> None:
        await super().setup(env, state)
        state.setdefault("acp_command", list(self.command))
        state.setdefault("acp_cwd", self.cwd or os.getcwd())
        state.setdefault("acp_session_model_id", self.session_model_id)
        state.setdefault("acp_mcp_servers", list(self.mcp_servers))
        state.setdefault("agent_exit_code", None)
        state.setdefault("agent_stdout", "")
        state.setdefault("agent_stderr", "")

    async def start(self, env: Any, state: State) -> None:
        state["agent_start_time"] = time.time()

    async def build_env_vars(self, env: Any, state: State) -> dict[str, str]:
        env_vars: dict[str, str] = {}
        resolved_model = self.resolve_session_model_id(state, None)
        if resolved_model:
            env_vars["OPENAI_MODEL"] = resolved_model
        return env_vars

    def normalize_response(self, env: Any, response: Response) -> Response:
        message = response.message
        content = message.content
        if isinstance(content, str):
            content = content.rstrip()
        normalized_message = message.model_copy(
            update={
                "content": content or "",
                "reasoning_content": None,
                "tool_calls": None,
            }
        )
        return response.model_copy(update={"message": normalized_message})

    def resolve_command(self, state: State) -> tuple[str, ...]:
        value = state.get("acp_command") or self.command
        if isinstance(value, str):
            return (value,)
        return tuple(str(part) for part in value)

    def resolve_cwd(self, state: State) -> str:
        value = state.get("acp_cwd") or self.cwd or os.getcwd()
        return str(value)

    def resolve_session_model_id(
        self,
        state: State,
        model: str | None,
    ) -> str | None:
        value = state.get("acp_session_model_id")
        if value:
            return str(value)
        return self.session_model_id or model or state.get("model")

    def resolve_mcp_servers(self, state: State) -> list[Any]:
        value = state.get("acp_mcp_servers")
        if value is None:
            return list(self.mcp_servers)
        return list(value)

    def build_prompt_text(self, prompt: Messages | str, state: State) -> str:
        prompt_text = state.get("acp_prompt_text")
        if not prompt_text:
            prompt_text = _user_prompt_text(prompt)

        system_prompt = state.get("acp_system_prompt", self.system_prompt)
        if system_prompt:
            return f"{system_prompt}\n\n{prompt_text}"
        return str(prompt_text)

    def build_client(
        self,
        collector: ACPCollector,
    ) -> Callable[[ACPAgent], ACPClient]:
        def to_client(agent: ACPAgent) -> ACPClient:
            collector.on_connect(agent)
            return cast(ACPClient, collector)

        return to_client

    async def get_model_response(
        self,
        env: Any,
        state: State,
        prompt: Messages | str,
        client: vf.Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: vf.SamplingArgs | None = None,
    ) -> Response:
        resolved_model = self.resolve_session_model_id(state, model) or "acp-agent"
        if not prompt:
            state["agent_completed"] = True
            state["agent_exit_code"] = 0
            state["agent_stdout"] = ""
            state["agent_stderr"] = ""
            return Response(
                id="acp-agent-completed",
                created=int(time.time()),
                model=resolved_model,
                usage=None,
                message=ResponseMessage(
                    content="",
                    reasoning_content=None,
                    tool_calls=None,
                    finish_reason="stop",
                    is_truncated=False,
                    tokens=None,
                ),
            )

        state.setdefault("agent_start_time", time.time())
        prompt_text = self.build_prompt_text(prompt, state)
        command = self.resolve_command(state)
        cwd = self.resolve_cwd(state)
        mcp_servers = self.resolve_mcp_servers(state)
        session_model_id = self.resolve_session_model_id(state, model)

        try:
            content, finish_reason, model_name = await self.run_acp_prompt(
                command=command,
                prompt_text=prompt_text,
                cwd=cwd,
                session_model_id=session_model_id,
                mcp_servers=mcp_servers,
            )
        except vf.InfraError as exc:
            state["agent_timed_out"] = "Timed out waiting for ACP agent" in str(exc)
            state["agent_exit_code"] = 124 if state["agent_timed_out"] else 1
            state["agent_stderr"] = str(exc)
            raise
        except vf.Error as exc:
            state["agent_exit_code"] = 1
            state["agent_stderr"] = str(exc)
            raise

        state["acp_last_prompt_text"] = prompt_text
        state["agent_exit_code"] = 0
        state["agent_stdout"] = content
        state["agent_stderr"] = ""
        state["agent_model"] = model_name or resolved_model
        state["agent_finish_reason"] = finish_reason

        return Response(
            id=f"acp-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model_name or resolved_model,
            usage=None,
            message=ResponseMessage(
                content=content,
                reasoning_content=None,
                tool_calls=None,
                finish_reason=finish_reason,
                is_truncated=finish_reason == "length",
                tokens=None,
            ),
        )

    async def add_model_response(
        self,
        env: Any,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ) -> None:
        if not prompt_messages:
            state["agent_completed"] = True
            return
        if len(state["trajectory"]) == 0:
            state["prompt"] = prompt_messages
        await env.record_model_response(
            state,
            prompt_messages,
            env.normalize_response(response),
        )
        state["agent_completed"] = True

    async def cleanup(self, env: Any, state: State) -> None:
        state.pop("acp_mcp_servers", None)

    async def teardown(self, env: Any) -> None:
        pass

    async def run_acp_prompt(
        self,
        *,
        command: Sequence[str],
        prompt_text: str,
        cwd: str,
        session_model_id: str | None,
        mcp_servers: list[Any],
    ) -> tuple[str, FinishReason, str]:
        collector = ACPCollector()

        try:
            message, finish_reason, model_name = await asyncio.wait_for(
                self._run_prompt(
                    collector=collector,
                    command=command,
                    cwd=cwd,
                    prompt_text=prompt_text,
                    session_model_id=session_model_id,
                    mcp_servers=mcp_servers,
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise vf.InfraError(
                f"Timed out waiting for ACP agent after {self.timeout_seconds} seconds."
            ) from exc

        if not message:
            raise vf.ModelError("ACP agent returned no assistant message content.")

        return message, finish_reason, model_name

    async def _run_prompt(
        self,
        *,
        collector: ACPCollector,
        command: Sequence[str],
        cwd: str,
        prompt_text: str,
        session_model_id: str | None,
        mcp_servers: list[Any],
    ) -> tuple[str, FinishReason, str]:
        async with spawn_agent_process(
            self.build_client(collector),
            command[0],
            *command[1:],
            cwd=cwd,
        ) as (
            conn,
            _process,
        ):
            await conn.initialize(PROTOCOL_VERSION)
            session = await conn.new_session(cwd=cwd, mcp_servers=mcp_servers)

            model_name = ""
            if session.models is not None:
                model_name = session.models.current_model_id
            if session_model_id and session_model_id != model_name:
                await conn.set_session_model(
                    model_id=session_model_id,
                    session_id=session.session_id,
                )
                model_name = session_model_id

            prompt_response = await conn.prompt(
                session_id=session.session_id,
                prompt=[text_block(prompt_text)],
            )

        stop_reason = getattr(prompt_response, "stop_reason", None)
        finish_reason: FinishReason = (
            "length" if stop_reason in {"max_tokens", "max_turn_requests"} else "stop"
        )
        return collector.message, finish_reason, model_name
