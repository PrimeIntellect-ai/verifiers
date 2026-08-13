"""RLM ACP entrypoint with exact VF conversation import."""

from __future__ import annotations

import asyncio
from typing import Any

from acp import RequestError, run_agent
from rlm.acp import RLMACPAgent

SESSION_IMPORT = "verifiers.dev/sessionImport"


def _chat_messages(messages: object) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")

    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            converted.append({"role": role, "content": content})
        elif role == "assistant":
            wire: dict[str, Any] = {"role": role, "content": content or ""}
            provider_state = message.get("provider_state")
            reasoning = message.get("reasoning_content")
            if provider_state:
                if not all(
                    str(item.get("type", "")).startswith("reasoning.")
                    for item in provider_state
                ):
                    raise ValueError(
                        "RLM can import Chat reasoning_details provider state only"
                    )
                wire["reasoning_details"] = provider_state
            if reasoning is not None:
                wire["reasoning_content"] = reasoning

            if tool_calls := message.get("tool_calls"):
                wire["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    for call in tool_calls
                ]
            converted.append(wire)
        elif role == "tool":
            wire = {
                "role": role,
                "tool_call_id": message["tool_call_id"],
                "content": content,
            }
            if name := message.get("name"):
                wire["name"] = name
            converted.append(wire)
        else:
            raise ValueError(f"RLM cannot import {role!r} messages")
    if not converted or converted[0]["role"] != "user":
        raise ValueError("RLM session history must start with a user message")
    return converted


class ImportingRLMACPAgent(RLMACPAgent):
    async def initialize(self, *args: Any, **kwargs: Any):
        response = await super().initialize(*args, **kwargs)
        metadata = response.agent_capabilities.field_meta or {}
        response.agent_capabilities.field_meta = {
            **metadata,
            SESSION_IMPORT: {"version": 1},
        }
        return response

    async def new_session(self, *args: Any, **kwargs: Any):
        extension = kwargs.pop(SESSION_IMPORT, None)
        if extension is None:
            return await super().new_session(*args, **kwargs)
        try:
            if not isinstance(extension, dict) or extension.get("version") != 1:
                raise ValueError("unsupported session import version")
            messages = _chat_messages(extension.get("messages"))
        except (TypeError, ValueError) as error:
            raise RequestError.invalid_params({"reason": str(error)}) from error

        response = await super().new_session(*args, **kwargs)
        state = self._sessions[response.session_id]
        try:
            # RLM builds its system prompt and persistent IPython kernel here; no
            # model call occurs until Engine.prompt() continues into its run loop.
            await state.engine._start(messages[0]["content"])
            assert state.engine._messages is not None
            state.engine._messages[1:] = messages
            state.engine._turn = sum(
                message["role"] == "assistant" for message in messages
            )
            state.engine._branch_start_turn = state.engine._turn
        except BaseException:
            state.engine.close()
            self._sessions.pop(response.session_id, None)
            raise
        return response


async def main() -> None:
    agent = ImportingRLMACPAgent()
    try:
        await run_agent(agent, use_unstable_protocol=True)
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
