# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["hermes-agent[acp,mcp]=={version}"]
# ///
"""Start Hermes Agent's native ACP server."""

import os
from copy import deepcopy
from typing import Any

from hermes_cli.models import detect_provider_for_model
from hermes_cli.providers import determine_api_mode

SESSION_IMPORT = "verifiers.dev/sessionImport"


def native_messages(
    messages: list[dict[str, Any]], api_mode: str
) -> list[dict[str, Any]]:
    """Convert typed VF messages into Hermes' persisted conversation shape."""
    native = []
    for message in messages:
        role = message["role"]
        imported = {"role": role, "content": deepcopy(message.get("content"))}
        if role == "tool":
            imported["tool_call_id"] = message["tool_call_id"]
            if name := message.get("name"):
                imported["tool_name"] = name
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            imported["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in calls
            ]
            reasoning = message.get("reasoning_content")
            if reasoning is not None:
                imported["reasoning_content"] = reasoning
            state = deepcopy(message.get("provider_state") or [])
            if api_mode == "codex_responses" and reasoning is not None and not state:
                raise ValueError(
                    "exact Responses reasoning history requires provider_state"
                )
            if state and api_mode == "codex_responses":
                if any(
                    item.get("type") not in {"reasoning", "message", "function_call"}
                    for item in state
                ):
                    raise ValueError(
                        "Hermes cannot exactly import this Responses provider state"
                    )
                reasoning_items = [
                    item for item in state if item.get("type") == "reasoning"
                ]
                message_items = [
                    item for item in state if item.get("type") == "message"
                ]
                function_items = [
                    item for item in state if item.get("type") == "function_call"
                ]
                try:
                    invalid_messages = any(
                        item.get("role") != "assistant"
                        or any(
                            part.get("type") != "output_text"
                            for part in item.get("content") or []
                        )
                        or any(
                            item.get(key) is not None
                            for key in ("encrypted_content", "signature", "data")
                        )
                        for item in message_items
                    )
                    state_content = "".join(
                        part.get("text", "")
                        for item in message_items
                        for part in item.get("content") or []
                    )
                    state_reasoning = "\n".join(
                        part.get("text", "")
                        for item in reasoning_items
                        for part in [
                            *(item.get("summary") or []),
                            *(item.get("content") or []),
                        ]
                        if part.get("text")
                    )
                except (AttributeError, TypeError):
                    invalid_messages = True
                    state_content = state_reasoning = ""
                replayed_calls = [
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                    for call in calls
                ]
                if (
                    invalid_messages
                    or state_content != (message.get("content") or "")
                    or state_reasoning != (reasoning or "")
                    or function_items != replayed_calls
                ):
                    raise ValueError(
                        "Hermes cannot exactly import this Responses provider state"
                    )
                if reasoning_items:
                    imported["codex_reasoning_items"] = reasoning_items
                if message_items:
                    imported["codex_message_items"] = message_items
            elif state:
                imported["reasoning_details"] = state
        native.append(imported)
    return native


model = os.environ["HERMES_INFERENCE_MODEL"].rsplit("/", 1)[-1]
provider, _ = detect_provider_for_model(model, "auto") or ("auto", model)
os.environ.setdefault("HERMES_INTERCEPT_TRANSPORT", determine_api_mode(provider))

import acp_adapter.server
from acp_adapter.server import HermesACPAgent


class VerifiersHermesACPAgent(HermesACPAgent):
    async def initialize(self, *args: Any, **kwargs: Any):
        response = await super().initialize(*args, **kwargs)
        capabilities = response.agent_capabilities
        capabilities.field_meta = {
            **(capabilities.field_meta or {}),
            SESSION_IMPORT: {"version": 1},
        }
        return response

    async def new_session(self, *args: Any, **kwargs: Any):
        session_import = kwargs.get(SESSION_IMPORT)
        if session_import is None:
            return await super().new_session(*args, **kwargs)
        if not isinstance(session_import, dict) or session_import.get("version") != 1:
            raise ValueError("unsupported verifiers.dev/sessionImport version")
        messages = session_import.get("messages")
        if not isinstance(messages, list):
            raise TypeError("verifiers.dev/sessionImport.messages must be a list")
        imported = native_messages(messages, os.environ["HERMES_INTERCEPT_TRANSPORT"])
        response = await super().new_session(*args, **kwargs)
        state = self.session_manager.get_session(response.session_id)
        if state is None:
            raise RuntimeError("Hermes did not retain its newly created session")
        try:
            db = self.session_manager._get_db()
            if db is None:
                raise RuntimeError("Hermes session persistence is unavailable")
            db.replace_messages(state.session_id, imported)
            state.history = imported
        except Exception:
            self.session_manager.remove_session(response.session_id)
            raise
        return response


acp_adapter.server.HermesACPAgent = VerifiersHermesACPAgent

from acp_adapter.entry import main

if __name__ == "__main__":
    main()
