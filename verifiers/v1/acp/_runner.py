# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["agent-client-protocol==0.11.0"]
# ///
"""Run one harness segment through an ACP agent."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
    image_block,
    spawn_agent_process,
    text_block,
)
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    HttpMcpServer,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
)


class VerifiersClient(Client):
    def __init__(self) -> None:
        self.visible_reply = ""
        self.message_id: str | None = None

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        if not isinstance(update, AgentMessageChunk) or not isinstance(
            update.content, TextContentBlock
        ):
            return
        if update.message_id is not None and update.message_id != self.message_id:
            self.visible_reply = ""
            self.message_id = update.message_id
        self.visible_reply += update.content.text

    async def request_permission(
        self,
        session_id: str,
        tool_call: Any,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        option = next(
            (item for item in options if item.kind in ("allow_once", "allow_always")),
            None,
        )
        outcome = (
            AllowedOutcome(outcome="selected", option_id=option.option_id)
            if option
            else DeniedOutcome(outcome="cancelled")
        )
        return RequestPermissionResponse(outcome=outcome)


def content_blocks(messages: list[dict], supports_images: bool) -> list:
    blocks = []
    transcript = len(messages) != 1 or messages[0].get("role") != "user"
    for message in messages:
        if transcript:
            separator = "\n\n" if blocks else ""
            blocks.append(
                text_block(f"{separator}[{message.get('role', 'message')}]\n")
            )
        content = message.get("content") or ""
        parts = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        for part in parts:
            if part["type"] == "text":
                blocks.append(text_block(part["text"]))
                continue
            if not supports_images:
                raise ValueError("ACP agent does not support image prompts")
            url = part["image_url"]["url"]
            metadata, separator, data = url.partition(",")
            media_type, *parameters = metadata.removeprefix("data:").split(";")
            if (
                not separator
                or not metadata.startswith("data:image/")
                or not any(value.lower() == "base64" for value in parameters)
            ):
                raise ValueError("ACP image prompts require base64 data:image URLs")
            blocks.append(image_block(data, media_type))
        metadata = {
            key: value
            for key, value in message.items()
            if key not in ("role", "content") and value
        }
        if metadata:
            blocks.append(text_block("\n" + json.dumps(metadata, ensure_ascii=False)))
    return blocks


async def run_client(config: dict) -> None:
    client = VerifiersClient()
    command = config["command"]
    async with spawn_agent_process(
        client,
        command[0],
        *command[1:],
        env=os.environ.copy(),
        transport_kwargs={"stderr": None},
    ) as (connection, _process):
        initialized = await connection.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
        )
        capabilities = initialized.agent_capabilities
        prompt_capabilities = capabilities and capabilities.prompt_capabilities
        supports_images = bool(prompt_capabilities and prompt_capabilities.image)
        mcp_servers = [
            HttpMcpServer(type="http", name=name, url=url, headers=[])
            for name, url in config["mcp_urls"].items()
        ]
        session_path = Path(config["session_path"]) if config["session_path"] else None
        is_new = session_path is None or not session_path.exists()
        if is_new:
            session = await connection.new_session(
                cwd=os.getcwd(), mcp_servers=mcp_servers
            )
            session_id = session.session_id
        else:
            session_id = session_path.read_text().strip()
            session_capabilities = capabilities and capabilities.session_capabilities
            if session_capabilities and session_capabilities.resume is not None:
                await connection.resume_session(
                    cwd=os.getcwd(), session_id=session_id, mcp_servers=mcp_servers
                )
            elif capabilities and capabilities.load_session:
                await connection.load_session(
                    cwd=os.getcwd(), session_id=session_id, mcp_servers=mcp_servers
                )
            else:
                raise RuntimeError("ACP agent does not support resuming sessions")

        messages = config["messages"]
        if not is_new:
            last_assistant = max(
                (
                    index
                    for index, message in enumerate(messages)
                    if message.get("role") == "assistant"
                ),
                default=-1,
            )
            messages = messages[last_assistant + 1 :]
        if is_new and config["system_prompt"]:
            messages = [
                {"role": "system", "content": config["system_prompt"]},
                *messages,
            ]
        prompt = content_blocks(messages, supports_images)
        if not prompt:
            raise ValueError("ACP prompt has no content")
        client.visible_reply = ""
        client.message_id = None
        try:
            await connection.prompt(session_id=session_id, prompt=prompt)
        except RequestError as error:
            detail = error.data.get("details") if isinstance(error.data, dict) else None
            raise RuntimeError(detail or str(error)) from error
        if not client.visible_reply.strip():
            raise RuntimeError("ACP agent produced no visible reply")
        sys.stdout.write(client.visible_reply)
        if session_path and is_new:
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(session_id)


async def main() -> None:
    path = Path(sys.argv[1])
    config = json.loads(path.read_text())
    path.unlink()
    await run_client(config)


if __name__ == "__main__":
    asyncio.run(main())
