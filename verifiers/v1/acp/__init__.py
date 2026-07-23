"""Public Agent Client Protocol support for harness programs."""

import json
import secrets
from pathlib import Path

from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

ACP_SOURCE = (Path(__file__).resolve().parent / "_runner.py").read_text()

__all__ = ["ACP"]


class ACP:
    """Run an ACP agent."""

    async def setup(self, harness: Harness, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            ACP_SOURCE, {**harness.config.resolved_env, "UV_FROZEN": "false"}
        )

    async def run(
        self,
        runtime: Runtime,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        *,
        mcp_urls: dict[str, str] | None = None,
        system_prompt: str | None = None,
        session_path: str | None = None,
    ) -> ProgramResult:
        if prompt is None:
            raise ValueError("ACP requires a prompt")
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else [message_to_wire(message) for message in prompt]
        )
        config = {
            "command": command,
            "messages": messages,
            "mcp_urls": mcp_urls or {},
            "system_prompt": system_prompt or "",
            "session_path": session_path,
        }
        program = await runtime.prepare_uv_script(
            ACP_SOURCE, {**env, "UV_FROZEN": "false"}
        )
        directory = f".vf-acp-{secrets.token_hex(8)}"
        created = await runtime.run(["mkdir", "-m", "700", directory], {})
        if created.exit_code != 0:
            raise RuntimeError(f"ACP config directory failed: {created.stderr.strip()}")
        path = f"{directory}/config.json"
        try:
            await runtime.write(path, json.dumps(config).encode())
            result = await runtime.run_program([*program, path], env)
            return result
        finally:
            await run_shielded(runtime.run(["rm", "-rf", directory], {}))
