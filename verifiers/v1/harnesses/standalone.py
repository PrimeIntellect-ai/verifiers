import json
from collections.abc import Sequence
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.mcp import client as mcp_client
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages

MCP_CLIENT_IMPORT = "from verifiers.v1.mcp.client import call_mcp, connect_mcp"
MCP_CLIENT_SOURCE = Path(mcp_client.__file__).read_text()


def inline_mcp_client(program: str) -> str:
    """Embed the public client so PEP 723 programs need only their declared packages."""
    return program.replace(MCP_CLIENT_IMPORT, MCP_CLIENT_SOURCE)


async def launch_chat_program(
    source: str,
    config: HarnessConfig,
    ctx: ModelContext,
    trace: Trace,
    runtime: Runtime,
    endpoint: str,
    secret: str,
    mcp_urls: dict[str, str],
    system_prompt: str | None,
    prompt: str | Messages | None,
    *,
    extra_args: Sequence[str] = (),
    env: dict[str, str] | None = None,
    activate: bool = True,
) -> ProgramResult:
    """Prepare and run a standalone chat program with the shared wire arguments."""
    args = [
        f"--base-url={endpoint}",
        f"--api-key={secret}",
        f"--model={ctx.model}",
        *extra_args,
    ]
    if system_prompt:
        args.append(f"--system-prompt={system_prompt}")
    if mcp_urls:
        args.append(
            "--mcp-config="
            + json.dumps(
                {
                    "mcpServers": {
                        name: {"url": url, "timeout": config.tool_timeout}
                        for name, url in mcp_urls.items()
                    }
                }
            )
        )
    if isinstance(prompt, str):
        args.append(f"--prompt={prompt}")
    elif prompt is not None:
        path = f".vf-initial-messages-{trace.id}.json"
        await runtime.write(
            path,
            json.dumps([message_to_wire(message) for message in prompt]).encode(),
        )
        args.append(f"--initial-messages-file={path}")
    program = await runtime.prepare_uv_script(
        source, config.resolved_env, activate=activate
    )
    return await runtime.run_program(
        [*program, *args], env if env is not None else {**config.resolved_env}
    )
