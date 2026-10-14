import inspect
import json
from collections.abc import Sequence
from types import ModuleType

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.utils import compaction, core, mcp
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages

PEP_723_END = "# ///\n"


def bundle_program(program: str, *modules: ModuleType) -> str:
    """Embed utils modules so PEP 723 programs need only their declared packages."""
    metadata, body = program.split(PEP_723_END, 1)
    sources = "\n".join(inspect.getsource(module) for module in modules)
    return f"{metadata}{PEP_723_END}{sources}\n{body}"


# The shared Null/Bash chat program is the utils modules themselves: `core` ends with
# the `__main__` entry point, so the program text is only the script metadata. Secrets
# reach it in its launch file, neither argv (`ps` shows it to the agent) nor its
# environment (which the agent's commands inherit).
CHAT_PROGRAM = (
    '# /// script\n# requires-python = ">=3.10"\n'
    '# dependencies = ["openai", "mcp==2.0.0", "httpx", "httpx2", "tenacity", "certifi"]\n'
    "# ///\n"
)
CHAT_PROGRAM_SOURCE = bundle_program(CHAT_PROGRAM, mcp, compaction, core)


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
    credentials: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    activate: bool = True,
) -> ProgramResult:
    """Prepare and run a standalone chat program with the shared wire arguments.
    `credentials` are extra secrets for the program's launch file, keyed as it reads
    them (the bash program's `serper_key`)."""
    args = [f"--model={ctx.model}", *extra_args]
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
    # The endpoint, its key and the conversation go in a file the program reads and
    # deletes, never argv: the box's `ps` would show the key to the agent (and into
    # its trace), and an agent command such as `pkill -f <word from the task>` would
    # match, and kill, its own harness.
    initial: list[dict] = []
    if system_prompt:
        initial.append({"role": "system", "content": system_prompt})
    if isinstance(prompt, str):
        if prompt:
            initial.append({"role": "user", "content": prompt})
    elif prompt is not None:
        initial.extend(message_to_wire(message) for message in prompt)
    launch = {
        **(credentials or {}),
        "base_url": endpoint,
        "api_key": secret,
        "messages": initial,
    }
    path = f".vf-launch-{trace.id}.json"
    await runtime.write(path, json.dumps(launch).encode())
    await runtime.run(["chmod", "600", path], {})
    args.append(f"--launch-file={path}")
    program = await runtime.prepare_uv_script(
        source, config.resolved_env, activate=activate
    )
    return await runtime.run_program(
        [*program, *args], env if env is not None else {**config.resolved_env}
    )
