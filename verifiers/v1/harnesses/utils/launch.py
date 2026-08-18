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
# use argv so tools do not inherit them.
CHAT_PROGRAM_SOURCE = bundle_program(
    '# /// script\n# requires-python = ">=3.10"\n'
    '# dependencies = ["openai", "mcp==2.0.0", "httpx", "httpx2", "tenacity"]\n'
    "# ///\n",
    mcp,
    compaction,
    core,
)


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
    stdin: bytes | None = None,
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
    process_env = env if env is not None else {**config.resolved_env}
    if stdin is not None:
        return await runtime.run_with_input([*program, *args], process_env, stdin)
    return await runtime.run_program([*program, *args], process_env)
