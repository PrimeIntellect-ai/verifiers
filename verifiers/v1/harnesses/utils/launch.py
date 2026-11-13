import inspect
import json
import uuid
from collections.abc import Sequence
from types import ModuleType

from pydantic import PositiveInt
from pydantic_config import BaseConfig

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.utils import compaction, core, mcp
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils import network

PEP_723_END = "# ///\n"


def bundle_program(program: str, *modules: ModuleType) -> str:
    """Embed utils modules so PEP 723 programs need only their declared packages."""
    metadata, body = program.split(PEP_723_END, 1)
    sources = "\n".join(inspect.getsource(module) for module in modules)
    return f"{metadata}{PEP_723_END}{sources}\n{body}"


# The shared Null/Bash chat program is the utils modules themselves: `core` ends with
# the `__main__` entry point, so the program text is only the script metadata. Secrets
# use argv so tools do not inherit them.
CHAT_PROGRAM = (
    '# /// script\n# requires-python = ">=3.10"\n'
    '# dependencies = ["openai", "mcp==2.3.0", "httpx", "httpx2", "tenacity", "certifi"]\n'
    "# ///\n"
)
CHAT_PROGRAM_SOURCE = bundle_program(CHAT_PROGRAM, network, mcp, compaction, core)


class CompactionConfig(BaseConfig):
    """Context compaction policy for the shared chat program."""

    summarize_at_tokens: PositiveInt | None = None
    """Compact at this token count. When unset, compact when 16k tokens remain below the
    model context window when the provider advertises it."""


def compaction_args(compaction: CompactionConfig | None) -> list[str]:
    """Chat-program flags for a compaction policy; `None` leaves compaction off."""
    if compaction is None:
        return []
    args = ["--compaction"]
    if compaction.summarize_at_tokens is not None:
        args.append(f"--summarize-at-tokens={compaction.summarize_at_tokens}")
    return args


async def stage_text(runtime: Runtime, trace: Trace, name: str, text: str) -> str:
    """Write `text` to a new file in `runtime` and return its path.

    Prompt text reaches programs by file, never on argv: an agent's `pkill -f <word>`
    kills every process whose command line contains the word, and Linux caps one
    argument at 128 KiB. Each call gets its own path, so concurrent launches on one
    trace do not collide. The program that reads the file deletes it."""
    path = f"/tmp/vf-{trace.id}-{uuid.uuid4().hex[:8]}-{name}"
    await runtime.write(path, text.encode())
    return path


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
        path = await stage_text(runtime, trace, "system-prompt.txt", system_prompt)
        args.append(f"--system-prompt-file={path}")
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
        initial = [{"role": "user", "content": prompt}] if prompt else []
    else:
        initial = [message_to_wire(message) for message in prompt or []]
    if initial:
        path = await stage_text(
            runtime, trace, "initial-messages.json", json.dumps(initial)
        )
        args.append(f"--initial-messages-file={path}")
    program = await runtime.prepare_uv_script(
        source, config.resolved_env, activate=activate
    )
    return await runtime.run(
        [*program, *args], env if env is not None else {**config.resolved_env}
    )
