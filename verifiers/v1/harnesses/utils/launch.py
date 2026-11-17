import asyncio
import inspect
import json
import logging
import traceback
import uuid
from collections.abc import Collection, Mapping, Sequence
from contextlib import AsyncExitStack, suppress
from types import ModuleType
from urllib.parse import urlsplit

import certifi
import httpx
from openai import APIStatusError, AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import PositiveInt
from pydantic_config import BaseConfig

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import RolloutError, SandboxError
from verifiers.v1.harnesses.utils import compaction, core, mcp
from verifiers.v1.harnesses.utils.install import remove_dir
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils import network
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import clip, get_logger

logger = get_logger(__name__)

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


async def stage_text(
    runtime: Runtime,
    trace: Trace,
    name: str,
    text: str,
    *,
    env: dict[str, str] | None = None,
) -> str:
    """Write `text` to a new file in `runtime` and return its path.

    Prompt text reaches programs by file, never on argv: an agent's `pkill -f <word>`
    kills every process whose command line contains the word, and Linux caps one
    argument at 128 KiB. Each call gets its own path, so concurrent launches on one
    trace do not collide. The program that reads the file deletes it."""
    stem, dot, suffix = name.partition(".")
    path = runtime.temp_path(
        f"{stem}-{trace.id}-{uuid.uuid4().hex[:8]}{dot}{suffix}", env=env
    )
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
    process_env = env if env is not None else {**config.resolved_env}
    prompt_files = []
    try:
        if system_prompt:
            path = await stage_text(
                runtime, trace, "system-prompt.txt", system_prompt, env=process_env
            )
            prompt_files.append(path)
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
                runtime,
                trace,
                "initial-messages.json",
                json.dumps(initial),
                env=process_env,
            )
            prompt_files.append(path)
            args.append(f"--initial-messages-file={path}")
        program = await runtime.prepare_uv_script(
            source, config.resolved_env, activate=activate
        )
        return await runtime.run([*program, *args], process_env)
    finally:
        for path in prompt_files:
            try:
                await remove_dir(runtime, path, "prompt file")
            except Exception as error:
                # Teardown is the backstop if the runtime is gone or the agent
                # changed permissions. Preserve the result or original failure.
                logger.warning(
                    "Failed to clean up prompt file - %s: %s",
                    type(error).__name__,
                    clip(str(error)),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )


def has_internal_mcp(
    mcp_urls: Mapping[str, str], internal_hosts: Collection[str] = ()
) -> bool:
    """Whether any MCP URL targets a deployment-internal host (such as a Prime
    `.sandbox.internal` LAN VM or Compose service) unreachable from the host."""
    for url in mcp_urls.values():
        host = (urlsplit(url).hostname or "").lower()
        if host.endswith(".sandbox.internal") or host in internal_hosts:
            return True
    return False


def _format_bash_output(stdout: core._BashOutput, stderr: core._BashOutput) -> str:
    size = stdout.size + stderr.size
    if size <= compaction.TOOL_OUTPUT_MAX_BYTES:
        text = "".join(
            output.head.decode(errors="replace")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            for output in (stdout, stderr)
        )
    else:
        marker = f"\n[... bash output truncated; {size} bytes total ...]\n"
        keep = (compaction.TOOL_OUTPUT_MAX_BYTES - len(marker.encode())) // 2
        head = (stdout.head + stderr.head)[:keep].decode(errors="ignore")
        tail = (stdout.tail + stderr.tail)[-keep:].decode(errors="ignore")
        text = head + marker + tail
    return text.replace("\r\n", "\n").replace("\r", "\n")


async def _run_bash_in_runtime(
    runtime: Runtime, command: object, env: dict[str, str]
) -> str:
    if not isinstance(command, str):
        return "error: 'command' must be a string"
    backend = getattr(runtime, "backend", None)
    if backend is not None and hasattr(backend, "ensure_started"):
        await backend.ensure_started()
    stdout, stderr = core._BashOutput(), core._BashOutput()
    try:
        if hasattr(runtime, "exec"):
            process = await runtime.exec(["bash", "-c", command], env)

            async def _pump(stream, sink: core._BashOutput) -> None:
                async for chunk in stream:
                    sink.add(chunk)

            try:
                async with asyncio.timeout(3600):
                    await asyncio.gather(
                        _pump(process.stdout, stdout),
                        _pump(process.stderr, stderr),
                        process.wait(),
                    )
            except BaseException:
                with suppress(Exception):
                    await run_shielded(process.kill())
                    await run_shielded(process.wait())
                raise
            return _format_bash_output(stdout, stderr)
        result = await asyncio.wait_for(
            runtime.run(["bash", "-c", command], env), timeout=3600
        )
        stdout.add(result.stdout.encode())
        stderr.add(result.stderr.encode())
        return _format_bash_output(stdout, stderr)
    except TimeoutError:
        return "error: Command '['bash', '-c', ...]' timed out after 3600 seconds"
    except RolloutError:
        raise
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"error: {e}"


async def _run_edit_in_runtime(
    runtime: Runtime, path: object, old_str: object, new_str: object
) -> str:
    if not isinstance(path, str) or not path:
        return "error: 'path' is required"
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        return "error: 'old_str' and 'new_str' must be strings"
    if not old_str:
        return "error: 'old_str' must be a non-empty string"
    backend = getattr(runtime, "backend", None)
    if backend is not None and hasattr(backend, "ensure_started"):
        await backend.ensure_started()
    try:
        raw = await runtime.read(path)
    except (SandboxError, OSError) as e:
        return f"error: could not read {path}: {e}"
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return f"error: could not read {path}: {e}"
    count = content.count(old_str)
    if count != 1:
        return f"error: old_str must appear exactly once in {path} (found {count})"
    try:
        await runtime.write(path, content.replace(old_str, new_str, 1).encode("utf-8"))
    except (SandboxError, OSError) as e:
        return f"error: could not write {path}: {e}"
    return f"Edited {path}"


async def run_in_process_chat(
    config: HarnessConfig,
    ctx: ModelContext,
    runtime: Runtime,
    endpoint: str,
    secret: str,
    mcp_urls: dict[str, str],
    system_prompt: str | None,
    prompt: str | Messages | None,
    *,
    bash: bool = False,
    edit: bool = False,
    search: bool = False,
    serper_key: str = "",
    search_block: list[str] | None = None,
    compaction_cfg: CompactionConfig | None = None,
    tool_interception_url: str | None = None,
    env: dict[str, str] | None = None,
) -> ProgramResult:
    """Drive the shared `null`/`bash` chat loop directly on the host event loop,
    provisioning `runtime` lazily only when `bash` or `edit` executes."""
    process_env = env if env is not None else {**config.resolved_env}
    disabled = set(getattr(config, "disabled_tools", None) or ())
    enable_bash = bash and "bash" not in disabled
    enable_edit = edit and "edit" not in disabled
    enable_search = search and "search" not in disabled

    tools: list[dict] = []
    reserved: set[str] = set()
    if enable_bash:
        tools.append(core.BASH_TOOL)
        reserved.add("bash")
    if enable_edit:
        tools.append(core.EDIT_TOOL)
        reserved.add("edit")
    if enable_search:
        tools.append(core.SEARCH_TOOL)
        reserved.add("search")

    if isinstance(prompt, str):
        initial = [{"role": "user", "content": prompt}] if prompt else []
    else:
        initial = [message_to_wire(message) for message in prompt or []]
    messages: list[dict] = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    messages.extend(initial)

    mcp_cfg = (
        {
            "mcpServers": {
                name: {"url": url, "timeout": config.tool_timeout}
                for name, url in mcp_urls.items()
            }
        }
        if mcp_urls
        else {}
    )

    client = AsyncOpenAI(
        base_url=endpoint,
        api_key=secret,
        timeout=httpx.Timeout(600.0 if bash else None, connect=5.0),
        http_client=DefaultAsyncHttpxClient(verify=certifi.where()),
    )
    tool_client = (
        httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))
        if tool_interception_url
        else None
    )
    try:
        async with AsyncExitStack() as mcp_stack:
            if mcp_cfg.get("mcpServers"):
                mcp_tools, dispatch, servers = await asyncio.wait_for(
                    mcp.connect_mcp(mcp_cfg, mcp_stack, reserved),
                    timeout=None if bash else 60,
                )
                mcp_tools = [
                    t for t in mcp_tools if t["function"]["name"] not in disabled
                ]
            else:
                mcp_tools, dispatch, servers = [], {}, {}
            all_tools = [*tools, *mcp_tools]
            compactor = compaction.Compactor(
                client,
                ctx.model,
                all_tools,
                compaction_cfg is not None,
                compaction_cfg.summarize_at_tokens
                if compaction_cfg is not None
                else None,
            )
            if compactor.enabled and compactor.threshold is None:
                compactor.threshold = await compaction.discover_threshold(
                    client, ctx.model
                )
            compactor.note_good(messages)

            while True:
                try:
                    completion, messages = await compactor.complete(messages)
                except APIStatusError as error:
                    compacting = compactor.enabled and compactor.threshold is not None
                    if compacting or not compaction.is_context_overflow(error):
                        raise
                    break
                message = completion.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    break
                tool_result_tokens = 0
                for call in message.tool_calls:
                    name = call.function.name
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "",
                        "name": name,
                    }
                    if tool_interception_url:
                        assert tool_client is not None
                        decision = await core.gate_tool_call(
                            tool_client, tool_interception_url, secret, call
                        )
                        if decision["action"] == "deny":
                            denied = compaction.bound_tool_message(decision["message"])
                            messages.append(denied)
                            tool_result_tokens += compaction.estimated_tokens(
                                str(denied.get("content", ""))
                            )
                            continue
                    try:
                        tool_args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        content: str | list[dict] = (
                            f"error: invalid JSON in tool arguments ({e}); resend the call with valid JSON"
                        )
                    else:
                        if not isinstance(tool_args, dict):
                            content = (
                                f"error: tool arguments must be a JSON object, got "
                                f"{type(tool_args).__name__}; resend as an object"
                            )
                        elif name in dispatch and name not in disabled:
                            content = await mcp.call_mcp(
                                servers, dispatch, name, tool_args
                            )
                        elif name == "bash" and enable_bash:
                            content = await _run_bash_in_runtime(
                                runtime, tool_args.get("command", ""), process_env
                            )
                        elif name == "edit" and enable_edit:
                            content = await _run_edit_in_runtime(
                                runtime,
                                tool_args.get("path"),
                                tool_args.get("old_str"),
                                tool_args.get("new_str"),
                            )
                        elif name == "search" and enable_search:
                            content = await asyncio.to_thread(
                                core.run_search,
                                tool_args.get("query", ""),
                                serper_key,
                                tool_args.get("num_results", 5),
                                search_block,
                            )
                        else:
                            content = f"error: unknown tool {name!r}"
                    tool_message["content"] = content
                    tool_message = compaction.bound_tool_message(tool_message)
                    messages.append(tool_message)
                    tool_result_tokens += compaction.estimated_tokens(
                        str(tool_message["content"])
                    )
                if compactor.reached(
                    completion, tool_result_tokens
                ) and compaction.compactable(messages):
                    messages = await compactor.compact(messages)
        return ProgramResult(exit_code=0, stdout="", stderr="")
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        return ProgramResult(exit_code=code, stdout="", stderr=traceback.format_exc())
    except RolloutError:
        raise
    except Exception as exc:  # noqa: BLE001 - mapped to HarnessError / stop check by _check_result
        return ProgramResult(
            exit_code=1, stdout="", stderr=f"{type(exc).__name__}: {exc}"
        )
    finally:
        await client.close()
        if tool_client is not None:
            await tool_client.aclose()
