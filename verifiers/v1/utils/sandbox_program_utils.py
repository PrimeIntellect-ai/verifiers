import importlib.machinery
import importlib.util
import json
import shlex
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from verifiers.errors import Error
from verifiers.utils.error_utils import error_from_data, validate_error_data
from verifiers.utils.interception_utils import serialize_tool_defs

from ..runtime import Runtime
from ..sandbox import SandboxConfig
from ..state import State
from ..task import Task
from .serialization_utils import serializable
from .sandbox_utils import (
    VF_STATE_INPUT_PATH_KEY,
    read_sandbox_artifact,
    run_sandbox_command,
    scrub_sandbox_private_fields,
)
from .sandbox_python_utils import (
    python_package_list,
    python_package_install_command,
    python_runtime_command,
    python_runtime_setup_command,
)
from .program_utils import (
    ProgramListInput,
    ProgramMappingInput,
    program_list_items,
    program_option_mapping,
)
from ..types import ConfigData

TASK_PATH = "/tmp/vf_task.json"
STATE_INPUT_PATH = "/tmp/vf_state_in.json"
STATE_OUTPUT_PATH = "/tmp/vf_state_out.json"
RUNNER_CONFIG_PATH = "/tmp/vf_runner_config.json"
TOOL_DEFS_PATH = "/tmp/vf_tool_defs.json"
TOOL_DEFS_BY_PROTOCOL_PATH = "/tmp/vf_tool_defs_by_protocol.json"
RUNNER_PATH = "/tmp/vf_program_runner.py"
PYTHON_PROGRAM_PACKAGES = ("openai", "anthropic", "requests")
PACKAGE_ROOT = "/tmp/vf_program_package"

def python_program_sandbox(sandbox_config: ConfigData) -> ConfigData:
    config = dict(sandbox_config)
    packages = python_package_list(config.get("packages"))
    for package in PYTHON_PROGRAM_PACKAGES:
        if not any(is_python_package(existing, package) for existing in packages):
            packages.append(package)
    config["packages"] = packages
    return config


def is_python_package(requirement: str, package: str) -> bool:
    return (
        requirement == package
        or requirement.startswith(f"{package}[")
        or requirement.startswith(f"{package}=")
        or requirement.startswith(f"{package}<")
        or requirement.startswith(f"{package}>")
        or requirement.startswith(f"{package}~")
        or requirement.startswith(f"{package}!")
    )


async def run_sandbox_python_program(
    program: ConfigData,
    sandbox_config: SandboxConfig,
    task: Task,
    state: State,
    runtime: Runtime,
    mode: str,
    fn_ref: str | None,
    max_turns: int,
) -> State:
    runner_program = sandbox_runner_program(
        program=program,
        task=task,
        state=state,
        mode=mode,
        fn_ref=fn_ref,
        max_turns=max_turns,
        tool_defs=runtime.tool_defs(state),
    )
    command_record = state.get("command")
    await run_sandbox_command(runner_program, sandbox_config, task, state, runtime)
    lease = runtime.active_program_sandbox_lease(state)
    if lease is None:
        raise RuntimeError("Sandbox Python program has no active sandbox lease.")
    output = json.loads(
        await read_sandbox_artifact(lease.client, lease.id, STATE_OUTPUT_PATH)
    )
    if not isinstance(output, dict):
        raise RuntimeError("Sandbox Python program did not return state.")
    patch = dict(cast(ConfigData, output))
    apply_internal_state_patch(state, patch, mode=mode)
    patch_artifacts = patch.pop("artifacts", None)
    if isinstance(patch_artifacts, dict):
        state.setdefault("artifacts", {})
        state["artifacts"].update(dict(patch_artifacts))
    state.update(patch)
    if command_record is not None:
        state["command"] = command_record
    return state


def apply_internal_state_patch(state: State, patch: ConfigData, *, mode: str) -> None:
    for key in State.INTERNAL_KEYS:
        if key not in patch:
            continue
        value = patch.pop(key)
        if value == state.get(key):
            continue
        if mode != "base" or key == "is_completed":
            raise RuntimeError(
                f"Sandbox Python program cannot set framework-managed state key {key!r}."
            )
        if key == "stop_condition":
            state._set_stop_condition(cast(str | None, value), overwrite=True)
        elif key == "is_truncated":
            state._set_truncated(bool(value), overwrite=True)
        elif key == "error":
            state._set_error(state_error(value))
        else:
            raise RuntimeError(
                f"Sandbox Python program cannot set framework-managed state key {key!r}."
            )


def state_error(value: object) -> Error | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Sandbox Python program error patch must be a mapping or None.")
    return error_from_data(validate_error_data(value))


def sandbox_runner_program(
    program: ConfigData,
    task: Task,
    state: State,
    mode: str,
    fn_ref: str | None,
    max_turns: int,
    tool_defs: object,
) -> ConfigData:
    package = sandbox_program_package(mode=mode, fn_ref=fn_ref)
    if package is not None:
        program = sandbox_program_with_package(program, package)
    files = program_option_mapping(
        cast(ProgramMappingInput, program.get("files")), "program.files"
    )
    files[TASK_PATH] = json.dumps(scrub_sandbox_private_fields(task))
    files[TOOL_DEFS_PATH] = json.dumps(
        serializable(serialize_tool_defs(tool_defs or [], "openai_chat_completions"))
    )
    files[TOOL_DEFS_BY_PROTOCOL_PATH] = json.dumps(
        {
            protocol: serializable(serialize_tool_defs(tool_defs or [], protocol))
            for protocol in (
                "vf",
                "openai_chat_completions",
                "openai_responses",
                "anthropic_messages",
            )
        }
    )
    files[RUNNER_PATH] = runner_source()
    runner_config: dict[str, object] = {"max_turns": max_turns}
    # Forward an opaque, harness-owned compaction blob (e.g. summarize mode +
    # checkpoint/framing prompts) so the in-sandbox base loop can emit a recorded
    # summary turn. Absent for prune mode -> RUNNER_CONFIG is byte-identical.
    compaction = state.runtime_state().get("compaction")
    if isinstance(compaction, dict) and compaction:
        runner_config["compaction"] = compaction
    files[RUNNER_CONFIG_PATH] = json.dumps(runner_config)
    command = python_runtime_command(
        RUNNER_PATH,
        *([mode] if fn_ref is None else [mode, fn_ref]),
    )
    package_setup = [] if package is None else [package.install_command]
    return {
        **dict(program),
        "files": files,
        "command": command,
        "env": program_option_mapping(
            cast(ProgramMappingInput, program.get("env")), "program.env"
        ),
        "setup": [
            python_runtime_setup_command(),
            *package_setup,
            *program_list_items(
                cast(ProgramListInput, program.get("setup")), "program.setup"
            ),
        ],
        VF_STATE_INPUT_PATH_KEY: STATE_INPUT_PATH,
    }


@dataclass(frozen=True)
class SandboxPackage:
    local_root: Path
    remote_root: str = PACKAGE_ROOT

    @property
    def install_command(self) -> str:
        return python_package_install_command(shlex.quote(self.remote_root))


def sandbox_program_package(*, mode: str, fn_ref: str | None) -> SandboxPackage | None:
    if mode != "fn" or fn_ref is None:
        return None
    module_name, _, _ = fn_ref.partition(":")
    if not module_name:
        raise ValueError("program.fn must include a module path.")
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Cannot resolve program.fn module {module_name!r}.")
    roots = package_roots_for_module(module_name, spec)
    if not roots:
        return None
    if len(roots) != 1:
        raise ValueError(
            f"program.fn {fn_ref!r} resolved to multiple package roots: "
            f"{sorted(str(root) for root in roots)}."
        )
    return SandboxPackage(local_root=next(iter(roots)))


def sandbox_program_with_package(
    program: ConfigData, package: SandboxPackage
) -> ConfigData:
    merged = dict(program)
    dirs = program_option_mapping(
        cast(ProgramMappingInput, merged.get("dirs")), "program.dirs"
    )
    if package.remote_root in dirs:
        raise ValueError(
            f"program.dirs already defines internal package path {package.remote_root!r}."
        )
    dirs[package.remote_root] = str(package.local_root)
    merged["dirs"] = dirs
    return merged


def package_roots_for_module(
    module_name: str, spec: importlib.machinery.ModuleSpec
) -> set[Path]:
    roots = set()
    for path in module_source_paths(spec):
        if is_external_import_path(path):
            continue
        root = module_package_root(path)
        if root is None:
            raise ValueError(
                f"Sandboxed program.fn {module_name!r} resolves to local source "
                f"{path}, but no pyproject.toml was found beside the resolved "
                "environment module or package."
            )
        roots.add(root)
    return roots


def module_source_paths(spec: importlib.machinery.ModuleSpec) -> list[Path]:
    origin = spec.origin
    if spec.submodule_search_locations:
        return [Path(path).resolve() for path in spec.submodule_search_locations]
    if origin in {None, "built-in", "frozen"}:
        return []
    return [Path(origin).resolve()]


def module_package_root(path: Path) -> Path | None:
    root = path if path.is_dir() else path.parent
    if (root / "pyproject.toml").is_file():
        return root
    return None


def is_external_import_path(path: Path) -> bool:
    parts = set(path.parts)
    if "site-packages" in parts or "dist-packages" in parts:
        return True
    for prefix in interpreter_prefixes():
        try:
            path.relative_to(prefix)
        except ValueError:
            continue
        return True
    return False


def interpreter_prefixes() -> list[Path]:
    prefixes: list[Path] = []
    for key in ("stdlib", "platstdlib"):
        value = sysconfig.get_path(key)
        if value:
            prefixes.append(Path(value).resolve())
    for value in (sys.base_prefix, sys.base_exec_prefix):
        if value:
            prefixes.append(Path(value).resolve())
    unique: list[Path] = []
    for prefix in prefixes:
        if prefix not in unique:
            unique.append(prefix)
    return unique


def runner_source() -> str:
    return r"""
import asyncio
import importlib
import inspect
import json
import os
import sys

import requests
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

TASK_PATH = "/tmp/vf_task.json"
STATE_INPUT_PATH = "/tmp/vf_state_in.json"
STATE_OUTPUT_PATH = "/tmp/vf_state_out.json"
RUNNER_CONFIG_PATH = "/tmp/vf_runner_config.json"
TOOL_DEFS_PATH = "/tmp/vf_tool_defs.json"
TOOL_DEFS_BY_PROTOCOL_PATH = "/tmp/vf_tool_defs_by_protocol.json"


class Client:
    def __init__(self, state):
        self.openai = AsyncOpenAI(
            api_key=endpoint_token(),
            base_url=os.environ.get("OPENAI_BASE_URL")
            or state["endpoint_base_url"],
        )
        self.anthropic = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
            or endpoint_token(),
            base_url=os.environ.get("ANTHROPIC_BASE_URL")
            or state["endpoint_root_url"],
        )
        self.chat = self.openai.chat
        self.responses = self.openai.responses
        self.messages = self.anthropic.messages

    async def close(self):
        await self.openai.close()
        await self.anthropic.close()


def endpoint_token():
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "intercepted"


def endpoint_headers():
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "python-requests/2.32.3",
        "Authorization": f"Bearer {endpoint_token()}",
    }


def vf_url(state, path):
    return f"{state['endpoint_root_url'].rstrip('/')}/vf/{path}"


CONTROL_ENDPOINT_TIMEOUT = 300.0
# Sandbox-local tools POST to a loopback service in this sandbox; allow up to the
# service's own per-call bound (600s) plus buffer for slow browser ops.
SANDBOX_LOCAL_TOOL_TIMEOUT = 660.0


def post_json(url, payload, headers=None, timeout=CONTROL_ENDPOINT_TIMEOUT):
    response = requests.post(
        url,
        json=payload,
        headers=headers or endpoint_headers(),
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc
    if not response.content:
        return {}
    return response.json()


async def vf_post(state, path, payload, timeout=CONTROL_ENDPOINT_TIMEOUT):
    return await asyncio.to_thread(
        post_json, vf_url(state, path), payload, endpoint_headers(), timeout
    )


async def call_tool(state, name, arguments):
    payload = await vf_post(state, f"tools/{name}", {"arguments": arguments})
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


async def call_sandbox_local_tool(state, name, arguments, endpoint):
    # In-sandbox dispatch: POST straight to the tool's loopback service in this
    # sandbox instead of the host /vf/tools tunnel + a per-call sandbox.execute.
    # Removes the control-plane round-trip and fresh-process import from each call.
    host = endpoint.get("host") or "127.0.0.1"
    port = int(endpoint["port"])
    path = endpoint.get("path") or "/"
    url = f"http://{host}:{port}{path}"
    payload = await asyncio.to_thread(
        post_json,
        url,
        {"tool_name": name, "args": arguments},
        {"Content-Type": "application/json"},
        SANDBOX_LOCAL_TOOL_TIMEOUT,
    )
    if isinstance(payload, dict) and payload.get("ok") is False:
        raise RuntimeError(
            f"{payload.get('error_type')}: {payload.get('error_message')}"
        )
    # Apply rollout-state events the tool returned (verifier evidence); they ride
    # home in the runner's state patch.
    appends = payload.get("state_appends") if isinstance(payload, dict) else None
    if isinstance(appends, dict):
        for key, value in appends.items():
            bucket = state.get(key)
            if not isinstance(bucket, list):
                bucket = []
                state[key] = bucket
            bucket.append(value)
    return payload.get("content") if isinstance(payload, dict) else payload


async def call_user(state, transcript):
    payload = await vf_post(state, "user", {"transcript": transcript})
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload.get("messages") or []


def set_stop_condition(state, value, *, overwrite=False):
    if overwrite or state.get("stop_condition") is None:
        state["stop_condition"] = value


async def check_stop(state):
    payload = await vf_post(state, "stop", {})
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    if payload.get("done"):
        if payload.get("stop_condition"):
            set_stop_condition(state, payload["stop_condition"])
        return True
    return False


async def maybe_call(fn, **objects):
    sig = inspect.signature(fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        result = fn(**objects)
    else:
        result = fn(**{key: value for key, value in objects.items() if key in sig.parameters})
    if inspect.isawaitable(result):
        return await result
    return result


def import_ref(ref):
    module_name, _, attr_path = ref.partition(":")
    obj = importlib.import_module(module_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def tool_call_name(tool_call):
    # The /vf/model bridge returns canonical vf tool calls ({id, name, arguments}).
    return tool_call["name"]


def tool_call_arguments(tool_call):
    raw = tool_call.get("arguments") or "{}"
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            snippet = raw if len(raw) <= 500 else f"{raw[:500]}..."
            raise ValueError(
                "Invalid JSON tool-call arguments: "
                f"{exc.msg} at line {exc.lineno} column {exc.colno}; "
                f"raw={snippet!r}"
            ) from exc
    return raw


def tool_error_content(error):
    return str(error)


def is_tool_content_parts(value):
    # Mirror is_valid_tool_content_parts (can't import into the lean runner):
    # pass {"type": text|image_url} part lists (e.g. screenshots) through, not str().
    if not isinstance(value, list):
        return False
    return all(
        isinstance(part, dict) and part.get("type") in ("text", "image_url")
        for part in value
    )


def load_tool_defs(protocol):
    defs = json.loads(open(TOOL_DEFS_BY_PROTOCOL_PATH).read())
    return defs.get(protocol) or []


class ToolProxy:
    def __init__(self, state, name, description=None, sandbox_endpoint=None):
        self.state = state
        self.name = name
        self.__name__ = name
        self.__doc__ = description or ""
        self.sandbox_endpoint = sandbox_endpoint

    async def __call__(self, **arguments):
        if self.sandbox_endpoint:
            return await call_sandbox_local_tool(
                self.state, self.name, arguments, self.sandbox_endpoint
            )
        return await call_tool(self.state, self.name, arguments)


def load_tools(state):
    return {
        tool["name"]: ToolProxy(
            state,
            tool["name"],
            tool.get("description"),
            tool.get("sandbox_endpoint"),
        )
        for tool in load_tool_defs("vf")
    }


async def create_model_message(state, messages):
    # The sandbox sends canonical Messages over the /vf/model bridge; the host
    # resolves the bound client, tokenizes, and records the trajectory step, then
    # returns the assistant message. The sandbox never formats a provider payload.
    # Returns (message, should_compact): ``should_compact`` is the host-computed
    # summarize trigger (False when not in summarize mode).
    payload = await vf_post(state, "model", {"messages": messages}, timeout=None)
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    return payload["message"], bool(payload.get("should_compact"))


def _message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


async def run_base(task, state):
    system_messages = list(state.get("system_prompt") or [])
    prompt_messages = [*system_messages, *(state.get("prompt") or [])]
    messages = list(prompt_messages)
    config = json.loads(open(RUNNER_CONFIG_PATH).read())
    max_turns = int(config["max_turns"])
    # Compaction config forwarded by the harness (summarize mode only). Absent ->
    # prune mode, where the host compacts the prompt copy in prepare_prompt and the
    # loop does nothing extra.
    compaction = config.get("compaction") or {}
    summarize = compaction.get("mode") == "summarize"
    checkpoint_prompt = compaction.get("checkpoint_prompt") or ""
    framing = compaction.get("framing") or ""
    # Tools tagged with a sandbox_endpoint dispatch in-sandbox (loopback) instead
    # of the host /vf/tools tunnel.
    tool_endpoints = {
        d["name"]: d.get("sandbox_endpoint") for d in load_tool_defs("vf")
    }
    turn = 0
    # Index in ``messages`` where the current branch's completion begins. Equals
    # len(prompt_messages) normally; reset after a summarize rebuild so completion
    # accounting tracks the last branch (matching completion_from_trajectory).
    branch_prefix_len = len(prompt_messages)
    # One-turn cooldown after a rebuild: the trigger uses the PREVIOUS turn's
    # recorded prompt-token count, and the summary turn's recorded prompt is the
    # full (large) pre-compaction history. Without this, the first post-rebuild
    # turn would re-trigger off that large summary prompt and compact again
    # immediately. Skip exactly one should_compact after compacting.
    suppress_compaction = False
    while max_turns <= 0 or turn < max_turns:
        if await check_stop(state):
            break
        message, should_compact = await create_model_message(state, messages)
        turn += 1
        messages.append(message)
        if summarize and should_compact and not suppress_compaction:
            # Learnable compaction: the policy writes a handoff summary as a real
            # recorded /vf/model turn (trainable action), then we rebuild the branch
            # around it. The new prefix starts a fresh training sample downstream;
            # prime-rl broadcasts the rollout return across samples (cross-sample
            # credit) so this summary turn is credited with what follows.
            messages.append({"role": "user", "content": checkpoint_prompt})
            summary_message, _ = await create_model_message(state, messages)
            turn += 1
            messages[:] = [
                *system_messages,
                {
                    "role": "user",
                    "content": framing + "\n\n" + _message_text(summary_message),
                },
            ]
            branch_prefix_len = len(messages)
            suppress_compaction = True
            continue
        suppress_compaction = False
        tool_calls = list(message.get("tool_calls") or [])
        if not tool_calls:
            user_messages = await call_user(state, messages)
            if user_messages:
                messages.extend(user_messages)
                continue
            set_stop_condition(state, "no_tools")
            break
        for tool_call in tool_calls:
            try:
                name = tool_call_name(tool_call)
                arguments = tool_call_arguments(tool_call)
                endpoint = tool_endpoints.get(name)
                if endpoint:
                    result = await call_sandbox_local_tool(
                        state, name, arguments, endpoint
                    )
                else:
                    result = await call_tool(state, name, arguments)
                content = result if is_tool_content_parts(result) else str(result)
            except Exception as exc:
                content = tool_error_content(exc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": content,
                }
            )
            if await check_stop(state):
                completed = True
                break
        else:
            completed = False
        if completed:
            break
    state["completion"] = messages[branch_prefix_len:]
    set_stop_condition(state, "max_turns_reached")
    return state


async def main():
    mode = sys.argv[1]
    task = json.loads(open(TASK_PATH).read())
    state = json.loads(open(STATE_INPUT_PATH).read())
    original_state = json.loads(json.dumps(state))
    if mode == "base":
        # Base loop talks to the host over /vf/model; no provider SDK client.
        result = await run_base(task, state)
    elif mode == "fn":
        # fn-mode authors call the model via the OpenAI/Anthropic SDK, which the
        # interception server transparently handles, so they still need a client.
        client = Client(state)
        try:
            result = await maybe_call(
                import_ref(sys.argv[2]),
                task=task,
                state=state,
                client=client,
                tools=load_tools(state),
                tool_defs=load_tool_defs("vf"),
            )
        finally:
            await client.close()
    else:
        raise ValueError(f"Unknown sandbox program mode: {mode}")
    if result is not None:
        if not isinstance(result, dict):
            raise TypeError("Sandbox Python program must return None or a mapping.")
        state.update(result)
    patch = {
        key: value
        for key, value in state.items()
        if key not in original_state or original_state[key] != value
    }
    with open(STATE_OUTPUT_PATH, "w") as f:
        json.dump(patch, f)


asyncio.run(main())
"""
