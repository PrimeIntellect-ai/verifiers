"""
Recursive Language Model (RLM) Environment.

Implements the RLM inference strategy where language models can decompose and
recursively interact with input data of unbounded length through REPL environments.

Based on: https://www.alexzhang.dev/blog/recursive-language-models

Architecture:
- REPL loop runs in the framework (MultiTurnEnv pattern)
- Code execution runs in a sandbox via a persistent worker
- Sub-LLM calls from worker code are intercepted via HTTP proxy

Key features:
- Works with any dataset that has a normal prompt
- Optional input data can be provided via info["context_dir"] (directory) or
  legacy info["context"] (builtin data written to a file)
- Root model only sees query, not full input data (unless it peeks via code)
- Model can make recursive sub-LLM calls via llm_batch() function
- Final answer returned via answer variable
"""

import asyncio
import base64
import contextvars
import json
import logging
import os
import pickle
import random
import re
import shutil
import shlex
import sys
import tarfile
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, cast, Literal

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from aiohttp import web
from openai.types.chat import ChatCompletionFunctionToolParam
from prime_sandboxes import SandboxClient
from prime_sandboxes.core import APIClient
from prime_tunnel import Tunnel
import verifiers as vf
from verifiers.types import (
    ChatMessage,
    ChatMessages,
    Messages,
    ModelResponse,
    State,
    TrajectoryStep,
)
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.data_utils import extract_boxed_answer
from verifiers.utils.message_utils import concat_messages
from verifiers.utils.response_utils import (
    parse_is_truncated,
    parse_response_messages,
    parse_response_tokens,
)
from verifiers.utils.tool_utils import convert_func_to_oai_tool
from verifiers.envs.experimental.sandbox_mixin import SandboxMixin
from verifiers.envs.sandbox_env import CreateSandboxRequest
from prime_sandboxes import CommandTimeoutError

logger = logging.getLogger(__name__)

_FIXED_REPL_TOOL_NAMES = frozenset({"llm_batch"})


def _tool_display_name(tool: Callable) -> str:
    return getattr(tool, "__name__", tool.__class__.__name__)


def _dedupe_tools(
    tools: list[Callable],
    *,
    context: str,
    reserved_names: set[str] | None = None,
) -> tuple[list[Callable], dict[str, Callable]]:
    deduped: list[Callable] = []
    seen: dict[str, Callable] = {}
    for tool in tools:
        name = _tool_display_name(tool)
        if reserved_names and name in reserved_names:
            raise ValueError(f"Tool '{name}' is reserved and cannot be overridden.")
        if name in seen:
            if seen[name] is not tool:
                raise ValueError(
                    f"Tool name collision in {context}: '{name}' is defined by both "
                    f"{seen[name]!r} and {tool!r}. Rename or remove one."
                )
            continue
        seen[name] = tool
        deduped.append(tool)
    return deduped, seen


def _merge_tool_lists(
    *,
    fixed_tools: list[Callable],
    shared_tools: list[Callable],
    role_tools: list[Callable],
    context: str,
    reserved_names: set[str],
) -> tuple[list[Callable], dict[str, Callable]]:
    fixed, fixed_map = _dedupe_tools(
        fixed_tools,
        context=f"{context} fixed tools",
        reserved_names=set(),
    )
    merged = list(fixed)
    deduped_shared, _ = _dedupe_tools(
        shared_tools,
        context=f"{context} shared tools",
        reserved_names=reserved_names,
    )
    merged.extend(deduped_shared)
    deduped_role, _ = _dedupe_tools(
        role_tools,
        context=f"{context} tools",
        reserved_names=reserved_names,
    )
    merged.extend(deduped_role)
    deduped_all, deduped_map = _dedupe_tools(
        merged,
        context=context,
        reserved_names=set(),
    )
    return deduped_all, deduped_map


class SubLLMEmptyModelResponseError(vf.EmptyModelResponseError):
    """Raised when a sub-LLM call returns an empty model response."""


class RLMCodeExecutionTimeout(vf.ToolCallError):
    """Raised when code execution exceeds the configured timeout."""


class RLMWorkerError(vf.SandboxError):
    """Raised when the RLM worker is not running, crashes, or fails to start."""


class RLMWorkerRecoveryError(RLMWorkerError):
    """Raised when the RLM worker cannot be restarted after a failure."""


class RLMSessionError(vf.SandboxError):
    """Raised when the RLM session or sandbox is not initialized."""


class RLMSetupError(vf.SandboxError):
    """Raised when RLM environment setup fails (package install, setup hook, etc.)."""


@dataclass(frozen=True)
class RLMWorkerPaths:
    base_dir: str
    command_fifo: str
    response_fifo: str
    ready_flag: str
    worker_path: str
    worker_pid_file: str
    context_file: str
    answer_file: str
    log_file: str

    def to_dict(self) -> dict[str, str]:
        return {
            "base_dir": self.base_dir,
            "command_fifo": self.command_fifo,
            "response_fifo": self.response_fifo,
            "ready_flag": self.ready_flag,
            "worker_path": self.worker_path,
            "worker_pid_file": self.worker_pid_file,
            "context_file": self.context_file,
            "answer_file": self.answer_file,
            "log_file": self.log_file,
        }


@dataclass(frozen=True)
class RLMExecResult:
    stdout: str
    stderr: str | None = None
    exit_code: int | None = None


@dataclass
class SandboxRLMReplSession:
    rollout_id: str
    local_rollout_dir: str
    local_fs_root: str
    local_control_dir: str
    sandbox_id: str | None = None
    sandbox_fs_root: str | None = None
    sandbox_control_dir: str | None = None
    paths: RLMWorkerPaths | None = None


def _extract_tokens_from_response(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if not usage and isinstance(response, dict):
        usage = response.get("usage")
    if not usage:
        return 0, 0
    if isinstance(usage, dict):
        return (
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
        )
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _ensure_rlm_metric_state(state: State) -> None:
    state.setdefault("sub_llm_call_count", 0)
    state.setdefault("sub_llm_total_turns", 0)
    state.setdefault("sub_llm_prompt_tokens", 0)
    state.setdefault("sub_llm_completion_tokens", 0)
    state.setdefault("sub_llm_total_tool_calls", 0)
    state.setdefault("sub_llm_batch_count", 0)
    state.setdefault("sub_llm_max_batch_size", 0)
    state.setdefault("sub_llm_mean_batch_size", 0.0)

    state.setdefault("main_rlm_turns", 0)
    state.setdefault("main_rlm_prompt_tokens", 0)
    state.setdefault("main_rlm_completion_tokens", 0)

    state.setdefault("repl_total_time_seconds", 0.0)
    state.setdefault("repl_call_count", 0)
    state.setdefault("repl_mean_time_seconds", 0.0)
    state.setdefault("root_tool_call_count", 0)
    state.setdefault("root_tool_calls", {})

    state.setdefault("_rlm_sub_llm_call_ids", {})
    state.setdefault("_rlm_sub_llm_batch_counts", {})


def _update_rlm_repl_metrics(state: State, execution_seconds: float) -> None:
    _ensure_rlm_metric_state(state)
    state["repl_total_time_seconds"] += execution_seconds
    state["repl_call_count"] += 1
    if state["repl_call_count"] > 0:
        state["repl_mean_time_seconds"] = (
            state["repl_total_time_seconds"] / state["repl_call_count"]
        )


def update_rlm_metrics_from_step(state: State, step: TrajectoryStep) -> None:
    _ensure_rlm_metric_state(state)
    extras = step.get("extras", {}) or {}
    is_sub_llm = bool(extras.get("is_sub_llm_call"))

    prompt_tokens, completion_tokens = _extract_tokens_from_response(
        step.get("response")
    )

    if is_sub_llm:
        state["sub_llm_total_turns"] += 1
        state["sub_llm_prompt_tokens"] += prompt_tokens
        state["sub_llm_completion_tokens"] += completion_tokens
        state["sub_llm_total_tool_calls"] += int(extras.get("tool_call_count", 0) or 0)

        batch_id = extras.get("batch_id")
        request_id = extras.get("request_id")
        call_ids: dict[str, bool] = state.get("_rlm_sub_llm_call_ids", {})
        batch_counts: dict[str, int] = state.get("_rlm_sub_llm_batch_counts", {})

        if batch_id:
            request_id_norm = request_id if request_id not in (None, "") else "_missing"
            key = f"{batch_id}:{request_id_norm}"
            if key not in call_ids:
                call_ids[key] = True
                state["sub_llm_call_count"] += 1
                batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1
        else:
            # Fallback: treat each turn as its own call if identifiers are missing.
            state["sub_llm_call_count"] += 1

        state["_rlm_sub_llm_call_ids"] = call_ids
        state["_rlm_sub_llm_batch_counts"] = batch_counts

        if batch_counts:
            batch_sizes = list(batch_counts.values())
            state["sub_llm_batch_count"] = len(batch_sizes)
            state["sub_llm_max_batch_size"] = max(batch_sizes)
            state["sub_llm_mean_batch_size"] = sum(batch_sizes) / len(batch_sizes)
        else:
            state["sub_llm_batch_count"] = 0
            state["sub_llm_max_batch_size"] = 0
            state["sub_llm_mean_batch_size"] = 0.0
    else:
        state["main_rlm_turns"] += 1
        state["main_rlm_prompt_tokens"] += prompt_tokens
        state["main_rlm_completion_tokens"] += completion_tokens


def _update_root_tool_metrics(state: State, tool_name: str) -> None:
    _ensure_rlm_metric_state(state)
    state["root_tool_call_count"] += 1
    tool_calls: dict[str, int] = state.get("root_tool_calls", {})
    tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
    state["root_tool_calls"] = tool_calls


class RLMMonitorRubric(vf.Rubric):
    _SIMPLE_METRICS = [
        "sub_llm_call_count",
        "sub_llm_total_turns",
        "sub_llm_prompt_tokens",
        "sub_llm_completion_tokens",
        "sub_llm_total_tool_calls",
        "sub_llm_batch_count",
        "sub_llm_max_batch_size",
        "sub_llm_mean_batch_size",
        "main_rlm_turns",
        "main_rlm_prompt_tokens",
        "main_rlm_completion_tokens",
        "repl_total_time_seconds",
        "repl_call_count",
        "repl_mean_time_seconds",
        "root_tool_call_count",
    ]

    def __init__(self, root_tool_names: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        for metric_name in self._SIMPLE_METRICS:
            metric_fn = self._make_state_metric(metric_name)
            setattr(self, metric_name, metric_fn)
            self.add_metric(metric_fn)
        for tool_name in root_tool_names or []:
            self.add_metric(self._make_root_tool_metric(tool_name))

    def _make_state_metric(self, key: str):
        async def metric(state: State):
            value = state.get(key, 0)
            return 0 if value is None else value

        metric.__name__ = key
        return metric

    def _make_root_tool_metric(self, tool_name: str):
        async def root_tool_metric(state: State) -> int:
            tool_calls: dict[str, int] = state.get("root_tool_calls", {})
            return int(tool_calls.get(tool_name, 0))

        root_tool_metric.__name__ = f"{tool_name}_root_calls"
        return root_tool_metric


class SubLLMTurn(TypedDict):
    """A single turn in a sub-LLM call (used by RLMEnv)."""

    prompt_messages: ChatMessages  # Messages before this LLM call
    response: ModelResponse  # Full response object (with token_ids, logprobs)
    tool_call_count: int  # Number of tool calls made in this turn


class SubLLMResult(TypedDict):
    """Result of a sub-LLM call, possibly with multiple turns (used by RLMEnv)."""

    final_content: str
    turns: list[SubLLMTurn]
    total_prompt_tokens: int
    total_completion_tokens: int
    tool_call_count: int
    num_turns: int
    max_turns_reached: bool


# Worker script handles code execution; REPL loop is managed by the framework.
_SUB_LLM_CONFIG_BLOCK = (
    textwrap.dedent(
        """
    SUB_LLM_TIMEOUT = int(os.environ.get("RLM_SUB_LLM_TIMEOUT", "300"))
    """
    )
    .strip("\n")
    .splitlines()
)

_ENSURE_FIFO_BLOCK = [
    "def ensure_fifo(path: str) -> None:",
    "    if os.path.exists(path):",
    "        os.remove(path)",
    "    os.mkfifo(path)",
    "",
    "for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):",
    "    ensure_fifo(fifo_path)",
]


def _build_python_worker_script_template() -> str:
    dict_open = "{"
    dict_close = "}"
    answer_default = f'{dict_open}"ready": False, "content": ""{dict_close}'
    fs_context_block = [
        "fs_root = None",
        "if Path(CONTEXT_FILE).exists():",
        '    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:',
        "        context = json.load(f)",
        '        fs_root = context.get("fs_root")',
    ]
    lines: list[str] = [
        "",
        "import ast",
        "import base64",
        "import contextlib",
        "import io",
        "import json",
        "import os",
        "import pickle",
        "import sys",
    ]

    lines.extend(
        ["import traceback", "from pathlib import Path", "import requests", ""]
    )

    lines.extend(
        [
            'COMMAND_FIFO = "{command_fifo}"',
            'RESPONSE_FIFO = "{response_fifo}"',
            'READY_FLAG = "{ready_flag}"',
            'CONTEXT_FILE = "{context_file}"',
            'ANSWER_FILE = "{answer_file}"',
            "",
        ]
    )

    lines.extend(_SUB_LLM_CONFIG_BLOCK)
    lines.append("")

    lines.extend(_ENSURE_FIFO_BLOCK)
    lines.append("")
    lines.extend(fs_context_block)
    lines.append("")
    lines.extend(["if fs_root:", "    os.chdir(fs_root)"])
    lines.append("")
    lines.append(f"answer = {answer_default}")
    lines.extend(
        [
            "if Path(ANSWER_FILE).exists():",
            '    with open(ANSWER_FILE, "r", encoding="utf-8") as f:',
            "        answer = json.load(f)",
            "",
            'ROOT_TOOL_URL = os.environ.get("RLM_ROOT_TOOL_URL", "")',
            'ROOT_TOOL_SERIALIZATION = os.environ.get("RLM_ROOT_TOOL_SERIALIZATION", "pickle")',
            'ROOT_TOOL_NAMES_RAW = os.environ.get("RLM_ROOT_TOOL_NAMES", "[]")',
            "try:",
            "    ROOT_TOOL_NAMES = json.loads(ROOT_TOOL_NAMES_RAW)",
            "except Exception:",
            "    ROOT_TOOL_NAMES = []",
            "",
            "def _call_root_tool(tool_name: str, args: tuple, kwargs: dict):",
            "    if not ROOT_TOOL_URL:",
            '        raise RuntimeError("Root tool URL not configured")',
            '    if ROOT_TOOL_SERIALIZATION != "pickle":',
            '        raise RuntimeError("Only pickle serialization is supported")',
            "",
            '    args_payload = base64.b64encode(pickle.dumps(args)).decode("ascii")',
            '    kwargs_payload = base64.b64encode(pickle.dumps(kwargs)).decode("ascii")',
            f"    payload = {dict_open}",
            '        "tool_name": tool_name,',
            '        "serialization": "pickle",',
            '        "args": args_payload,',
            '        "kwargs": kwargs_payload,',
            f"    {dict_close}",
            "",
            "    resp = requests.post(",
            "        ROOT_TOOL_URL,",
            "        json=payload,",
            "        timeout=SUB_LLM_TIMEOUT,",
            "    )",
            "    resp.raise_for_status()",
            "    data = resp.json()",
            '    if data.get("print_lines"):',
            '        for line in data["print_lines"]:',
            "            print(line)",
            '    if data.get("error"):',
            '        raise RuntimeError(data["error"])',
            '    return pickle.loads(base64.b64decode(data.get("result", "")))',
            "",
            "def _make_root_tool(name: str):",
            "    def _tool(*args, **kwargs):",
            "        return _call_root_tool(name, args, kwargs)",
            "",
            "    _tool.__name__ = name",
            "    return _tool",
            "",
        ]
    )

    lines.append("extra_data = fs_root")
    lines.append("")
    lines.append(f"namespace: dict[str, object] = {dict_open}")
    lines.extend(
        [
            '    "__name__": "__main__",',
        ]
    )
    lines.extend(
        [
            '    "extra_data": extra_data,',
            '    "answer": answer,',
            f"{dict_close}",
            "for tool_name in ROOT_TOOL_NAMES:",
            "    namespace[tool_name] = _make_root_tool(tool_name)",
            "",
        ]
    )
    lines.append('Path(READY_FLAG).write_text("ready", encoding="utf-8")')
    lines.extend(
        [
            "",
            "execution_count = 0",
            "",
            "while True:",
            '    with open(COMMAND_FIFO, "r", encoding="utf-8") as command_file:',
            "        payload = command_file.read()",
            "    if not payload:",
            "        continue",
            "    request = json.loads(payload)",
            '    if request.get("shutdown"):',
            "        break",
            "",
            '    code = request.get("code", "")',
        ]
    )
    lines.append('    seq = request.get("seq", 0)')
    lines.extend(
        [
            "    execution_count += 1",
            "",
            f"    result = {dict_open}",
            '        "status": "ok",',
            '        "stdout": "",',
            '        "stderr": "",',
            '        "result": None,',
            '        "execution_count": execution_count,',
        ]
    )
    lines.append('        "seq": seq,')
    lines.extend(
        [
            f'        "answer": namespace.get("answer", {answer_default}),',
            f"    {dict_close}",
            "",
            "    stdout_buffer = io.StringIO()",
            "    stderr_buffer = io.StringIO()",
            "",
            "    try:",
            "        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):",
            '            module_ast = ast.parse(code, mode="exec")',
            "            body = list(module_ast.body)",
            "            trailing_expr = None",
            "            if body and isinstance(body[-1], ast.Expr):",
            "                trailing_expr = body.pop()",
            "            if body:",
            "                exec_module = ast.Module(body=body, type_ignores=[])",
            '                exec(compile(exec_module, "<cell>", "exec"), namespace, namespace)',
            "            if trailing_expr is not None:",
            "                value = eval(",
            '                    compile(ast.Expression(trailing_expr.value), "<cell>", "eval"),',
            "                    namespace,",
            "                    namespace,",
            "                )",
            "                if value is not None:",
            '                    result["result"] = repr(value)',
            "    except Exception:",
            '        result["status"] = "error"',
            '        result["result"] = traceback.format_exc()',
            "",
            '    result["stdout"] = stdout_buffer.getvalue()',
            '    result["stderr"] = stderr_buffer.getvalue()',
            f'    result["answer"] = namespace.get("answer", {answer_default})',
            "",
        ]
    )
    lines.extend(
        [
            '    with open(ANSWER_FILE, "w", encoding="utf-8") as f:',
            '        json.dump(result["answer"], f)',
            "",
            '    with open(RESPONSE_FIFO, "w", encoding="utf-8") as response_file:',
            "        response_file.write(json.dumps(result))",
        ]
    )

    return "\n".join(lines) + "\n"


_RLM_BASH_TOOL_HELPER_SCRIPT = textwrap.dedent(
    """
    import base64
    import json
    import os
    import pickle
    import sys
    import urllib.error
    import urllib.request

    ROOT_TOOL_URL = os.environ.get("RLM_ROOT_TOOL_URL", "")
    ROOT_TOOL_SERIALIZATION = os.environ.get("RLM_ROOT_TOOL_SERIALIZATION", "pickle")
    ROOT_TOOL_USER_AGENT = os.environ.get(
        "RLM_ROOT_TOOL_USER_AGENT", "python-requests/2.32.3"
    )


    def _decode_arg(raw: str):
        try:
            return json.loads(raw)
        except Exception:
            return raw


    def _call_root_tool(tool_name: str, args: tuple, kwargs: dict):
        if not ROOT_TOOL_URL:
            raise RuntimeError("Root tool URL not configured")
        if ROOT_TOOL_SERIALIZATION != "pickle":
            raise RuntimeError("Only pickle serialization is supported")

        args_payload = base64.b64encode(pickle.dumps(args)).decode("ascii")
        kwargs_payload = base64.b64encode(pickle.dumps(kwargs)).decode("ascii")
        payload = {
            "tool_name": tool_name,
            "serialization": "pickle",
            "args": args_payload,
            "kwargs": kwargs_payload,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            ROOT_TOOL_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": ROOT_TOOL_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                error_body = str(e)
            raise RuntimeError(error_body) from e

        response_data = json.loads(resp_body)
        if response_data.get("error"):
            raise RuntimeError(response_data["error"])

        result_payload = response_data.get("result", "")
        result = pickle.loads(base64.b64decode(result_payload))
        print_lines = response_data.get("print_lines") or []
        return result, print_lines


    def _print_result(result):
        if isinstance(result, str):
            sys.stdout.write(result)
            if not result.endswith("\\n"):
                sys.stdout.write("\\n")
            return
        try:
            sys.stdout.write(json.dumps(result))
            sys.stdout.write("\\n")
        except Exception:
            sys.stdout.write(repr(result))
            sys.stdout.write("\\n")


    def _print_lines(lines):
        for line in lines:
            text = str(line)
            sys.stdout.write(text)
            if not text.endswith("\\n"):
                sys.stdout.write("\\n")


    def _split_batch_lines(lines):
        summary = []
        per_item = {}
        for line in lines:
            text = str(line).strip()
            if text.startswith("[") and "]:" in text:
                idx_text = text[1 : text.index("]")]
                if idx_text.isdigit():
                    per_item[int(idx_text)] = text[text.index("]:") + 2 :].strip()
                    continue
            summary.append(text)
        return summary, per_item


    def _print_llm_batch_result(result_list, per_item_meta=None):
        for index, item in enumerate(result_list):
            if index > 0:
                sys.stdout.write("\\n")
            header = f"----- llm_batch[{index}]"
            if per_item_meta and index in per_item_meta:
                header = f"{header} ({per_item_meta[index]})"
            sys.stdout.write(f"{header} -----\\n")
            if not isinstance(item, str):
                try:
                    item = json.dumps(item)
                except Exception:
                    item = repr(item)
            sys.stdout.write(item)
            if not item.endswith("\\n"):
                sys.stdout.write("\\n")


    def _load_json_payload(json_payload):
        raw = json_payload
        if raw is None and not sys.stdin.isatty():
            raw = sys.stdin.read().strip()
        if not raw:
            raise RuntimeError("Missing JSON payload.")
        try:
            return json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON payload: {exc}") from exc


    def _coerce_args_kwargs(data):
        if isinstance(data, dict):
            if "args" in data or "kwargs" in data:
                args = data.get("args", [])
                kwargs = data.get("kwargs", {})
            else:
                args = []
                kwargs = data
        elif isinstance(data, list):
            args = data
            kwargs = {}
        else:
            args = [data]
            kwargs = {}
        if not isinstance(args, (list, tuple)):
            raise RuntimeError("JSON args must be a list.")
        if not isinstance(kwargs, dict):
            raise RuntimeError("JSON kwargs must be an object.")
        return list(args), kwargs


    def main():
        argv = sys.argv[1:]
        tool_name = None
        use_lines = False
        use_json = False
        json_payload = None
        args = []
        while argv:
            token = argv.pop(0)
            if token == "--tool":
                if not argv:
                    raise RuntimeError("--tool requires a name")
                tool_name = argv.pop(0)
            elif token == "--lines":
                use_lines = True
            elif token == "--json":
                use_json = True
                if argv and not argv[0].startswith("--"):
                    json_payload = argv.pop(0)
            else:
                args.append(token)

        if not tool_name:
            raise RuntimeError("Tool name not provided")

        if tool_name == "llm_batch":
            def _coerce_prompts(data):
                if isinstance(data, dict):
                    if "prompts" in data:
                        return data["prompts"]
                    if "messages" in data:
                        return data["messages"]
                    if "prompt" in data:
                        return [data["prompt"]]
                    return [data]
                if isinstance(data, list):
                    return data
                if isinstance(data, str):
                    return [data]
                return [data]

            prompts = []
            if use_json:
                if args:
                    raise RuntimeError("llm_batch --json does not accept extra args.")
                data = _load_json_payload(json_payload)
                if isinstance(data, dict) and "prompts" in data:
                    prompts = data["prompts"]
                elif isinstance(data, dict) and ("args" in data or "kwargs" in data):
                    parsed_args, parsed_kwargs = _coerce_args_kwargs(data)
                    if "prompts" in parsed_kwargs:
                        prompts = parsed_kwargs["prompts"]
                    elif parsed_args:
                        prompts = parsed_args
                    else:
                        raise RuntimeError(
                            "llm_batch --json requires 'prompts' or non-empty 'args'."
                        )
                else:
                    prompts = _coerce_prompts(data)
            elif use_lines:
                prompts = sys.stdin.read().splitlines()
            elif args:
                if len(args) == 1:
                    raw_arg = args[0].strip()
                    if raw_arg.startswith("{") or raw_arg.startswith("["):
                        try:
                            data = json.loads(raw_arg)
                        except Exception:
                            prompts = [args[0]]
                        else:
                            prompts = _coerce_prompts(data)
                    else:
                        prompts = list(args)
                else:
                    prompts = list(args)
            elif not sys.stdin.isatty():
                raw = sys.stdin.read().strip()
                if raw:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        prompts = [raw]
                    else:
                        prompts = _coerce_prompts(data)
            result, print_lines = _call_root_tool(tool_name, (prompts,), {})
            summary_lines, per_item = _split_batch_lines(print_lines)
            if summary_lines:
                _print_lines(summary_lines)
            if isinstance(result, list):
                _print_llm_batch_result(result, per_item_meta=per_item)
            else:
                if print_lines:
                    _print_lines(print_lines)
                _print_result(result)
            return

        if use_json:
            if args:
                raise RuntimeError("--json does not accept extra args.")
            data = _load_json_payload(json_payload)
            parsed_args, parsed_kwargs = _coerce_args_kwargs(data)
            result, print_lines = _call_root_tool(
                tool_name, tuple(parsed_args), parsed_kwargs
            )
            if print_lines:
                _print_lines(print_lines)
            _print_result(result)
            return

        parsed_args = tuple(_decode_arg(arg) for arg in args)
        result, print_lines = _call_root_tool(tool_name, parsed_args, {})
        if print_lines:
            _print_lines(print_lines)
        _print_result(result)


    if __name__ == "__main__":
        try:
            main()
        except Exception as exc:
            sys.stderr.write(f"Error: {exc}\\n")
            sys.exit(1)
    """
)


_RLM_PY_WORKER_SCRIPT_TEMPLATE = _build_python_worker_script_template()


_RLM_BASH_WORKER_SCRIPT_TEMPLATE = textwrap.dedent(
    """
    import base64
    import json
    import os
    import subprocess
    import sys
    import uuid
    from pathlib import Path

    COMMAND_FIFO = "{command_fifo}"
    RESPONSE_FIFO = "{response_fifo}"
    READY_FLAG = "{ready_flag}"
    CONTEXT_FILE = "{context_file}"
    ANSWER_FILE = "{answer_file}"
    ROOT_TOOL_HELPER_SCRIPT = {root_tool_helper_script}
    STATE_FILE = os.path.join(os.path.dirname(CONTEXT_FILE), "rlm_env_state.json")

    def ensure_fifo(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        os.mkfifo(path)

    for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):
        ensure_fifo(fifo_path)

    fs_root = None
    if Path(CONTEXT_FILE).exists():
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            context = json.load(f)
            fs_root = context.get("fs_root")

    if fs_root:
        os.chdir(fs_root)

    helper_path = os.path.join(os.path.dirname(CONTEXT_FILE), "rlm_root_tool.py")
    Path(helper_path).write_text(ROOT_TOOL_HELPER_SCRIPT, encoding="utf-8")
    try:
        os.chmod(helper_path, 0o700)
    except Exception:
        pass

    ROOT_TOOL_NAMES_RAW = os.environ.get("RLM_ROOT_TOOL_NAMES", "[]")
    try:
        ROOT_TOOL_NAMES = json.loads(ROOT_TOOL_NAMES_RAW)
    except Exception:
        ROOT_TOOL_NAMES = []

    def _tool_defs():
        lines = []
        for tool_name in ROOT_TOOL_NAMES:
            lines.append(
                f'{tool_name}() {{ "$RLM_ROOT_TOOL_PYTHON" "$RLM_ROOT_TOOL_HELPER" --tool "{tool_name}" "$@"; }}'
            )
        return "\\n".join(lines)

    TOOL_DEF_SCRIPT = _tool_defs()

    def _load_state():
        if Path(STATE_FILE).exists():
            try:
                return json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "cwd": fs_root or os.getcwd(),
            "ready": False,
            "content": "",
        }

    def _save_state(state):
        Path(STATE_FILE).write_text(json.dumps(state), encoding="utf-8")

    def _parse_bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    state = _load_state()
    answer = {"ready": bool(state.get("ready")), "content": state.get("content", "")}

    Path(READY_FLAG).write_text("ready", encoding="utf-8")

    execution_count = 0

    while True:
        with open(COMMAND_FIFO, "r", encoding="utf-8") as command_file:
            payload = command_file.read()
        if not payload:
            continue
        request = json.loads(payload)
        if request.get("shutdown"):
            break

        code = request.get("code", "")
        seq = request.get("seq", 0)
        execution_count += 1

        marker = uuid.uuid4().hex
        end_marker = f"__RLM_END__{marker}__"
        env_marker = f"__RLM_ENV__{marker}__"
        pwd_marker = f"__RLM_PWD__{marker}__"

        bash_script = (
            f'cd "${{RLM_PWD:-$PWD}}"\\n'
            f'export RLM_READY="${{RLM_READY:-}}"\\n'
            f'export RLM_CONTENT="${{RLM_CONTENT-}}"\\n'
            f"{TOOL_DEF_SCRIPT}\\n"
            f"(\\n"
            f"trap '"
            f'__RLM_STATUS=$?; '
            f'printf "\\n{end_marker}__%s\\n" "$__RLM_STATUS"; '
            f'printf "{env_marker}__%s__" "${{RLM_READY:-}}"; '
            f'printf "%s" "${{RLM_CONTENT-}}" | base64 | tr -d "\\n"; '
            f'printf "\\n{pwd_marker}__%s\\n" "$PWD"'
            f"' EXIT\\n"
            f"{code}\\n"
            f")\\n"
        )

        env = os.environ.copy()
        env.update(
            {
                "RLM_READY": "1" if state.get("ready") else "",
                "RLM_CONTENT": state.get("content", ""),
                "RLM_PWD": state.get("cwd", ""),
                "RLM_ROOT_TOOL_HELPER": helper_path,
                "RLM_ROOT_TOOL_PYTHON": sys.executable,
            }
        )

        proc = subprocess.run(
            ["bash", "-lc", bash_script],
            text=True,
            capture_output=True,
            env=env,
            cwd=state.get("cwd") or None,
        )

        text = proc.stdout or ""
        output = text
        exit_code = None
        ready_val = ""
        content_val = ""
        cwd_val = state.get("cwd", "")

        end_idx = text.find(end_marker)
        if end_idx != -1:
            output = text[:end_idx]
            after_end = text[end_idx + len(end_marker):]
            if after_end.startswith("__"):
                after_end = after_end[2:]
                exit_str = after_end.split("\\n", 1)[0]
                try:
                    exit_code = int(exit_str.strip())
                except Exception:
                    exit_code = None

        env_idx = text.find(env_marker)
        if env_idx != -1:
            after_env = text[env_idx + len(env_marker):]
            if after_env.startswith("__"):
                after_env = after_env[2:]
                parts = after_env.split("__", 1)
                if len(parts) == 2:
                    ready_val = parts[0]
                    content_b64 = parts[1].split("\\n", 1)[0]
                    if content_b64:
                        try:
                            content_val = base64.b64decode(content_b64).decode(
                                "utf-8", errors="replace"
                            )
                        except Exception:
                            content_val = ""

        pwd_idx = text.find(pwd_marker)
        if pwd_idx != -1:
            after_pwd = text[pwd_idx + len(pwd_marker):]
            if after_pwd.startswith("__"):
                after_pwd = after_pwd[2:]
                cwd_val = after_pwd.split("\\n", 1)[0].strip() or cwd_val

        state["ready"] = _parse_bool(ready_val)
        state["content"] = content_val
        if cwd_val:
            state["cwd"] = cwd_val
        _save_state(state)

        answer = {"ready": bool(state.get("ready")), "content": state.get("content", "")}
        Path(ANSWER_FILE).write_text(json.dumps(answer), encoding="utf-8")

        result = {
            "status": "ok",
            "stdout": output,
            "stderr": proc.stderr or "",
            "result": None,
            "execution_count": execution_count,
            "seq": seq,
            "answer": answer,
        }

        with open(RESPONSE_FIFO, "w", encoding="utf-8") as response_file:
            response_file.write(json.dumps(result))
    """
)


def _build_worker_paths(base_dir: str) -> RLMWorkerPaths:
    base_dir = base_dir.rstrip("/") or base_dir
    return RLMWorkerPaths(
        base_dir=base_dir,
        command_fifo=os.path.join(base_dir, "rlm_cmd"),
        response_fifo=os.path.join(base_dir, "rlm_res"),
        ready_flag=os.path.join(base_dir, "rlm_ready"),
        worker_path=os.path.join(base_dir, "rlm_worker.py"),
        worker_pid_file=os.path.join(base_dir, "rlm_worker.pid"),
        context_file=os.path.join(base_dir, "rlm_context.json"),
        answer_file=os.path.join(base_dir, "rlm_answer.json"),
        log_file=os.path.join(base_dir, "rlm_worker.log"),
    )


def _render_worker_script(paths: RLMWorkerPaths, *, repl_language: str) -> str:
    if repl_language == "bash":
        script = _RLM_BASH_WORKER_SCRIPT_TEMPLATE
        script = script.replace("{command_fifo}", paths.command_fifo)
        script = script.replace("{response_fifo}", paths.response_fifo)
        script = script.replace("{ready_flag}", paths.ready_flag)
        script = script.replace("{context_file}", paths.context_file)
        script = script.replace("{answer_file}", paths.answer_file)
        script = script.replace(
            "{root_tool_helper_script}", repr(_RLM_BASH_TOOL_HELPER_SCRIPT)
        )
        return script
    script = _RLM_PY_WORKER_SCRIPT_TEMPLATE
    script = script.replace("{command_fifo}", paths.command_fifo)
    script = script.replace("{response_fifo}", paths.response_fifo)
    script = script.replace("{ready_flag}", paths.ready_flag)
    script = script.replace("{context_file}", paths.context_file)
    script = script.replace("{answer_file}", paths.answer_file)
    return script


# System prompt for sub-LLMs (called via llm_batch)
_SUB_LLM_SYSTEM_PROMPT_STORE = {
    "light": ("You have {num_turns} turns available to fulfill your task."),
    "medium": (
        "You will be given a task to perform."
        " Consider the tools at your disposal closely,"
        " and don't be afraid to think as much as you need about every step."
        "\n\nYou have {num_turns} turns available to fulfill your task."
        " You will be warned when there's only one turn left."
    ),
    "heavy": (
        "You will be given a task to perform."
        " Consider the tools at your disposal closely,"
        " and don't be afraid to think as much as you need about every step."
        "\n\nYou have {num_turns} turns available to fulfill your task."
        " Unless the task is trivial, use the turns to their fullest to make sure you get the answer right."
        " Plan well for how to fulfill the task within the turn limit, but don't be afraid to experiment;"
        " there's a tradeoff to be had and you should think very carefully about how to optimize it."
        " You will be warned when there's only one turn left."
    ),
}


# System prompt for RLM
_RLM_PYTHON_SYSTEM_PROMPT_STORE = {
    "light": """You have the `call_python_repl` tool and a filesystem available to you.

There exists an `answer` variable, which is a dict. `answer["content"]` must contain your answer. When the final answer is set, set `answer["ready"] = True`.
""",
    "medium": """You have the `call_python_repl` tool and a filesystem available to you.

There exists an `answer` variable, which is a dict. `answer["content"]` must contain your answer. When the final answer is set, set `answer["ready"] = True`.

This is an iterative environment. Make use of sub-LLMs via `llm_batch` whenever they could be useful; prefer calling them in parallel to calling them sequentially.
""",
    "heavy": """You are operating in a Recursive Language Model (RLM) environment - an iterative Python REPL where you explore data step by step.

A filesystem is available; explore it as needed.

## Critical: This is an ITERATIVE environment

You will write code, see its output, then write more code based on what you learned. **Do NOT try to solve everything in one tool call.** Each tool call executes and returns output before you continue.

Use the `call_python_repl` tool to execute Python code. The REPL maintains state across calls. See the tool description for available variables and functions.

## Workflow

**Step 1: Explore the filesystem**
```python
import os
print(os.getcwd())
print(os.listdir("."))
```
Wait for output. Now you know the actual format.

**Step 2: Process and build your answer**
```python
answer["content"] = "your current best answer"
```

**Step 3: Verify and finalize (only after reviewing output)**
```python
print(f"My answer: {answer['content']}")
answer["ready"] = True
```

## Important Rules

1. **NEVER set `answer["ready"] = True` until you have seen execution output** - you need feedback first
2. **One step at a time** - make small tool calls, see output, then continue
3. **Use `llm_batch()` for semantic tasks** - summarization, understanding text, classification, etc.
   Pass a list of strings only (no message dicts).
""",
}


_RLM_BASH_SYSTEM_PROMPT_STORE = {
    "light": """You have the `call_bash_repl` tool and a filesystem available to you.

In the end, the `RLM_CONTENT` environment variable must contain your answer. When the final answer is set, call `export RLM_READY=1`.
""",
    "medium": """You have the `call_bash_repl` tool and a filesystem available to you.

In the end, the `RLM_CONTENT` environment variable must contain your answer. When the final answer is set, call `export RLM_READY=1`.

This is an iterative environment. Make use of sub-LLMs via `llm_batch` whenever they could be useful; prefer calling them in parallel to calling them sequentially.
""",
    "heavy": """You are operating in a Recursive Language Model (RLM) environment - an iterative Bash REPL where you explore data step by step.

A filesystem is available; explore it as needed.

## Critical: This is an ITERATIVE environment

You will run shell commands, see their output, then run more commands based on what you learned. **Do NOT try to solve everything in one tool call.** Each tool call executes and returns output before you continue.

Use the `call_bash_repl` tool to execute Bash commands. The shell maintains state across calls. See the tool description for available variables and commands.

## Workflow

**Step 1: Explore the filesystem**
```bash
pwd
ls
```
Wait for output. Now you know the actual format.

**Step 2: Build your answer**
```bash
export RLM_CONTENT="your current best answer"
```

**Step 3: Verify and finalize (only after reviewing output)**
```bash
printf "My answer: %s\\n" "$RLM_CONTENT"
export RLM_READY=1
```

## Important Rules

1. **NEVER set `RLM_READY=1` until you have seen execution output** - you need feedback first
2. **One step at a time** - make small tool calls, see output, then continue
3. **Use `llm_batch` for semantic tasks** - summarization, understanding text, classification, etc.
   Pass a list of strings only (no message dicts).
4. **Tool usage in Bash**:
   - Call tools as shell commands with positional args (each arg is JSON-decoded if possible).
   - For structured args/kwargs, use `--json` with a payload like `{"args":[...],"kwargs":{...}}`
     (or provide the JSON via stdin).
   - `llm_batch` accepts `--json` with `{"prompts":[...]}`
""",
}


class RLMExecutor(SandboxMixin):
    def __init__(self, env: "RLMEnv") -> None:
        self.env = env
        self._sessions: dict[str, SandboxRLMReplSession] = {}
        self._retained_dirs: set[str] = set()
        self.init_sandbox_client(
            sandbox_client_max_workers=env.sandbox_client_max_workers,
            sandbox_client_max_connections=env.sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=env.sandbox_client_max_keepalive_connections,
        )

    def create_rollout_dirs(self, state: State) -> None:
        session = self._get_or_create_session(state)
        state["rlm_rollout_dir"] = session.local_rollout_dir
        state["rlm_fs_root"] = session.local_fs_root
        state["rlm_control_dir"] = session.local_control_dir
        state["rlm_paths"] = _build_worker_paths(session.local_control_dir).to_dict()

    async def prepare_filesystem(self, state: State) -> None:
        session = self._get_session(state)
        if session.sandbox_id is None:
            request = self._build_sandbox_request(state)
            state["sandbox_state"] = {
                "ready": False,
                "command_execution_times": [],
            }
            session.sandbox_id = await self.create_sandbox(state, request)
            state["sandbox_state"]["ready"] = True

        if not session.sandbox_id:
            raise RLMSessionError("Sandbox not initialized")

        sandbox_fs_root = state.get("rlm_fs_root_remote")
        sandbox_control_dir = state.get("rlm_control_dir_remote")
        if not sandbox_fs_root or not sandbox_control_dir:
            sandbox_root = f"/tmp/rlm_{session.rollout_id}"
            sandbox_fs_root = f"{sandbox_root}/rlm_fs"
            sandbox_control_dir = f"{sandbox_root}/rlm_control"

        mkdir_cmd = f"mkdir -p {sandbox_fs_root} {sandbox_control_dir}"
        await self._execute_sandbox_command(
            session.sandbox_id,
            f"bash -lc '{mkdir_cmd}'",
            timeout=self.env.max_startup_wait_seconds,
        )

        await self._upload_directory(
            session.sandbox_id, session.local_fs_root, sandbox_fs_root
        )

        session.sandbox_fs_root = sandbox_fs_root
        session.sandbox_control_dir = sandbox_control_dir
        session.paths = _build_worker_paths(sandbox_control_dir)

        state["rlm_fs_staging_root"] = session.local_fs_root
        state["rlm_control_dir_local"] = session.local_control_dir
        state["rlm_fs_root_remote"] = sandbox_fs_root
        state["rlm_control_dir_remote"] = sandbox_control_dir
        state["rlm_paths_remote"] = session.paths.to_dict()

    async def setup(self, state: State) -> None:
        session = self._get_session(state)
        if not session.sandbox_id:
            raise RLMSessionError("Sandbox not initialized")
        if not session.paths:
            raise RLMSessionError("Sandbox paths not initialized")

        await self._install_packages(session)
        await self._write_sandbox_files(session, state)
        await self._start_worker(session, state)

    async def execute(self, payload: dict[str, Any], state: State) -> RLMExecResult:
        session = self._get_session(state)
        if not session.sandbox_id or not session.paths:
            raise RLMSessionError("Sandbox session not initialized")

        try:
            raw = await self._send_worker_request(session, payload)
        except CommandTimeoutError as e:
            raise RLMCodeExecutionTimeout from e
        except RLMCodeExecutionTimeout:
            raise
        except Exception as e:
            raise vf.SandboxError() from e

        return RLMExecResult(stdout=raw, stderr="")

    async def read_answer(self, state: State) -> str:
        session = self._sessions.get(state.get("rollout_id", ""))
        if not session or not session.sandbox_id or not session.paths:
            return ""
        cmd = f"bash -lc 'cat {session.paths.answer_file} 2>/dev/null || true'"
        try:
            result = await self._execute_sandbox_command(
                session.sandbox_id,
                cmd,
                timeout=self.env.code_execution_timeout,
            )
        except Exception:
            return ""
        content = (result.stdout or "").strip()
        if not content:
            return ""
        try:
            return json.loads(content).get("content", "")
        except Exception:
            return ""

    async def recover_from_timeout(self, state: State) -> bool:
        session = self._sessions.get(state.get("rollout_id", ""))
        if not session or not session.sandbox_id or not session.paths:
            logger.error("Cannot recover from timeout: missing sandbox session")
            return False
        try:
            await self._stop_worker(session)
            await self._write_sandbox_files(session, state)
            await self._start_worker(session, state)
        except Exception as e:
            logger.error(f"Failed to recover from code timeout: {e}")
            return False
        state["rlm_worker_ready"] = True
        state["_exec_seq"] = 0
        return True

    async def cleanup(self, state: State) -> None:
        rollout_id = state.get("rollout_id")
        if not rollout_id:
            return
        session = self._sessions.pop(rollout_id, None)
        if not session:
            return
        await self._stop_worker(session)

        retain = state.get("retain_filesystem_after_rollout", False)
        if retain:
            self._retained_dirs.add(session.local_rollout_dir)
            if session.sandbox_id:
                sandbox_fs_root = session.sandbox_fs_root or state.get(
                    "rlm_fs_root_remote"
                )
                if sandbox_fs_root:
                    try:
                        await self._download_directory(
                            session.sandbox_id,
                            sandbox_fs_root,
                            session.local_fs_root,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to sync sandbox filesystem for rollout %s: %s",
                            rollout_id,
                            e,
                        )
                await self.delete_sandbox(session.sandbox_id)
            return

        if session.sandbox_id:
            await self.delete_sandbox(session.sandbox_id)

        await asyncio.to_thread(shutil.rmtree, session.local_rollout_dir, True)

    async def teardown(self) -> None:
        if self._sessions:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                try:
                    await self._stop_worker(session)
                finally:
                    if session.sandbox_id:
                        self.active_sandboxes.add(session.sandbox_id)
                    if session.local_rollout_dir not in self._retained_dirs:
                        shutil.rmtree(session.local_rollout_dir, True)
        if self.active_sandboxes:
            sandbox_ids = list(self.active_sandboxes)
            batch_size = 100
            sync_client = SandboxClient(APIClient())
            for i in range(0, len(sandbox_ids), batch_size):
                batch = sandbox_ids[i : i + batch_size]
                try:
                    sync_client.bulk_delete(sandbox_ids=batch)
                    for sandbox_id in batch:
                        self.active_sandboxes.discard(sandbox_id)
                    logger.debug(f"Bulk deleted batch of {len(batch)} sandboxes")
                except Exception as e:
                    logger.warning(f"Bulk delete failed for batch: {e}")
        self.teardown_sandbox_client()

    def _get_or_create_session(self, state: State) -> SandboxRLMReplSession:
        rollout_id = state.get("rollout_id")
        if not rollout_id:
            raise ValueError("rollout_id must be set before creating sandbox session")
        session = self._sessions.get(rollout_id)
        if session:
            return session
        rollout_dir = Path(tempfile.mkdtemp(prefix=f"rlm_rollout_{rollout_id}_"))
        fs_root = rollout_dir / "rlm_fs"
        control_dir = rollout_dir / "rlm_control"
        fs_root.mkdir(parents=True, exist_ok=True)
        control_dir.mkdir(parents=True, exist_ok=True)
        session = SandboxRLMReplSession(
            rollout_id=rollout_id,
            local_rollout_dir=str(rollout_dir),
            local_fs_root=str(fs_root),
            local_control_dir=str(control_dir),
        )
        self._sessions[rollout_id] = session
        return session

    def _get_session(self, state: State) -> SandboxRLMReplSession:
        rollout_id = state.get("rollout_id")
        if not rollout_id or rollout_id not in self._sessions:
            raise RLMSessionError("Sandbox session not initialized")
        return self._sessions[rollout_id]

    def _build_sandbox_request(self, state: State) -> CreateSandboxRequest:
        return self.env.get_sandbox_request(state)

    async def post_sandbox_setup(self, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            raise RLMSessionError("Sandbox not initialized")
        try:
            # Allow environments to run repo/tool setup before the worker starts.
            await self.env.on_sandbox_ready(state, sandbox_id)
        except Exception as exc:
            raise RLMSetupError(f"Sandbox setup hook failed: {exc}") from exc

    async def _execute_sandbox_command(
        self,
        sandbox_id: str,
        command: str,
        *,
        working_dir: str | None = None,
        timeout: int = 30,
    ) -> Any:
        try:
            return await self.sandbox_client.execute_command(
                sandbox_id,
                command,
                working_dir=working_dir,
                timeout=timeout,
            )
        except CommandTimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(e)

    async def _install_packages(self, session: SandboxRLMReplSession) -> None:
        sandbox_id = session.sandbox_id
        if not sandbox_id:
            raise RLMSessionError("Sandbox not initialized")
        packages = ["requests"]
        extras = [p.strip() for p in self.env.pip_install_packages.split() if p.strip()]
        packages.extend(extras)
        if not packages:
            return
        # Check each package with a quick import and only
        # install the ones that are missing. This avoids failures when pip is
        # unavailable on PATH but the package is already present in the image.
        # For example, in mini-swe-agent-plus-rlm
        missing: list[str] = []
        for pkg in packages:
            name = pkg.strip()
            name = name.split("@", 1)[0].strip()
            name = name.split("[", 1)[0].strip()
            # Strip version constraints (e.g., "numpy>1.20,<2.0") at the first specifier.
            name = re.split(r"[<>=!~]", name, 1)[0].strip()
            module = name.replace("-", "_")
            check_cmd = f"bash -lc 'python -c \"import {module}\"'"
            try:
                result = await self._execute_sandbox_command(
                    sandbox_id,
                    check_cmd,
                    timeout=self.env.max_startup_wait_seconds,
                )
            except Exception:
                missing.append(pkg)
                continue
            exit_code = getattr(result, "exit_code", 0)
            if exit_code not in (0, None):
                missing.append(pkg)
        if not missing:
            return
        pkg_list = " ".join(missing)
        cmd = f"bash -lc 'python -m pip install -q {pkg_list}'"
        result = await self._execute_sandbox_command(
            sandbox_id,
            cmd,
            timeout=self.env._compute_install_wait_seconds(),
        )
        self._raise_on_command_error(result, "pip install")

    async def _write_sandbox_files(
        self, session: SandboxRLMReplSession, state: State
    ) -> None:
        assert session.paths is not None
        context = {
            "fs_root": state.get("rlm_fs_root_remote") or state.get("rlm_fs_root"),
        }
        context_path = Path(session.local_control_dir) / "rlm_context.json"
        answer_path = Path(session.local_control_dir) / "rlm_answer.json"
        worker_path = Path(session.local_control_dir) / "rlm_worker.py"

        context_path.write_text(json.dumps(context), encoding="utf-8")
        answer_path.write_text(
            json.dumps({"ready": False, "content": ""}), encoding="utf-8"
        )

        worker_script = _render_worker_script(
            session.paths,
            repl_language=self.env.repl_language,
        )
        worker_script = self.env.customize_worker_script(worker_script, state)
        worker_path.write_text(worker_script, encoding="utf-8")

        await self.sandbox_client.upload_file(
            session.sandbox_id, session.paths.context_file, str(context_path)
        )
        await self.sandbox_client.upload_file(
            session.sandbox_id, session.paths.answer_file, str(answer_path)
        )
        await self.sandbox_client.upload_file(
            session.sandbox_id, session.paths.worker_path, str(worker_path)
        )

    async def _start_worker(self, session: SandboxRLMReplSession, state: State) -> None:
        assert session.paths is not None
        sandbox_id = session.sandbox_id
        if not sandbox_id:
            raise RLMSessionError("Sandbox not initialized")
        env_vars = self.env._build_worker_env_vars(state)

        exports = " ".join(
            f"{key}={shlex.quote(str(value))}"
            for key, value in env_vars.items()
            if value is not None
        )
        export_cmd = f"export {exports}; " if exports else ""
        script = (
            f'rm -f "{session.paths.command_fifo}" "{session.paths.response_fifo}" '
            f'"{session.paths.ready_flag}" "{session.paths.worker_pid_file}"; '
            f"{export_cmd}"
            f'python -u "{session.paths.worker_path}" > "{session.paths.log_file}" 2>&1 & '
            f'echo $! > "{session.paths.worker_pid_file}"'
        )
        cmd = f"bash -lc {shlex.quote(script)}"
        result = await self._execute_sandbox_command(
            sandbox_id,
            cmd,
            timeout=self.env.max_startup_wait_seconds,
        )
        self._raise_on_command_error(result, "start worker")
        await self._wait_for_ready(session)

    async def _wait_for_ready(self, session: SandboxRLMReplSession) -> None:
        assert session.paths is not None
        sandbox_id = session.sandbox_id
        if not sandbox_id:
            raise RLMSessionError("Sandbox not initialized")
        cmd = (
            "bash -lc '"
            f"for i in $(seq 1 {self.env.max_startup_wait_seconds * 10}); do "
            f'if [ -f "{session.paths.ready_flag}" ]; then exit 0; fi; '
            "sleep 0.1; "
            "done; exit 1'"
        )
        try:
            result = await self._execute_sandbox_command(
                sandbox_id,
                cmd,
                timeout=self.env.max_startup_wait_seconds,
            )
        except CommandTimeoutError as exc:
            log_tail = await self._read_worker_log_tail(session)
            raise RLMWorkerError(
                "RLM worker failed to become ready before timeout."
                + (f"\nLog tail:\n{log_tail}" if log_tail else "")
            ) from exc
        exit_code = getattr(result, "exit_code", 0)
        if exit_code != 0:
            log_tail = await self._read_worker_log_tail(session)
            raise RLMWorkerError(
                "RLM worker failed to become ready."
                + (f"\nLog tail:\n{log_tail}" if log_tail else "")
            )

    async def _stop_worker(self, session: SandboxRLMReplSession) -> None:
        if not session.sandbox_id or not session.paths:
            return
        sandbox_id = session.sandbox_id
        cmd = (
            "bash -lc '"
            f'if [ -f "{session.paths.worker_pid_file}" ]; then '
            f'pid=$(cat "{session.paths.worker_pid_file}"); '
            'kill "$pid" 2>/dev/null || true; '
            "fi'"
        )
        try:
            await self._execute_sandbox_command(
                sandbox_id,
                cmd,
                timeout=self.env.max_startup_wait_seconds,
            )
        except Exception:
            pass

    def _raise_on_command_error(self, result: Any, context: str) -> None:
        exit_code = getattr(result, "exit_code", 0)
        if exit_code == 0 or exit_code is None:
            return
        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        detail = ""
        if stdout:
            detail += f"\nstdout:\n{stdout}"
        if stderr:
            detail += f"\nstderr:\n{stderr}"
        raise RLMSetupError(f"{context} failed with exit code {exit_code}.{detail}")

    async def _read_worker_log_tail(self, session: SandboxRLMReplSession) -> str:
        if not session.sandbox_id or not session.paths:
            return ""
        sandbox_id = session.sandbox_id
        cmd = f"bash -lc 'tail -n 200 \"{session.paths.log_file}\" 2>/dev/null || true'"
        try:
            result = await self._execute_sandbox_command(
                sandbox_id,
                cmd,
                timeout=self.env.max_startup_wait_seconds,
            )
        except Exception:
            return ""
        return (getattr(result, "stdout", "") or "").strip()

    async def _send_worker_request(
        self, session: SandboxRLMReplSession, payload: dict[str, Any]
    ) -> str:
        assert session.paths is not None
        sandbox_id = session.sandbox_id
        if not sandbox_id:
            raise RLMSessionError("Sandbox not initialized")
        payload_json = json.dumps(payload)
        payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("utf-8")
        timeout_seconds = int(self.env.code_execution_timeout)
        alive_check = (
            f'[ -f "{session.paths.worker_pid_file}" ] '
            f'&& [ -d "/proc/$(cat {session.paths.worker_pid_file})" ] '
            '|| { echo "WORKER_DEAD"; exit 0; }'
        )
        command = textwrap.dedent(
            f"""
            {alive_check}
            python - <<'PY'
    import base64
    import errno
    import json
    import os
    import select
    import sys
    import time

    data = base64.b64decode('{payload_b64}').decode('utf-8')
    command_fifo = '{session.paths.command_fifo}'
    response_fifo = '{session.paths.response_fifo}'
    timeout_seconds = {timeout_seconds}
    deadline = time.time() + timeout_seconds

    try:
        cmd_fd = os.open(command_fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno in (errno.ENXIO, errno.ENOENT):
            print("WORKER_DEAD")
            sys.exit(0)
        raise
    try:
        payload_bytes = data.encode("utf-8")
        remaining = payload_bytes
        while remaining:
            try:
                written = os.write(cmd_fd, remaining)
                remaining = remaining[written:]
            except BlockingIOError:
                now = time.time()
                if now >= deadline:
                    print("WORKER_TIMEOUT")
                    sys.exit(0)
                timeout = min(0.05, deadline - now)
                _, writable, _ = select.select([], [cmd_fd], [], timeout)
                if not writable:
                    continue
    finally:
        os.close(cmd_fd)

    try:
        res_fd = os.open(response_fifo, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno in (errno.ENOENT,):
            print("WORKER_DEAD")
            sys.exit(0)
        raise
    chunks = []
    try:
        while True:
            now = time.time()
            if now >= deadline:
                print("WORKER_TIMEOUT")
                sys.exit(0)
            timeout = min(0.05, deadline - now)
            ready, _, _ = select.select([res_fd], [], [], timeout)
            if not ready:
                continue
            chunk = os.read(res_fd, 4096)
            if chunk:
                chunks.append(chunk)
                continue
            if not chunks:
                time.sleep(0.01)
                continue
            break
    finally:
        os.close(res_fd)
    if not chunks:
        print("WORKER_DEAD")
        sys.exit(0)
    sys.stdout.write(b"".join(chunks).decode("utf-8", errors="replace"))
    PY
            """
        )
        result = await self._execute_sandbox_command(
            sandbox_id,
            command,
            timeout=self.env.code_execution_timeout,
        )
        raw_response = result.stdout or ""
        if raw_response and raw_response.strip() == "WORKER_DEAD":
            raise RLMCodeExecutionTimeout
        if raw_response and raw_response.strip() == "WORKER_TIMEOUT":
            raise RLMCodeExecutionTimeout
        return raw_response

    async def _upload_directory(
        self, sandbox_id: str, local_dir: str, remote_dir: str
    ) -> None:
        local_path = Path(local_dir)
        tar_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
            tar_path = Path(tmp.name)
            tmp.close()
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(local_path, arcname=".")
            remote_tar = f"/tmp/rlm_upload_{uuid.uuid4().hex}.tar.gz"
            await self.sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            extract_cmd = (
                "bash -lc '"
                f'mkdir -p "{remote_dir}"; '
                f'tar -xzf "{remote_tar}" -C "{remote_dir}"; '
                f'rm -f "{remote_tar}"\''
            )
            await self._execute_sandbox_command(
                sandbox_id,
                extract_cmd,
                timeout=self.env.max_startup_wait_seconds,
            )
        finally:
            if tar_path and tar_path.exists():
                try:
                    tar_path.unlink()
                except Exception:
                    pass

    async def _download_directory(
        self, sandbox_id: str, remote_dir: str, local_dir: str
    ) -> None:
        local_path = Path(local_dir)
        tar_path = None
        remote_tar = f"/tmp/rlm_download_{uuid.uuid4().hex}.tar.gz"
        try:
            create_cmd = f'bash -lc \'tar -czf "{remote_tar}" -C "{remote_dir}" .\''
            await self._execute_sandbox_command(
                sandbox_id,
                create_cmd,
                timeout=self.env.max_startup_wait_seconds,
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
            tar_path = Path(tmp.name)
            tmp.close()
            await self.sandbox_client.download_file(
                sandbox_id, remote_tar, str(tar_path)
            )
            if local_path.exists():
                shutil.rmtree(local_path, True)
            local_path.mkdir(parents=True, exist_ok=True)
            base_dir = local_path.resolve()
            with tarfile.open(tar_path, "r:gz") as tar:
                safe_members = []
                for member in tar.getmembers():
                    if member.issym() or member.islnk():
                        logger.warning(
                            "Skipping symlink in sandbox download: %s", member.name
                        )
                        continue
                    try:
                        member_path = (base_dir / member.name).resolve()
                        member_path.relative_to(base_dir)
                    except Exception:
                        logger.warning(
                            "Skipping unsafe tar member in sandbox download: %s",
                            member.name,
                        )
                        continue
                    safe_members.append(member)
                tar.extractall(local_path, members=safe_members)
        finally:
            try:
                await self._execute_sandbox_command(
                    sandbox_id,
                    f"bash -lc 'rm -f \"{remote_tar}\"'",
                    timeout=self.env.max_startup_wait_seconds,
                )
            except Exception:
                pass
            if tar_path and tar_path.exists():
                try:
                    tar_path.unlink()
                except Exception:
                    pass


class RLMEnv(vf.StatefulToolEnv):
    """
    Recursive Language Model Environment.

    Extends StatefulToolEnv to provide a Python REPL environment where the model can:
    - Interact with large input data stored in a working directory (filesystem)
    - Make recursive sub-LLM calls via `llm_batch()`
    - Return final answers via an `answer` variable
    - Record the actual model-visible prompt (with RLM scaffolding) in state["prompt"]
      on the first turn; the original prompt is preserved in state["raw_prompt"]

    Architecture:
    - REPL loop runs in the framework (standard MultiTurnEnv pattern)
    - Code execution runs in a sandbox via a persistent worker
    - Sub-LLM calls from worker code are intercepted via HTTP proxy

    Works with any dataset that has a normal prompt. Input data can optionally
    be provided via info[context_dir_key] (directory path) or info[context_key]
    (legacy builtin data written to a file).
    When using the sandbox backend, the sandbox and worker are started eagerly
    during setup_state. Environments that need the worker to start in an existing
    sandbox path can set state["rlm_fs_root_remote"] (and optionally
    state["rlm_control_dir_remote"]) before calling super().setup_state; otherwise
    the default remote paths are /tmp/rlm_<id>/rlm_fs and /tmp/rlm_<id>/rlm_control.
    Environments can also override get_sandbox_request, on_sandbox_ready, and
    customize_worker_script to adjust sandbox creation, perform post-create setup
    (e.g., repo initialization), or tweak the worker script without forking the
    executor implementation.

    Args:
        tools: List of tools shared by both the root REPL and sub-LLMs.
                   These are added first in the tool documentation order.
        root_tools: List of tools available only to the root REPL.
                   The root model can call these inside the REPL as Python functions.
        sub_tools: List of tools available only to sub-LLMs.
                   Sub-LLMs access these via standard tool calling.
        (Ordering) The root tool list is: fixed tools (e.g. llm_batch), then `tools`,
                   then `root_tools`. The sub-LLM tool list is: `tools`, then `sub_tools`.
                   Each list is deduplicated by tool name. If two different tools
                   share a name within a list, initialization raises an error.
        sub_tool_max_turns: Maximum tool-calling turns for sub-LLM calls (default: 5)
        sub_model: Model to use for sub-LLM calls (defaults to same as root model)
        sub_prompt_verbosity: The verbosity of the sub-LLMs' system prompt; "light", "medium", or "high"
        root_prompt_verbosity: The verbosity of the root-LLM's system prompt; "light", "medium", or "high"
        max_iterations: Maximum REPL iterations before stopping (maps to max_turns)
        max_output_length: Maximum length of code execution output
        max_sub_llm_parallelism: Maximum number of concurrent sub-LLM calls
        sub_llm_stagger_ms: Optional fixed per-call stagger delay (ms) within llm_batch.
        sub_llm_stagger_jitter_ms: Optional random jitter (ms) added to stagger delay.
        context_key: Key in info containing legacy context data (default: "context")
        context_dir_key: Key in info containing directory path (default: "context_dir")
        system_prompt: Custom system prompt (default: RLM standard prompt)
        repl_language: REPL language to use: "bash" or "python" (default: "bash")
        execution_backend: Deprecated, has no effect. Sandbox execution is always used.
        interception_host: Optional hostname/IP for interception server (default: 127.0.0.1)
        interception_port: Port for interception server (default: 8766)
        interception_url: Optional base URL for interception (sandbox only). If set,
                   tunnel startup is skipped.
        pip_install_packages: Space-separated packages to install in addition to requests
                   (default: "")
        include_sub_llm_in_trajectory: Whether to include sub-LLM calls as trajectory steps.
                   When True, sub-LLM turns are added to the trajectory as TrajectoryStep
                   objects with tokens, enabling training on sub-LLM calls. Interleaved
                   rollouts are supported in this mode; the environment ensures that
                   get_prompt_messages, get_model_response, and stop conditions always
                   reference the last main-model step rather than a sub-LLM step.
                   When False (default), sub-LLM calls happen but are not stored.
        context_warning_threshold: Fraction of max_seq_len at which to warn the model
                   to finish (default: 0.80). Only active if max_seq_len is set.
        max_startup_wait_seconds: Maximum seconds to wait for worker startup (default: 120)
        code_execution_timeout: Timeout in seconds for code execution (default: 120).
                   This is longer than the default command timeout to allow for
                   llm_batch calls which can take several minutes.
        abort_on_code_timeout: If True, abort the rollout when code execution times out.
                   If False (default), return an error message to the model so it can
                   try a more efficient approach.
        retain_filesystem_after_rollout: If True, keep filesystem after rollout.
        filesystem_copy_max_bytes: Optional max bytes for context directory copy.
        local_repl_max_workers: Deprecated, has no effect. Sandbox execution is always used.
        sandbox_docker_image: Docker image for sandbox backend (default: python:3.11-slim)
        sandbox_start_command: Start command for sandbox backend (default: tail -f /dev/null)
        sandbox_cpu_cores: Sandbox CPU cores (default: 1)
        sandbox_memory_gb: Sandbox memory in GB (default: 2)
        sandbox_disk_size_gb: Sandbox disk size in GB (default: 5)
        sandbox_gpu_count: Sandbox GPU count (default: 0)
        sandbox_timeout_minutes: Sandbox timeout in minutes (default: 60)
        sandbox_environment_vars: Extra environment vars for sandbox (default: None)
        sandbox_team_id: Optional team id for sandbox (default: None)
        sandbox_advanced_configs: Optional advanced configs for sandbox (default: None)
        sandbox_labels: Optional labels for sandbox (default: None)
        sandbox_client_max_workers: Sandbox client pool size (default: 10)
        sandbox_client_max_connections: Sandbox client max connections (default: 100)
        sandbox_client_max_keepalive_connections: Sandbox client keepalive conns (default: 50)
        **kwargs: Additional arguments passed to StatefulToolEnv
    """

    def __init__(
        self,
        tools: list[Callable] | None = None,
        root_tools: list[Callable] | None = None,
        sub_tools: list[Callable] | None = None,
        sub_tool_max_turns: int = 5,
        sub_model: str | None = None,
        sub_prompt_verbosity: Literal["light", "medium", "heavy"] = "light",
        root_prompt_verbosity: Literal["light", "medium", "heavy"] = "light",
        max_iterations: int = 50,
        max_output_length: int = 8192,
        max_sub_llm_parallelism: int = 5,
        sub_llm_stagger_ms: int = 200,
        sub_llm_stagger_jitter_ms: int = 50,
        context_key: str = "context",
        context_dir_key: str = "context_dir",
        system_prompt: str | None = None,
        repl_language: Literal["bash", "python"] = "bash",
        execution_backend: Literal["local", "sandbox"] | None = None,
        interception_host: str | None = None,
        interception_port: int = 8766,
        interception_url: str | None = None,
        pip_install_packages: str = "",
        include_sub_llm_in_trajectory: bool = False,
        context_warning_threshold: float = 0.80,
        max_startup_wait_seconds: int = 120,
        code_execution_timeout: int = 120,
        abort_on_code_timeout: bool = False,
        retain_filesystem_after_rollout: bool = False,
        filesystem_copy_max_bytes: int | None = 1_000_000_000,
        local_repl_max_workers: int | None = None,  # deprecated
        sandbox_docker_image: str = "python:3.11-slim",
        sandbox_start_command: str = "tail -f /dev/null",
        sandbox_cpu_cores: int = 1,
        sandbox_memory_gb: int = 2,
        sandbox_disk_size_gb: int = 5,
        sandbox_gpu_count: int = 0,
        sandbox_timeout_minutes: int = 60,
        sandbox_environment_vars: dict[str, str] | None = None,
        sandbox_team_id: str | None = None,
        sandbox_advanced_configs: Any | None = None,
        sandbox_labels: list[str] | None = None,
        sandbox_client_max_workers: int = 50,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        **kwargs,
    ):
        if repl_language not in {"bash", "python"}:
            raise ValueError(
                f"Unsupported repl_language '{repl_language}'. Expected 'bash' or 'python'."
            )
        if execution_backend is not None:
            import warnings

            warnings.warn(
                "The 'execution_backend' parameter is deprecated and has no effect. "
                "RLMEnv now always uses sandbox execution.",
                DeprecationWarning,
                stacklevel=2,
            )
        if sub_prompt_verbosity not in {"light", "medium", "heavy"}:
            raise ValueError(
                f"Unsupported sub_prompt_verbosity '{sub_prompt_verbosity}. "
                "Expected 'light', 'medium', or 'high'"
            )
        if root_prompt_verbosity not in {"light", "medium", "heavy"}:
            raise ValueError(
                f"Unsupported root_prompt_verbosity '{root_prompt_verbosity}. "
                "Expected 'light', 'medium', or 'high'"
            )
        self.sub_prompt_verbosity = sub_prompt_verbosity
        self.root_prompt_verbosity = root_prompt_verbosity
        self.repl_language = repl_language
        self.execution_backend = "sandbox"
        self.sub_model = sub_model
        self.shared_tools = tools or []
        self.root_only_tools = root_tools or []
        self.sub_only_tools = sub_tools or []
        self.sub_tool_max_turns = sub_tool_max_turns
        self.max_iterations = max_iterations
        self.max_output_length = max_output_length
        self.max_sub_llm_parallelism = max_sub_llm_parallelism
        self.sub_llm_stagger_ms = sub_llm_stagger_ms
        self.sub_llm_stagger_jitter_ms = sub_llm_stagger_jitter_ms
        self.context_key = context_key
        self.context_dir_key = context_dir_key
        self.custom_system_prompt = system_prompt
        self.interception_host = interception_host or "127.0.0.1"
        self.interception_port = interception_port
        self.interception_url_override = interception_url
        self.pip_install_packages = pip_install_packages
        self.max_startup_wait_seconds = max_startup_wait_seconds
        self.include_sub_llm_in_trajectory = include_sub_llm_in_trajectory
        self.context_warning_threshold = context_warning_threshold
        self.code_execution_timeout = code_execution_timeout
        self.abort_on_code_timeout = abort_on_code_timeout
        self.retain_filesystem_after_rollout = retain_filesystem_after_rollout
        self.filesystem_copy_max_bytes = filesystem_copy_max_bytes
        self._interception_bind_host = self.interception_host
        self.sandbox_docker_image = sandbox_docker_image
        self.sandbox_start_command = sandbox_start_command
        self.sandbox_cpu_cores = sandbox_cpu_cores
        self.sandbox_memory_gb = sandbox_memory_gb
        self.sandbox_disk_size_gb = sandbox_disk_size_gb
        self.sandbox_gpu_count = sandbox_gpu_count
        self.sandbox_timeout_minutes = sandbox_timeout_minutes
        self.sandbox_environment_vars = sandbox_environment_vars
        self.sandbox_team_id = sandbox_team_id
        self.sandbox_advanced_configs = sandbox_advanced_configs
        self.sandbox_labels = sandbox_labels
        self.sandbox_client_max_workers = sandbox_client_max_workers
        self.sandbox_client_max_connections = sandbox_client_max_connections
        self.sandbox_client_max_keepalive_connections = (
            sandbox_client_max_keepalive_connections
        )
        if local_repl_max_workers is not None:
            import warnings

            warnings.warn(
                "The 'local_repl_max_workers' parameter is deprecated and has no effect. "
                "RLMEnv now always uses sandbox execution.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Server-side timeout for LLM API calls (shorter than worker HTTP timeout)
        # This ensures server responds before the worker request times out
        (
            self.sub_llm_api_timeout,
            self.sub_llm_timeout,
        ) = self._compute_sub_llm_timeouts()

        fixed_root_tools = self._build_fixed_root_tools()
        self.root_tools, self.root_tool_map = _merge_tool_lists(
            fixed_tools=fixed_root_tools,
            shared_tools=self.shared_tools,
            role_tools=self.root_only_tools,
            context="root tools",
            reserved_names=set(_FIXED_REPL_TOOL_NAMES),
        )
        self.sub_tools, self.sub_tool_map = _merge_tool_lists(
            fixed_tools=[],
            shared_tools=self.shared_tools,
            role_tools=self.sub_only_tools,
            context="sub-LLM tools",
            reserved_names=set(_FIXED_REPL_TOOL_NAMES),
        )
        self.sub_oai_tools: list[ChatCompletionFunctionToolParam] = [
            convert_func_to_oai_tool(tool) for tool in self.sub_tools
        ]
        self.root_tool_doc_funcs: list[Callable] = []
        for tool in self.root_tools:
            self.root_tool_doc_funcs.append(tool)
        self.root_oai_tools: list[ChatCompletionFunctionToolParam] = [
            convert_func_to_oai_tool(tool) for tool in self.root_tool_doc_funcs
        ]
        self.root_tool_names = [_tool_display_name(tool) for tool in self.root_tools]
        self.sub_tool_names = [_tool_display_name(tool) for tool in self.sub_tools]
        self.root_tool_serialization = "pickle"
        self._root_tool_context_var: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar("rlm_root_tool_context", default=None)
        )

        # Interception server state (shared across rollouts)
        self._interception_server: Any = None
        self._server_lock = asyncio.Lock()
        self._server_runner: Any = None
        self._server_site: Any = None
        self._tunnel: Tunnel | None = None
        self._tunnel_lock = asyncio.Lock()

        # Active rollout tracking for sub-LLM request routing
        self.active_rollouts: dict[str, dict[str, Any]] = {}

        super().__init__(
            tools=[],
            max_turns=max_iterations,
            **kwargs,
        )
        self.add_rubric(RLMMonitorRubric(root_tool_names=self.root_tool_names))
        self._executor = RLMExecutor(self)

        # Add the REPL tool (state is injected via update_tool_args)
        if self.repl_language == "bash":
            self.add_tool(self.call_bash_repl, args_to_skip=["state"])
        else:
            self.add_tool(self.call_python_repl, args_to_skip=["state"])

    def get_sandbox_request(self, state: State) -> CreateSandboxRequest:
        """Return the sandbox request for this rollout.

        Override this to customize the sandbox image or resources per-state.
        This is invoked by the sandbox executor when a sandbox needs to be created.
        """
        env_vars = dict(self.sandbox_environment_vars or {})
        return CreateSandboxRequest(
            name=f"rlm-{state.get('rollout_id', 'unknown')}",
            docker_image=self.sandbox_docker_image,
            start_command=self.sandbox_start_command,
            cpu_cores=self.sandbox_cpu_cores,
            memory_gb=self.sandbox_memory_gb,
            disk_size_gb=self.sandbox_disk_size_gb,
            gpu_count=self.sandbox_gpu_count,
            timeout_minutes=self.sandbox_timeout_minutes,
            environment_vars=env_vars,
            team_id=self.sandbox_team_id,
            advanced_configs=self.sandbox_advanced_configs,
            labels=self.sandbox_labels or [],
        )

    async def on_sandbox_ready(self, state: State, sandbox_id: str) -> None:
        """Hook for environment-specific sandbox setup.

        Override to perform repo initialization, tool upload, or other sandbox
        preparation steps after the sandbox is created and ready but before the
        worker starts. Defaults to a no-op.
        """
        return None

    def customize_worker_script(self, script: str, state: State) -> str:
        """Hook to adjust the generated worker script before it is written.

        Override to inject additional imports or tweaks without replacing the
        executor implementation. Defaults to returning the script unchanged.
        """
        return script

    # =========================================================================
    # Sub-Agent Tool Infrastructure
    # =========================================================================

    def _compute_sub_llm_timeouts(self) -> tuple[int, int]:
        """Compute sub-LLM timeouts based on the overall code execution timeout."""
        code_timeout = max(1, int(self.code_execution_timeout))
        min_timeout = min(10, max(1, code_timeout - 1))

        api_timeout = max(min_timeout, int(code_timeout * 0.8))
        worker_timeout = max(min_timeout, int(code_timeout * 0.9))

        if code_timeout > 1:
            api_timeout = min(api_timeout, code_timeout - 1)
            worker_timeout = min(worker_timeout, code_timeout - 1)

        api_timeout = min(api_timeout, worker_timeout)

        if code_timeout < 10:
            logger.warning(
                f"code_execution_timeout={code_timeout}s is low; sub-LLM calls may be unreliable"
            )

        return api_timeout, worker_timeout

    def _build_fixed_root_tools(self) -> list[Callable]:
        """Return the fixed root REPL tools (non-overridable)."""

        async def llm_batch(prompts: list[str]) -> list[str]:
            """
            Call the sub-LLM on multiple prompts in parallel.

            - Input: a list of prompt strings.
            - Output: a list of responses in the same order as the input prompts.
            - Use this inside the REPL to get help on sub-tasks.
            """
            # Context is injected only when called via the REPL root-tool endpoint.
            context = self._root_tool_context_var.get()
            if context is None:
                raise RuntimeError(
                    "llm_batch called outside of a tool request context."
                )
            results, _ = await self._root_llm_batch(context, prompts)
            return results

        llm_batch.__name__ = "llm_batch"
        return [llm_batch]

    def _compute_install_wait_seconds(self) -> int:
        """Estimate how long to wait for pip installs based on package count."""
        packages = [p.strip() for p in self.pip_install_packages.split() if p.strip()]
        package_count = len(packages) + 1  # Always includes requests
        estimated_seconds = 30 * package_count
        return max(self.max_startup_wait_seconds, estimated_seconds)

    def _build_worker_env_vars(self, state: State) -> dict[str, str]:
        env_vars = {
            "RLM_INTERCEPTION_URL": state.get("interception_url", ""),
            "RLM_ROOT_TOOL_URL": state.get("root_tool_url", ""),
            "RLM_ROOT_TOOL_NAMES": json.dumps(self.root_tool_names),
            "RLM_ROOT_TOOL_SERIALIZATION": self.root_tool_serialization,
            "RLM_SUB_MODEL": self.sub_model or state.get("model", ""),
            "RLM_MAX_SUB_LLM_PARALLELISM": str(self.max_sub_llm_parallelism),
            "RLM_SUB_LLM_STAGGER_MS": str(self.sub_llm_stagger_ms),
            "RLM_SUB_LLM_STAGGER_JITTER_MS": str(self.sub_llm_stagger_jitter_ms),
            "RLM_SUB_LLM_TIMEOUT": str(self.sub_llm_timeout),
        }
        return env_vars

    def _generate_packages_documentation(self) -> str:
        """Generate documentation for installed packages to include in system prompt."""
        if self.repl_language != "python":
            return ""
        if not self.pip_install_packages:
            return ""

        # Parse package names from pip_install_packages string
        packages = [p.strip() for p in self.pip_install_packages.split() if p.strip()]
        if not packages:
            return ""

        lines = [
            "\n## Installed Packages\n",
            "The following Python packages are pre-installed in the REPL environment:\n",
        ]
        for pkg in packages:
            lines.append(f"- `{pkg}`")
        lines.append("")
        lines.append("You can import and use these packages directly in your code.\n")

        return "\n".join(lines)

    def _append_tool_docs(
        self, lines: list[str], oai_tools: list[ChatCompletionFunctionToolParam]
    ) -> None:
        for oai_tool in oai_tools:
            func_def = oai_tool["function"]
            name = func_def["name"]
            desc = func_def.get("description", "No description")
            params = cast(
                dict[str, Any], func_def.get("parameters", {}).get("properties", {})
            )

            lines.append(f"### `{name}`")
            lines.append(f"{desc}\n")

            if params:
                lines.append("**Parameters:**")
                for param_name, param_info in params.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    lines.append(f"- `{param_name}` ({param_type}): {param_desc}")
                lines.append("")

    def _generate_sub_tools_documentation(self) -> str:
        """Generate documentation for sub-agent tools to include in system prompt."""
        if not self.sub_tools:
            return ""

        lines = ["\n## Sub-LLM Tools\n"]
        lines.append(
            "The sub-LLMs called via `llm_batch()` have access to the following tools:\n"
        )

        self._append_tool_docs(lines, self.sub_oai_tools)

        lines.append(
            "When delegating tasks to sub-LLMs via `llm_batch()`, they can use these "
            "tools autonomously."
        )
        lines.append(
            "You do NOT need to manage tool calls yourself - just describe the task "
            "in your prompt.\n"
        )

        return "\n".join(lines)

    def _generate_root_tools_documentation(self) -> str:
        """Generate documentation for root REPL tools to include in system prompt."""
        if not self.root_tools:
            return ""

        lines = ["\n## Root REPL Tools\n"]
        if self.repl_language == "bash":
            lines.append(
                "The root model can call the following tools inside the Bash REPL as shell commands:\n"
            )
        else:
            lines.append(
                "The root model can call the following tools inside the Python REPL:\n"
            )

        self._append_tool_docs(lines, self.root_oai_tools)

        lines.append(
            "These tools run on the host and are only accessible from within the REPL."
        )
        if self.repl_language == "bash":
            lines.append(
                "Bash usage: `tool_name arg1 arg2` (args are JSON-decoded). For "
                'structured args/kwargs, use `tool_name --json \'{"args": [...], '
                '"kwargs": {...}}\'` or provide the JSON via stdin.'
            )
            lines.append(
                "For `llm_batch`, use positional string prompts or "
                '`--json \'{"prompts": ["..."]}\'`.'
            )
        lines.append("")

        return "\n".join(lines)

    def _compute_fs_metadata(
        self, fs_root: str, *, disallow_symlinks: bool = False
    ) -> dict[str, int]:
        file_count = 0
        total_size = 0
        for root, dirs, files in os.walk(fs_root, followlinks=False):
            if disallow_symlinks:
                for name in [*dirs, *files]:
                    path = os.path.join(root, name)
                    if os.path.islink(path):
                        raise ValueError(
                            f"context_dir contains a symlink, which is not allowed: {path}"
                        )
            for name in files:
                file_count += 1
                path = os.path.join(root, name)
                try:
                    total_size += os.path.getsize(path)
                except OSError:
                    continue
        return {
            "file_count": file_count,
            "total_size": total_size,
            "total_bytes": total_size,
        }

    def _copy_context_directory(self, src: str, dst: str) -> None:
        src_path = os.fspath(src)
        if not os.path.isdir(src_path):
            raise ValueError(f"context_dir must be a directory: {src_path}")
        size_limit = self.filesystem_copy_max_bytes
        if os.path.islink(src_path):
            raise ValueError(f"context_dir cannot be a symlink: {src_path}")
        if size_limit is not None:
            metadata = self._compute_fs_metadata(src_path, disallow_symlinks=True)
            total_size = metadata.get("total_size", 0)
            if total_size > size_limit:
                raise ValueError(
                    "Context directory exceeds size limit: "
                    f"{total_size} bytes > {size_limit} bytes."
                )
        else:
            self._compute_fs_metadata(src_path, disallow_symlinks=True)
        shutil.copytree(src_path, dst, dirs_exist_ok=True)

    def _write_builtin_context(self, context_data: Any, fs_root: str) -> None:
        if isinstance(context_data, str):
            path = os.path.join(fs_root, "context.txt")
            Path(path).write_text(context_data, encoding="utf-8")
            return
        try:
            payload = json.dumps(context_data, ensure_ascii=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Legacy context data must be JSON-serializable or a raw string."
            ) from exc
        path = os.path.join(fs_root, "context.json")
        Path(path).write_text(payload, encoding="utf-8")

    async def _call_sub_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str
    ) -> dict:
        """Execute a sub-agent tool call. Returns tool message dict."""
        try:
            tool_func = self.sub_tool_map[tool_name]
            result = await maybe_await(tool_func, **tool_args)
            return {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call_id,
            }
        except Exception as e:
            if self._should_stop_for_error(e):
                raise
            return {
                "role": "tool",
                "content": f"Error: {e}",
                "tool_call_id": tool_call_id,
            }

    async def _call_sub_llm_api(
        self,
        state: State,
        client: Any,
        model: str,
        messages: ChatMessages,
        tools: list | None = None,
    ) -> Any | None:
        """Make a single sub-LLM API call matching main-model request mode."""
        sampling_args = dict(state.get("sampling_args") or {})
        extra_body = sampling_args.get("extra_body")
        if isinstance(extra_body, dict):
            sampling_args["extra_body"] = dict(extra_body)

        try:
            # Use a minimal state with an empty trajectory so get_model_response
            # never tries to compute interleaved prompt_ids from the main rollout.
            # Sub-LLM prompts are independent tool calls, not continuations of the
            # root conversation; using the real state would treat them as such.
            # We also mirror sampling_args/oai_tools onto the fake state because
            # get_model_response falls back to state values when args are falsy
            # (e.g., {} or None), which would otherwise raise KeyError.
            prompt_state = State()
            prompt_state["trajectory"] = []
            prompt_state["sampling_args"] = sampling_args
            prompt_state["oai_tools"] = tools or []
            return await asyncio.wait_for(
                self.get_model_response(
                    prompt_state,
                    messages,
                    client=client,
                    model=model,
                    message_type="chat",
                ),
                timeout=self.sub_llm_api_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Sub-LLM API call timed out after {self.sub_llm_api_timeout}s"
            )
            return None
        except vf.EmptyModelResponseError as e:
            raise SubLLMEmptyModelResponseError(str(e)) from e
        except Exception as e:
            raise e

    def _make_timeout_result(
        self,
        turns: list[SubLLMTurn],
        total_prompt_tokens: int,
        total_completion_tokens: int,
        tool_call_count: int,
        num_turns: int,
    ) -> SubLLMResult:
        """Create a SubLLMResult for timeout cases."""
        return SubLLMResult(
            final_content=f"Error: Sub-LLM API call timed out after {self.sub_llm_api_timeout}s",
            turns=turns,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            tool_call_count=tool_call_count,
            num_turns=num_turns,
            max_turns_reached=True,
        )

    async def _run_sub_llm(
        self, state: State, client: Any, model: str, messages: ChatMessages
    ) -> SubLLMResult:
        """Run a sub-LLM call, with optional tool-calling loop."""
        # Fast path: no tools configured - single LLM call
        if not self.sub_tools:
            response = await self._call_sub_llm_api(state, client, model, messages)
            if response is None:
                return self._make_timeout_result([], 0, 0, 0, 0)

            prompt_tokens, completion_tokens = _extract_tokens_from_response(response)
            return SubLLMResult(
                final_content=response.choices[0].message.content or "",
                turns=[
                    SubLLMTurn(
                        prompt_messages=[cast(ChatMessage, dict(m)) for m in messages],
                        response=response,
                        tool_call_count=0,
                    )
                ],
                total_prompt_tokens=prompt_tokens,
                total_completion_tokens=completion_tokens,
                tool_call_count=0,
                num_turns=1,
                max_turns_reached=False,
            )

        # Tool-calling loop path
        current_messages = list(messages)
        total_prompt_tokens = 0
        total_completion_tokens = 0
        tool_call_count = 0
        num_turns = 0
        turns: list[SubLLMTurn] = []
        tools = self.sub_oai_tools if self.sub_oai_tools else None

        for _ in range(self.sub_tool_max_turns):
            num_turns += 1
            prompt_snapshot = [cast(ChatMessage, dict(m)) for m in current_messages]

            response = await self._call_sub_llm_api(
                state,
                client,
                model,
                current_messages,
                tools,
            )
            if response is None:
                return self._make_timeout_result(
                    turns,
                    total_prompt_tokens,
                    total_completion_tokens,
                    tool_call_count,
                    num_turns,
                )

            prompt_tokens, completion_tokens = _extract_tokens_from_response(response)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            assistant_message = response.choices[0].message
            tool_calls = getattr(assistant_message, "tool_calls", None)
            turn_tool_count = len(tool_calls) if tool_calls else 0
            tool_call_count += turn_tool_count

            turns.append(
                SubLLMTurn(
                    prompt_messages=prompt_snapshot,
                    response=response,
                    tool_call_count=turn_tool_count,
                )
            )

            if not tool_calls:
                return SubLLMResult(
                    final_content=assistant_message.content or "",
                    turns=turns,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    tool_call_count=tool_call_count,
                    num_turns=num_turns,
                    max_turns_reached=False,
                )

            current_messages.append(cast(ChatMessage, assistant_message.model_dump()))

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                tool_result = await self._call_sub_tool(
                    tool_name, tool_args, tool_call.id
                )
                current_messages.append(cast(ChatMessage, tool_result))

        # Max turns reached - add prompt for final answer and make call without tools
        num_turns += 1
        current_messages.append(
            cast(
                ChatMessage,
                {
                    "role": "user",
                    "content": "You've reached the maximum number of tool calls. "
                    "Based on the information gathered, provide your final answer inside \\boxed{}.",
                },
            )
        )

        prompt_snapshot = [cast(ChatMessage, dict(m)) for m in current_messages]
        response = await self._call_sub_llm_api(
            state,
            client,
            model,
            current_messages,
        )
        if response is None:
            return self._make_timeout_result(
                turns,
                total_prompt_tokens,
                total_completion_tokens,
                tool_call_count,
                num_turns,
            )

        turns.append(
            SubLLMTurn(
                prompt_messages=prompt_snapshot, response=response, tool_call_count=0
            )
        )
        prompt_tokens, completion_tokens = _extract_tokens_from_response(response)

        return SubLLMResult(
            final_content=response.choices[0].message.content or "",
            turns=turns,
            total_prompt_tokens=total_prompt_tokens + prompt_tokens,
            total_completion_tokens=total_completion_tokens + completion_tokens,
            tool_call_count=tool_call_count,
            num_turns=num_turns,
            max_turns_reached=True,
        )

    async def _root_llm_batch(
        self,
        context: dict[str, Any],
        prompts: list[Any],
    ) -> tuple[list[str], list[str]]:
        """Run a batch of sub-LLM calls for root REPL usage."""
        if not isinstance(prompts, list):
            raise ValueError("llm_batch expects a list of prompts.")

        client = context.get("client")
        sub_model = context.get("sub_model") or context.get("model")
        state_ref = context.get("state")
        parent_turn = context.get("parent_turn", 0)
        if not client or not sub_model or state_ref is None:
            raise RuntimeError("Sub-LLM context is not available.")

        batch_start = perf_counter()
        batch_id = uuid.uuid4().hex[:8]
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        semaphore = asyncio.Semaphore(self.max_sub_llm_parallelism)

        def _coerce_prompt_messages(prompt: Any, index: int) -> ChatMessages:
            if isinstance(prompt, str):
                return [cast(ChatMessage, {"role": "user", "content": prompt})]
            raise ValueError(
                "llm_batch prompt at index " + str(index) + " must be a string."
            )

        async def _call_one(index: int, prompt: Any) -> None:
            jitter_ms = (
                random.random() * self.sub_llm_stagger_jitter_ms
                if self.sub_llm_stagger_jitter_ms > 0
                else 0.0
            )
            delay_s = max(0.0, (index * self.sub_llm_stagger_ms + jitter_ms) / 1000.0)
            if delay_s:
                await asyncio.sleep(delay_s)

            async with semaphore:
                request_id = uuid.uuid4().hex[:8]
                start_time = perf_counter()
                try:
                    messages = _coerce_prompt_messages(prompt, index)
                    response_dict = await self._run_sub_llm_request(
                        state_ref=state_ref,
                        client=client,
                        sub_model=sub_model,
                        messages=messages,
                        batch_id=batch_id,
                        request_id=request_id,
                        parent_turn=parent_turn,
                    )
                    elapsed = perf_counter() - start_time
                    response_dict.setdefault("_rlm_metadata", {})["elapsed_seconds"] = (
                        elapsed
                    )
                except Exception as exc:
                    if self._should_stop_for_error(exc):
                        raise
                    elapsed = perf_counter() - start_time
                    response_dict = {
                        "choices": [
                            {"message": {"content": f"Error in sub-LLM call: {exc}"}}
                        ],
                        "_rlm_metadata": {
                            "error": True,
                            "elapsed_seconds": elapsed,
                        },
                    }
                results[index] = response_dict

        await asyncio.gather(
            *[_call_one(i, prompt) for i, prompt in enumerate(prompts)]
        )

        batch_elapsed = perf_counter() - batch_start
        summary_lines = [f"llm_batch: {len(prompts)} call(s) in {batch_elapsed:.2f}s"]
        contents: list[str] = []
        for index, result in enumerate(results):
            if not result:
                contents.append("")
                summary_lines.append(f"  [{index}]: error (0.00s)")
                continue
            message = result.get("choices", [{}])[0].get("message", {})
            contents.append(message.get("content", ""))
            meta = result.get("_rlm_metadata", {})
            elapsed = meta.get("elapsed_seconds", 0.0)
            if meta.get("error"):
                summary_lines.append(f"  [{index}]: error ({elapsed:.2f}s)")
                continue
            prompt_tokens = meta.get("prompt_tokens", 0)
            completion_tokens = meta.get("completion_tokens", 0)
            tool_calls = meta.get("tool_call_count", 0)
            max_turns = meta.get("max_turns_reached", False)
            status = "⚠ max turns" if max_turns else "✓"
            summary_lines.append(
                f"  [{index}]: {prompt_tokens} prompt tokens, "
                f"{completion_tokens} completion tokens, "
                f"{tool_calls} tool calls, {elapsed:.2f}s {status}"
            )

        return contents, summary_lines

    # =========================================================================
    # Interception Server (for sub-LLM calls from worker code)
    # =========================================================================

    async def _ensure_interception_server(self):
        """Start shared HTTP server for sub-LLM interception if needed."""
        async with self._server_lock:
            if self._interception_server is not None:
                return

            app = web.Application()
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self._handle_sub_llm_request,
            )
            app.router.add_post(
                "/rollout/{rollout_id}/v1/rlm/tools",
                self._handle_root_tool_request,
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(
                runner, self._interception_bind_host, self.interception_port
            )
            await site.start()

            self._interception_server = app
            self._server_runner = runner
            self._server_site = site

            if self.interception_port == 0:
                server = getattr(site, "_server", None)
                sockets = getattr(server, "sockets", None) if server else None
                if sockets:
                    self.interception_port = sockets[0].getsockname()[1]

            logger.debug(
                f"Started RLM interception server on port {self.interception_port}"
            )

    async def _get_tunnel_url(self) -> str:
        """Get tunnel URL, starting the tunnel if needed."""
        async with self._tunnel_lock:
            if self._tunnel is None:
                if logger.isEnabledFor(logging.DEBUG):
                    self._tunnel = Tunnel(
                        local_port=self.interception_port,
                        log_level="debug",
                    )
                else:
                    self._tunnel = Tunnel(local_port=self.interception_port)
                url = await self._tunnel.start()
                logger.debug(f"Prime Tunnel started: {url}")
                return url
            else:
                assert self._tunnel.url is not None, "Tunnel started but URL is None"
                return self._tunnel.url

    async def _run_sub_llm_request(
        self,
        *,
        state_ref: State,
        client: Any,
        sub_model: str,
        messages: ChatMessages,
        batch_id: str,
        request_id: str,
        parent_turn: int,
        elapsed_seconds: float | None = None,
    ) -> dict[str, Any]:
        messages_with_system: ChatMessages = [
            cast(
                ChatMessage,
                {
                    "role": "system",
                    "content": _SUB_LLM_SYSTEM_PROMPT_STORE[
                        self.sub_prompt_verbosity
                    ].format(num_turns=self.sub_tool_max_turns),
                },
            ),
            *messages,
        ]

        result = await self._run_sub_llm(
            state_ref, client, sub_model, messages_with_system
        )
        final_content = result["final_content"]
        prompt_tokens = result["total_prompt_tokens"]
        completion_tokens = result["total_completion_tokens"]
        tool_call_count = result["tool_call_count"]
        num_turns = result["num_turns"]
        max_turns_reached = result["max_turns_reached"]
        turns = result["turns"]

        boxed_content = extract_boxed_answer(final_content)

        timestamp = time.time()
        total_sub_turns = len(turns)
        for sub_turn_index, turn in enumerate(turns):
            extras = {
                "is_sub_llm_call": True,
                "parent_turn": parent_turn,
                "batch_id": batch_id,
                "request_id": request_id,
                "sub_turn_index": sub_turn_index,
                "total_sub_turns": total_sub_turns,
                "timestamp": timestamp,
                "tool_call_count": turn["tool_call_count"],
            }

            if self.include_sub_llm_in_trajectory:
                tokens = await parse_response_tokens(
                    turn["response"], "chat", self.max_seq_len
                )
                completion_messages = await parse_response_messages(
                    turn["response"], "chat"
                )
                response_is_truncated = await parse_is_truncated(
                    turn["response"], "chat"
                )
                is_truncated = response_is_truncated or (
                    tokens is not None and bool(tokens.get("is_truncated"))
                )

                trajectory_step = TrajectoryStep(
                    prompt=cast(Messages, turn["prompt_messages"]),
                    completion=completion_messages,
                    response=turn["response"],
                    tokens=tokens,
                    reward=None,
                    advantage=None,
                    is_truncated=is_truncated,
                    trajectory_id=f"{batch_id}_{request_id}",
                    extras=extras,
                )
                await self.add_trajectory_step(state_ref, trajectory_step)
            else:
                trajectory_step = TrajectoryStep(
                    prompt=cast(Messages, turn["prompt_messages"]),
                    completion=[],
                    response=turn["response"],
                    tokens=None,
                    reward=None,
                    advantage=None,
                    is_truncated=False,
                    trajectory_id=f"{batch_id}_{request_id}",
                    extras=extras,
                )
                update_rlm_metrics_from_step(state_ref, trajectory_step)

        metadata = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_call_count": tool_call_count,
            "num_turns": num_turns,
            "max_turns_reached": max_turns_reached,
        }
        if elapsed_seconds is not None:
            metadata["elapsed_seconds"] = elapsed_seconds

        return {
            "choices": [{"message": {"content": boxed_content}}],
            "_rlm_metadata": metadata,
        }

    async def _handle_root_tool_request(self, request: Any) -> Any:
        """Handle root tool requests from worker."""
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        tool_name = request_body.get("tool_name", "")
        serialization = request_body.get("serialization", "pickle")
        if not tool_name:
            return web.json_response({"error": "Tool name not provided"}, status=400)
        if tool_name not in self.root_tool_map:
            return web.json_response(
                {"error": f"Tool '{tool_name}' not found"}, status=404
            )

        state_ref = context.get("state")
        if state_ref is None:
            return web.json_response({"error": "State not available"}, status=500)

        try:
            if serialization != "pickle":
                raise ValueError("Only pickle serialization is supported.")
            args = pickle.loads(base64.b64decode(request_body.get("args", "")))
            kwargs = pickle.loads(base64.b64decode(request_body.get("kwargs", "")))
            if not isinstance(args, tuple):
                raise ValueError("Pickle args payload must be a tuple.")
            if not isinstance(kwargs, dict):
                raise ValueError("Pickle kwargs payload must be a dict.")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

        parent_turn = context.get("current_turn", 0)
        root_tool_context = {
            "state": state_ref,
            "client": context.get("client"),
            "sub_model": context.get("sub_model") or context.get("model"),
            "parent_turn": parent_turn,
        }
        token = self._root_tool_context_var.set(root_tool_context)
        try:
            _update_root_tool_metrics(state_ref, tool_name)
            tool_func = self.root_tool_map[tool_name]
            if tool_name == "llm_batch":
                if args and "prompts" in kwargs:
                    raise ValueError("llm_batch received prompts twice.")
                if args:
                    if len(args) != 1:
                        raise ValueError("llm_batch expects a single prompts argument.")
                    prompts = args[0]
                elif "prompts" in kwargs:
                    prompts = kwargs.pop("prompts")
                else:
                    raise ValueError("llm_batch requires a prompts argument.")
                if kwargs:
                    raise ValueError(
                        "llm_batch does not accept extra keyword arguments: "
                        + ", ".join(sorted(kwargs))
                    )
                result_value, print_lines = await self._root_llm_batch(
                    root_tool_context, prompts
                )
            else:
                result_value = await maybe_await(tool_func, *args, **kwargs)
                print_lines = None
        except Exception as e:
            if self._should_stop_for_error(e):
                state_ref["_rlm_stop_error"] = e
            return web.json_response({"error": str(e)}, status=500)
        finally:
            self._root_tool_context_var.reset(token)

        result_payload = base64.b64encode(pickle.dumps(result_value)).decode("ascii")

        response_body: dict[str, Any] = {"result": result_payload}
        if print_lines:
            response_body["print_lines"] = print_lines
        return web.json_response(response_body)

    async def _handle_sub_llm_request(self, request: Any) -> Any:
        """Handle sub-LLM requests from worker code."""
        rollout_id = request.match_info["rollout_id"]
        context = self.active_rollouts.get(rollout_id)
        if not context:
            return web.json_response({"error": "Rollout not found"}, status=404)

        try:
            request_body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

        # Get client and model from rollout context
        client = context.get("client")
        sub_model = context.get("sub_model") or context.get("model")

        if not client:
            return web.json_response({"error": "Client not available"}, status=500)
        if not sub_model:
            return web.json_response({"error": "Model not available"}, status=500)

        messages = cast(ChatMessages, request_body.get("messages", []))
        batch_id = request_body.get("_batch_id", "")
        request_id = request_body.get("_request_id", "")

        state_ref = context.get("state") if context else None
        if state_ref is None:
            return web.json_response({"error": "State not available"}, status=500)

        parent_turn = context.get("current_turn", 0)
        try:
            response_dict = await self._run_sub_llm_request(
                state_ref=state_ref,
                client=client,
                sub_model=sub_model,
                messages=messages,
                batch_id=batch_id,
                request_id=request_id,
                parent_turn=parent_turn,
            )
            return web.json_response(response_dict)
        except Exception as e:
            if self._should_stop_for_error(e):
                state_ref["_rlm_stop_error"] = e
            return web.json_response({"error": str(e)}, status=500)

    async def _teardown_interception_server(self):
        """Stop the interception server if it was started."""
        async with self._server_lock:
            if self._server_site is not None:
                try:
                    await self._server_site.stop()
                finally:
                    self._server_site = None
            if self._server_runner is not None:
                try:
                    await self._server_runner.cleanup()
                finally:
                    self._server_runner = None
                    self._interception_server = None

    @vf.teardown
    async def teardown_interception_server(self):
        """Stop the interception server if it was started."""
        await self._teardown_interception_server()

    async def _teardown_tunnel(self) -> None:
        """Stop Prime Tunnel if it was started."""
        async with self._tunnel_lock:
            if self._tunnel is not None:
                try:
                    await self._tunnel.stop()
                    logger.debug("Prime Tunnel stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Prime Tunnel: {e}")
                finally:
                    self._tunnel = None

    @vf.teardown
    async def teardown_tunnel(self):
        """Stop Prime Tunnel if it was started."""
        await self._teardown_tunnel()

    @vf.teardown
    async def teardown_executor(self):
        """Cleanup executor-level resources (e.g., sandbox sessions)."""
        await self._executor.teardown()

    # =========================================================================
    # State Management
    # =========================================================================

    def set_interleaved_rollouts(self, interleaved_rollouts: bool) -> None:
        super().set_interleaved_rollouts(interleaved_rollouts)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict[str, Any]:
        """Inject state into REPL tool args."""
        if tool_name in {"call_python_repl", "call_bash_repl"}:
            updated_args = dict(tool_args)
            updated_args["state"] = state
            return updated_args
        else:
            return super().update_tool_args(
                tool_name, tool_args, messages, state, **kwargs
            )

    async def _setup_interception_and_register(
        self, state: State, rollout_id: str
    ) -> State:
        """Start interception server and register rollout."""
        await self._ensure_interception_server()
        if self.interception_url_override:
            base_url = self.interception_url_override.rstrip("/")
        else:
            base_url = await self._get_tunnel_url()
        interception_url = f"{base_url}/rollout/{rollout_id}/v1/chat/completions"
        root_tool_url = f"{base_url}/rollout/{rollout_id}/v1/rlm/tools"

        state["interception_url"] = interception_url
        state["root_tool_url"] = root_tool_url

        self.active_rollouts[rollout_id] = {
            "client": state.get("client"),
            "model": state.get("model"),
            "sub_model": self.sub_model or state.get("model"),
            "state": state,
        }
        return state

    async def setup_state(self, state: State, **kwargs) -> State:
        """Setup worker, filesystem context, and interception for sub-LLM calls."""
        state = await vf.StatefulToolEnv.setup_state(self, state, **kwargs)

        rollout_id = f"rlm_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id
        state.setdefault("rlm_fs_root_remote", f"/tmp/rlm_{rollout_id}/rlm_fs")
        state.setdefault("rlm_control_dir_remote", f"/tmp/rlm_{rollout_id}/rlm_control")

        try:
            # 1. Setup interception and register rollout
            state = await self._setup_interception_and_register(state, rollout_id)

            # 2. Create rollout directories
            self._executor.create_rollout_dirs(state)

            # 3. Build filesystem context
            info = state.get("info") or {}
            if not isinstance(info, dict):
                info = {}
            fs_root = state.get("rlm_fs_root")
            if not fs_root:
                raise ValueError("RLM filesystem root not initialized")
            fs_has_data = False
            fs_source: str | None = None

            context_dir = info.get(self.context_dir_key)
            if context_dir:
                fs_source = str(context_dir)
                self._copy_context_directory(fs_source, fs_root)
                fs_has_data = True
            else:
                context_data = info.get(self.context_key, None)
                if context_data is not None:
                    fs_has_data = True
                    self._write_builtin_context(context_data, fs_root)

            state["rlm_fs_root"] = fs_root
            state["rlm_fs_source"] = fs_source
            state["rlm_fs_has_data"] = fs_has_data
            state["retain_filesystem_after_rollout"] = (
                self.retain_filesystem_after_rollout
            )
            if self.custom_system_prompt:
                base_system_prompt = self.custom_system_prompt
            elif self.repl_language == "bash":
                base_system_prompt = _RLM_BASH_SYSTEM_PROMPT_STORE[
                    self.root_prompt_verbosity
                ]
            else:
                base_system_prompt = _RLM_PYTHON_SYSTEM_PROMPT_STORE[
                    self.root_prompt_verbosity
                ]

            packages_docs = self._generate_packages_documentation()
            root_tools_docs = self._generate_root_tools_documentation()
            sub_tools_docs = self._generate_sub_tools_documentation()
            state["rlm_system_prompt"] = (
                base_system_prompt + packages_docs + root_tools_docs + sub_tools_docs
            )
            state["rlm_packages_docs"] = packages_docs
            state["rlm_root_tools_docs"] = root_tools_docs
            state["rlm_sub_tools_docs"] = sub_tools_docs
            deduped_shared, _ = _dedupe_tools(
                self.shared_tools, context="shared tools", reserved_names=set()
            )
            state["rlm_shared_tools"] = [
                _tool_display_name(tool) for tool in deduped_shared
            ]
            state["rlm_root_tools"] = [
                _tool_display_name(tool) for tool in self.root_tools
            ]
            state["rlm_sub_tools"] = [
                _tool_display_name(tool) for tool in self.sub_tools
            ]

            # 4. Prepare backend and start worker (always eager)
            await self._executor.prepare_filesystem(state)
            await self._executor.setup(state)
            state["rlm_worker_ready"] = True

            # Initialize context warning flag (feature enabled if max_seq_len is set)
            state["context_warning_sent"] = False

            # Initialize FIFO sequence counter for detecting stale responses
            state["_exec_seq"] = 0

            _ensure_rlm_metric_state(state)

            return state
        except Exception:
            # Best-effort cleanup to avoid leaking tunnels/sandboxes on setup failure.
            if rollout_id in self.active_rollouts:
                del self.active_rollouts[rollout_id]
            try:
                await self._executor.cleanup(state)
            except Exception:
                logger.exception("Failed to cleanup RLM executor after setup error")
            if not self.active_rollouts:
                try:
                    await self._teardown_interception_server()
                finally:
                    await self._teardown_tunnel()
            raise

    # =========================================================================
    # Code Execution
    # =========================================================================

    async def _recover_from_code_timeout(self, state: State) -> bool:
        """Attempt to recover from a code execution timeout via the active backend."""
        return await self._executor.recover_from_timeout(state)

    async def _execute_code(self, code: str, state: State) -> dict[str, Any]:
        """Execute code in worker and return result."""
        if not state.get("rlm_worker_ready", False):
            await self._executor.prepare_filesystem(state)
            await self._executor.setup(state)
            state["rlm_worker_ready"] = True
        # Increment and track sequence number for this execution
        seq = state.get("_exec_seq", 0) + 1
        state["_exec_seq"] = seq

        payload = {"code": code, "seq": seq}
        try:
            result = await self._executor.execute(payload, state)
        except RLMCodeExecutionTimeout as e:
            logger.warning(
                "Code execution timed out after %ss", self.code_execution_timeout
            )
            if self.abort_on_code_timeout:
                raise
            recovered = await self._recover_from_code_timeout(state)
            if not recovered:
                raise RLMWorkerRecoveryError(
                    "Code execution timed out and the worker could not be restarted."
                ) from e
            # Return error to model so it can try more efficient code
            return {
                "status": "error",
                "stdout": "",
                "stderr": "",
                "result": (
                    f"Code execution timed out after {self.code_execution_timeout} seconds."
                    " The worker was restarted and the REPL state was reset."
                    " Your code may be too slow - consider a more "
                    "efficient algorithm or breaking the computation into smaller steps."
                ),
                "answer": {"ready": False, "content": ""},
            }

        if not result.stdout:
            return {
                "status": "error",
                "stdout": "",
                "stderr": result.stderr or "",
                "result": "Worker returned no output",
                "answer": {"ready": False, "content": ""},
            }

        try:
            parsed_result = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "stdout": result.stdout,
                "stderr": result.stderr or "",
                "result": f"Failed to parse worker response: {e}",
                "answer": {"ready": False, "content": ""},
            }

        # Check sequence number to detect stale responses (FIFO desync)
        response_seq = parsed_result.get("seq", -1)
        if response_seq != seq:
            logger.warning(
                f"FIFO sequence mismatch: expected seq={seq}, got seq={response_seq}. "
                "This indicates a desync - likely from a previous timeout."
            )
            return {
                "status": "error",
                "stdout": "",
                "stderr": "",
                "result": (
                    f"Communication desync detected: received stale response "
                    f"(expected seq={seq}, got seq={response_seq}). "
                    "This may happen after a timeout. Please retry your command."
                ),
                "answer": {"ready": False, "content": ""},
            }

        return parsed_result

    def _format_execution_output(self, result: dict[str, Any]) -> str:
        """Format execution result for display to model."""
        if self.repl_language == "bash":
            stdout = result.get("stdout") or ""
            stderr = result.get("stderr") or ""
            result_text = result.get("result") or ""
            output = f"{stdout}{stderr}"
            if not output and result_text:
                output = str(result_text)
            if not output:
                output = "(no output)"
            if len(output) > self.max_output_length:
                output = output[: self.max_output_length] + "\n... [output truncated]"
            return output

        parts: list[str] = []

        stdout = (result.get("stdout") or "").rstrip()
        if stdout:
            parts.append(stdout)

        stderr = (result.get("stderr") or "").rstrip()
        if stderr:
            parts.append(f"stderr:\n{stderr}")

        status = result.get("status")
        result_text = result.get("result")
        execution_count = result.get("execution_count", 0)

        if status == "error" and result_text:
            parts.append(result_text.rstrip())
        elif status == "ok" and result_text is not None:
            parts.append(f"Out[{execution_count}]: {result_text}")

        output = "\n".join(parts) if parts else "(no output)"

        # Truncate if too long
        if len(output) > self.max_output_length:
            output = output[: self.max_output_length] + "\n... [output truncated]"

        return output

    def _maybe_add_context_warning(
        self, output: str, state: State, *, ready_instruction: str
    ) -> str:
        """Append a context-limit warning if nearing max_seq_len."""
        if not self.max_seq_len or state.get("context_warning_sent"):
            return output

        trajectory = state.get("trajectory", [])
        last_main = next(
            (
                step
                for step in reversed(trajectory)
                if not step.get("extras", {}).get("is_sub_llm_call")
            ),
            None,
        )
        response = last_main.get("response") if last_main else None
        usage = getattr(response, "usage", None) if response else None
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        warning_threshold = int(self.max_seq_len * self.context_warning_threshold)

        if prompt_tokens >= warning_threshold:
            state["context_warning_sent"] = True
            pct = prompt_tokens / self.max_seq_len
            output += (
                f"\n\n[CONTEXT LIMIT WARNING] You have used {prompt_tokens:,} of "
                f"{self.max_seq_len:,} tokens ({pct:.0%}). {ready_instruction}"
            )

        return output

    # =========================================================================
    # REPL Tool
    # =========================================================================

    async def _call_repl(
        self,
        code: str,
        state: Any,
        *,
        ready_instruction: str,
        append_execution_time: bool,
    ) -> str:
        rollout_id = state.get("rollout_id")
        if rollout_id and rollout_id in self.active_rollouts:
            self.active_rollouts[rollout_id]["current_turn"] = state.get("turn", 0)

        execution_start = perf_counter()
        result = await self._execute_code(code, state)
        stop_exc = state.pop("_rlm_stop_error", None)
        if stop_exc is not None:
            raise stop_exc
        execution_time = perf_counter() - execution_start
        output = self._format_execution_output(result)

        state.setdefault("tool_call_timings", []).append(
            {
                "turn": state.get("turn", 0),
                "execution_seconds": execution_time,
            }
        )
        _update_rlm_repl_metrics(state, execution_time)

        if append_execution_time:
            output += f"\n[Execution time: {execution_time:.2f}s]"

        answer = result.get("answer", {})
        if answer.get("ready", False):
            state["final_answer"] = answer.get("content", "")
            logger.debug(f"Answer ready: {state['final_answer'][:100]}...")

        output = self._maybe_add_context_warning(
            output, state, ready_instruction=ready_instruction
        )

        return output

    async def call_bash_repl(self, code: str, state: Any) -> str:
        """
        Execute Bash commands in a persistent REPL environment.

        The Bash session maintains state across calls and provides access to:

        - Files in the working directory.
        - `RLM_CONTENT`: Your current best answer (string).
        - `RLM_READY`: Set to a truthy value to finish (terminates execution immediately).

        - `llm_batch` and other root tools: available as shell commands.

        Args:
            code: Bash code to execute in the persistent REPL

        Returns:
            Raw execution output (stdout/stderr combined)
        """
        return await self._call_repl(
            code,
            state,
            ready_instruction="Please finalize your answer soon by setting RLM_READY=1.",
            append_execution_time=False,
        )

    async def call_python_repl(self, code: str, state: Any) -> str:
        """
        Execute Python code in a persistent REPL environment.

        The REPL maintains state across calls and provides access to:

        - Files in the working directory.

        - `answer`: A dictionary for your final answer:
          - `answer["content"]`: Your answer (string) - update this as you work
          - `answer["ready"]`: Set to `True` to finish (terminates execution immediately)

        - `llm_batch(prompts)`: Make sub-LLM calls for help with subtasks
          - Takes a list of prompts, returns a list of answers (same order)
          - Useful for semantic understanding, summarization, complex reasoning
          - Prints metadata summary showing tokens and tool calls per sub-LLM

        Args:
            code: Python code to execute in the persistent REPL

        Returns:
            Execution output including stdout, stderr, and expression results
        """
        return await self._call_repl(
            code,
            state,
            ready_instruction="Please finalize your answer soon by setting answer['ready'] = True.",
            append_execution_time=True,
        )

    def _last_main_trajectory_step(self, state: State) -> TrajectoryStep | None:
        """Find the last trajectory step belonging to the main (root) model."""
        main_id = state.get("trajectory_id")
        for step in reversed(state.get("trajectory", [])):
            if step.get("trajectory_id") == main_id:
                return step
        return None

    async def add_trajectory_step(self, state: State, trajectory_step: TrajectoryStep):
        update_rlm_metrics_from_step(state, trajectory_step)
        await super().add_trajectory_step(state, trajectory_step)

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: ModelResponse,
    ):
        """Add model response and align stored prompt with injected scaffold on first turn."""
        if len(state["trajectory"]) == 0:
            if "raw_prompt" not in state:
                state["raw_prompt"] = state["prompt"]
            state["prompt"] = prompt_messages
        await super().add_model_response(state, prompt_messages, response)

    async def get_prompt_messages(self, state: State) -> Messages:
        """Build prompt messages, adding system prompt with tool docs on first turn."""
        if len(state["trajectory"]) == 0:
            # First turn: inject RLM scaffolding into the first user message
            prompt = state.get("prompt", [])
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

            system_prompt = state.get("rlm_system_prompt")
            packages_docs = state.get("rlm_packages_docs")
            root_tools_docs = state.get("rlm_root_tools_docs")
            sub_tools_docs = state.get("rlm_sub_tools_docs")
            if (
                system_prompt is None
                or packages_docs is None
                or root_tools_docs is None
                or sub_tools_docs is None
            ):
                raise ValueError("RLM setup_state must run before get_prompt_messages")

            messages = [cast(ChatMessage, dict(m)) for m in prompt]
            scaffold = (
                "<RLM_SCAFFOLDING>\n" + system_prompt + "\n</RLM_SCAFFOLDING>\n\n"
            )
            inserted = False
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                msg_mut = cast(dict[str, Any], msg)
                content = msg_mut.get("content")
                if isinstance(content, str) or content is None:
                    text = content or ""
                    if text.startswith("<RLM_SCAFFOLDING>"):
                        inserted = True
                        break
                    msg_mut["content"] = scaffold + text
                elif isinstance(content, list):
                    if (
                        content
                        and isinstance(content[0], dict)
                        and content[0].get("type") == "text"
                        and str(content[0].get("text", "")).startswith(
                            "<RLM_SCAFFOLDING>"
                        )
                    ):
                        inserted = True
                        break
                    msg_mut["content"] = [{"type": "text", "text": scaffold}, *content]
                elif isinstance(content, dict):
                    msg_mut["content"] = [
                        {"type": "text", "text": scaffold},
                        content,
                    ]
                inserted = True
                break

            if not inserted:
                messages.append(
                    cast(ChatMessage, {"role": "user", "content": scaffold})
                )

            return cast(Messages, messages)
        else:
            # Subsequent turns: use last main trajectory step (skip sub-LLM steps)
            last_main = self._last_main_trajectory_step(state)
            if last_main is None:
                return state["prompt"]
            prev_turn_prompt = last_main["prompt"]
            prev_turn_completion = last_main["completion"]
            messages = concat_messages([prev_turn_prompt, prev_turn_completion])
            env_response = await self.env_response(messages, state)
            return concat_messages([messages, env_response])

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        """Override to set final_env_response when answer is ready to avoid extra model call"""
        tool_messages = await super().env_response(messages, state, **kwargs)
        if "final_answer" in state:
            state["final_env_response"] = tool_messages
        return tool_messages

    async def get_model_response(  # type: ignore[override]
        self, state: State, prompt: Messages, **kwargs: Any
    ) -> ModelResponse:
        """Ensure get_prompt_ids sees the last main trajectory step, not a sub-LLM step.

        In interleaved mode, get_prompt_ids (called from super) reads
        state["trajectory"][-1] to build token-level prompts.  After
        env_response adds sub-LLM steps, trajectory[-1] may be a sub-LLM
        step with incompatible tokens.  We temporarily move trailing sub-LLM
        steps out of the trajectory for the duration of the super call.
        """
        if not (self.include_sub_llm_in_trajectory and self.interleaved_rollouts):
            return await super().get_model_response(state, prompt, **kwargs)

        trajectory = state.get("trajectory", [])
        if not trajectory:
            return await super().get_model_response(state, prompt, **kwargs)

        main_id = state["trajectory_id"]
        if trajectory[-1].get("trajectory_id") == main_id:
            return await super().get_model_response(state, prompt, **kwargs)

        # Find last main step and temporarily move trailing sub-LLM steps aside
        last_main_idx = None
        for i in range(len(trajectory) - 1, -1, -1):
            if trajectory[i].get("trajectory_id") == main_id:
                last_main_idx = i
                break

        if last_main_idx is None:
            return await super().get_model_response(state, prompt, **kwargs)

        trailing = trajectory[last_main_idx + 1 :]
        del trajectory[last_main_idx + 1 :]
        try:
            result = await super().get_model_response(state, prompt, **kwargs)
        finally:
            trajectory.extend(trailing)
        return result

    # =========================================================================
    # Stop Conditions
    # =========================================================================

    async def _ensure_final_answer(self, state: State) -> None:
        """Read final answer from worker if not already set."""
        if "final_answer" in state:
            return
        state["final_answer"] = await self._executor.read_answer(state)

    @vf.stop
    async def answer_ready(self, state: State) -> bool:
        """Stop when model sets answer['ready'] = True."""
        return "final_answer" in state

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        """Count only main-model trajectory steps, not sub-LLM steps."""
        if self.max_turns <= 0:
            return False
        main_id = state.get("trajectory_id")
        count = sum(
            1 for s in state.get("trajectory", []) if s.get("trajectory_id") == main_id
        )
        return count >= self.max_turns

    @vf.stop
    async def no_tools_called(self, state: State) -> bool:
        """Check last main-model completion for tool calls, ignoring sub-LLM steps."""
        last_main = self._last_main_trajectory_step(state)
        if last_main is None:
            return False
        last_message = cast(dict[str, Any], last_main["completion"][-1])
        is_assistant = last_message.get("role") == "assistant"
        no_tool_calls = (
            "tool_calls" not in last_message or last_message["tool_calls"] is None
        )
        return is_assistant and no_tool_calls

    @vf.stop
    async def prompt_too_long(self, state: State) -> bool:
        """Stop when API returns overlong prompt error."""
        if not state.get("prompt_too_long", False):
            return False

        await self._ensure_final_answer(state)
        return True

    # =========================================================================
    # Cleanup
    # =========================================================================

    @vf.cleanup
    async def cleanup_rlm_state(self, state: State):
        """Cleanup RLM-specific state and prepend sub-LLM trajectory steps."""
        rollout_id = state.get("rollout_id")

        if rollout_id and rollout_id in self.active_rollouts:
            del self.active_rollouts[rollout_id]

        try:
            await self._executor.cleanup(state)
        finally:
            if not self.active_rollouts:
                await self._teardown_interception_server()
                await self._teardown_tunnel()

    async def render_completion(self, state: State):
        """Render completion from main model steps only, ignoring sub-LLM steps."""

        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return

        # Find the last trajectory step from the main model (matching trajectory_id)
        main_trajectory_id = state["trajectory_id"]
        last_main_step = None
        for step in reversed(state["trajectory"]):
            if step.get("trajectory_id") == main_trajectory_id:
                last_main_step = step
                break

        if last_main_step is None:
            state["completion"] = []
            return

        last_prompt = last_main_step["prompt"]
        last_completion = last_main_step["completion"]
        full_conversation = concat_messages([last_prompt, last_completion])
        if state.get("final_env_response"):
            full_conversation = concat_messages(
                [full_conversation, state["final_env_response"]]
            )
        state["completion"] = full_conversation[len(state["prompt"]) :]

    async def post_rollout(self, state: State):
        """Read final answer from worker if not already set."""
        await self._ensure_final_answer(state)
