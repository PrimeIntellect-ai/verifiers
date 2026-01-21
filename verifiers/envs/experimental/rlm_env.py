"""
Recursive Language Model (RLM) Environment.

Implements the RLM inference strategy where language models can decompose and
recursively interact with input data of unbounded length through REPL environments.

Based on: https://www.alexzhang.dev/blog/recursive-language-models

Architecture:
- REPL loop runs in the framework (MultiTurnEnv pattern)
- Code execution runs locally in a persistent Python worker
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
import inspect
import json
import logging
import os
import pickle
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, cast

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from aiohttp import web
from openai.types.chat import ChatCompletion
import verifiers as vf
from verifiers.rubrics.rubric import Rubric
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
from verifiers.utils.token_utils import (
    prepare_sampling_args_for_token_prompts,
    tokenize_vllm,
)
import verifiers.utils.rlm_filesystem_jail as rlm_jail_module

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


class RLMCodeExecutionTimeout(Exception):
    """Raised when code execution exceeds the configured timeout."""


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


@dataclass
class LocalRLMReplSession:
    rollout_id: str
    rollout_dir: str
    paths: RLMWorkerPaths
    fs_root: str
    control_dir: str
    worker_process: subprocess.Popen | None = None
    venv_path: str | None = None


@dataclass(frozen=True)
class RLMExecResult:
    stdout: str
    stderr: str | None = None
    exit_code: int | None = None


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
    def __init__(self, root_tool_names: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.sub_llm_call_count)
        self.add_metric(self.sub_llm_total_turns)
        self.add_metric(self.sub_llm_prompt_tokens)
        self.add_metric(self.sub_llm_completion_tokens)
        self.add_metric(self.sub_llm_total_tool_calls)
        self.add_metric(self.sub_llm_batch_count)
        self.add_metric(self.sub_llm_max_batch_size)
        self.add_metric(self.sub_llm_mean_batch_size)
        self.add_metric(self.main_rlm_turns)
        self.add_metric(self.main_rlm_prompt_tokens)
        self.add_metric(self.main_rlm_completion_tokens)
        self.add_metric(self.repl_total_time_seconds)
        self.add_metric(self.repl_call_count)
        self.add_metric(self.repl_mean_time_seconds)
        self.add_metric(self.root_tool_call_count)
        for tool_name in root_tool_names or []:
            self.add_metric(self._make_root_tool_metric(tool_name))

    async def sub_llm_call_count(self, state: State) -> int:
        return state["sub_llm_call_count"]

    async def sub_llm_total_turns(self, state: State) -> int:
        return state["sub_llm_total_turns"]

    async def sub_llm_prompt_tokens(self, state: State) -> int:
        return state["sub_llm_prompt_tokens"]

    async def sub_llm_completion_tokens(self, state: State) -> int:
        return state["sub_llm_completion_tokens"]

    async def sub_llm_total_tool_calls(self, state: State) -> int:
        return state["sub_llm_total_tool_calls"]

    async def sub_llm_batch_count(self, state: State) -> int:
        return state["sub_llm_batch_count"]

    async def sub_llm_max_batch_size(self, state: State) -> int:
        return state["sub_llm_max_batch_size"]

    async def sub_llm_mean_batch_size(self, state: State) -> float:
        return state["sub_llm_mean_batch_size"]

    async def main_rlm_turns(self, state: State) -> int:
        return state["main_rlm_turns"]

    async def main_rlm_prompt_tokens(self, state: State) -> int:
        return state["main_rlm_prompt_tokens"]

    async def main_rlm_completion_tokens(self, state: State) -> int:
        return state["main_rlm_completion_tokens"]

    async def repl_total_time_seconds(self, state: State) -> float:
        return state["repl_total_time_seconds"]

    async def repl_call_count(self, state: State) -> int:
        return state["repl_call_count"]

    async def repl_mean_time_seconds(self, state: State) -> float:
        return state["repl_mean_time_seconds"]

    async def root_tool_call_count(self, state: State) -> int:
        return state["root_tool_call_count"]

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


# Worker script that runs locally - handles code execution only
# The REPL loop is managed by the framework, not this script
_RLM_WORKER_SCRIPT_TEMPLATE = textwrap.dedent(
    """
    import ast
    import base64
    import contextlib
    import io
    import json
    import os
    import pickle
    import sys
    import sysconfig
    import time
    import traceback
    from pathlib import Path
    import requests

    {filesystem_jail_code}

    COMMAND_FIFO = "{command_fifo}"
    RESPONSE_FIFO = "{response_fifo}"
    READY_FLAG = "{ready_flag}"
    CONTEXT_FILE = "{context_file}"
    ANSWER_FILE = "{answer_file}"

    # Sub-LLM configuration from environment
    INTERCEPTION_URL = os.environ.get("RLM_INTERCEPTION_URL", "")
    SUB_MODEL = os.environ.get("RLM_SUB_MODEL", "")
    MAX_SUB_LLM_PARALLELISM = int(os.environ.get("RLM_MAX_SUB_LLM_PARALLELISM", "5"))
    SUB_LLM_TIMEOUT = int(os.environ.get("RLM_SUB_LLM_TIMEOUT", "300"))
    SUB_LLM_STAGGER_MS = int(os.environ.get("RLM_SUB_LLM_STAGGER_MS", "0"))
    SUB_LLM_STAGGER_JITTER_MS = int(
        os.environ.get("RLM_SUB_LLM_STAGGER_JITTER_MS", "0")
    )

    # Guardrails for user code execution (best-effort, not an OS sandbox)
    def _parse_disallowed(raw: str) -> list[str]:
        if not raw:
            return []
        raw = raw.replace(",", " ")
        return [item.strip() for item in raw.split() if item.strip()]

    DISALLOWED_MODULES = set(
        _parse_disallowed(os.environ.get("RLM_DISALLOWED_MODULES", ""))
    )
    DISALLOWED_BUILTINS = set(
        _parse_disallowed(os.environ.get("RLM_DISALLOWED_BUILTINS", ""))
    )

    def _build_restricted_builtins() -> dict:
        builtins_obj = __builtins__
        if not isinstance(builtins_obj, dict):
            builtins_obj = builtins_obj.__dict__
        restricted = dict(builtins_obj)

        if DISALLOWED_MODULES:
            original_import = restricted.get("__import__")

            def _restricted_import(
                name, globals=None, locals=None, fromlist=(), level=0
            ):
                for blocked in DISALLOWED_MODULES:
                    if name == blocked or name.startswith(blocked + "."):
                        raise ImportError(
                            f"Import of '{{name}}' is blocked by RLM policy"
                        )
                if original_import is None:
                    raise ImportError("Import mechanism unavailable")
                return original_import(name, globals, locals, fromlist, level)

            restricted["__import__"] = _restricted_import

        for builtin_name in DISALLOWED_BUILTINS:
            restricted.pop(builtin_name, None)

        return restricted

    def ensure_fifo(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        os.mkfifo(path)

    for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):
        ensure_fifo(fifo_path)

    # Load filesystem context from file (written by setup_state)
    fs_root = None
    fs_metadata = {{}}
    allowed_paths = []
    def _get_stdlib_paths() -> list:
        paths = []
        try:
            config_paths = sysconfig.get_paths()
        except Exception:
            return paths
        for key in ("stdlib", "platstdlib"):
            value = config_paths.get(key)
            if value:
                paths.append(value)
        return paths

    if Path(CONTEXT_FILE).exists():
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            context = json.load(f)
            fs_root = context.get("fs_root")
            fs_metadata = context.get("fs_metadata") or {{}}
            allowed_paths = context.get("allowed_paths") or []
            for stdlib_path in _get_stdlib_paths():
                if stdlib_path not in allowed_paths:
                    allowed_paths.append(stdlib_path)

    if fs_root:
        os.chdir(fs_root)
        jail = FilesystemJail(
            fs_root,
            allowed_paths=allowed_paths,
            disallowed_modules=DISALLOWED_MODULES,
            disallowed_builtins=DISALLOWED_BUILTINS,
        )
        jail.install()

    # Initialize answer structure
    answer = {{"ready": False, "content": ""}}
    if Path(ANSWER_FILE).exists():
        with open(ANSWER_FILE, "r", encoding="utf-8") as f:
            answer = json.load(f)

    ROOT_TOOL_URL = os.environ.get("RLM_ROOT_TOOL_URL", "")
    ROOT_TOOL_SERIALIZATION = os.environ.get("RLM_ROOT_TOOL_SERIALIZATION", "pickle")
    ROOT_TOOL_NAMES_RAW = os.environ.get("RLM_ROOT_TOOL_NAMES", "[]")
    try:
        ROOT_TOOL_NAMES = json.loads(ROOT_TOOL_NAMES_RAW)
    except Exception:
        ROOT_TOOL_NAMES = []

    def _call_root_tool(tool_name: str, args: tuple, kwargs: dict):
        if not ROOT_TOOL_URL:
            raise RuntimeError("Root tool URL not configured")
        if ROOT_TOOL_SERIALIZATION != "pickle":
            raise RuntimeError("Only pickle serialization is supported")

        args_payload = base64.b64encode(pickle.dumps(args)).decode("ascii")
        kwargs_payload = base64.b64encode(pickle.dumps(kwargs)).decode("ascii")
        payload = {{
            "tool_name": tool_name,
            "serialization": "pickle",
            "args": args_payload,
            "kwargs": kwargs_payload,
        }}

        resp = requests.post(
            ROOT_TOOL_URL,
            json=payload,
            timeout=SUB_LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("print_lines"):
            for line in data["print_lines"]:
                print(line)
        if data.get("error"):
            raise RuntimeError(data["error"])
        return pickle.loads(base64.b64decode(data.get("result", "")))

    def _make_root_tool(name: str):
        def _tool(*args, **kwargs):
            return _call_root_tool(name, args, kwargs)

        _tool.__name__ = name
        return _tool

    restricted_builtins = _build_restricted_builtins()
    extra_data = fs_root

    # Persistent execution namespace
    namespace: dict[str, object] = {{
        "__name__": "__main__",
        "__builtins__": restricted_builtins,
        "extra_data": extra_data,
        "fs_metadata": fs_metadata,
        "answer": answer,
    }}
    for tool_name in ROOT_TOOL_NAMES:
        namespace[tool_name] = _make_root_tool(tool_name)

    # Signal ready
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
        seq = request.get("seq", 0)  # Sequence number for request/response matching
        execution_count += 1
        
        result = {{
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": None,
            "execution_count": execution_count,
            "seq": seq,  # Echo back sequence number for framework to verify
            "answer": namespace.get("answer", {{"ready": False, "content": ""}}),
        }}
        
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                module_ast = ast.parse(code, mode="exec")
                body = list(module_ast.body)
                trailing_expr = None
                if body and isinstance(body[-1], ast.Expr):
                    trailing_expr = body.pop()
                if body:
                    exec_module = ast.Module(body=body, type_ignores=[])
                    exec(compile(exec_module, "<cell>", "exec"), namespace, namespace)
                if trailing_expr is not None:
                    value = eval(
                        compile(ast.Expression(trailing_expr.value), "<cell>", "eval"),
                        namespace,
                        namespace,
                    )
                    if value is not None:
                        result["result"] = repr(value)
        except Exception:
            result["status"] = "error"
            result["result"] = traceback.format_exc()
        
        result["stdout"] = stdout_buffer.getvalue()
        result["stderr"] = stderr_buffer.getvalue()
        result["answer"] = namespace.get("answer", {{"ready": False, "content": ""}})
        
        # Save answer to file for persistence
        with open(ANSWER_FILE, "w", encoding="utf-8") as f:
            json.dump(result["answer"], f)
        
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


def _render_worker_script(paths: RLMWorkerPaths) -> str:
    filesystem_jail_code = textwrap.dedent(inspect.getsource(rlm_jail_module))
    return _RLM_WORKER_SCRIPT_TEMPLATE.format(
        filesystem_jail_code=filesystem_jail_code,
        command_fifo=paths.command_fifo,
        response_fifo=paths.response_fifo,
        ready_flag=paths.ready_flag,
        context_file=paths.context_file,
        answer_file=paths.answer_file,
    )


# System prompt for sub-LLMs (called via llm_batch)
_SUB_LLM_SYSTEM_PROMPT = """You are a sub-agent being called by a parent model to help with a specific task.
Answer the query directly and concisely. Put your final answer inside \\boxed{}.

Example: If asked "What is 2+2?", respond with reasoning then \\boxed{4}."""


# System prompt for RLM
_RLM_SYSTEM_PROMPT = """You are operating in a Recursive Language Model (RLM) environment - an iterative Python REPL where you explore data step by step.

## Critical: This is an ITERATIVE environment

You will write code, see its output, then write more code based on what you learned. **Do NOT try to solve everything in one tool call.** Each tool call executes and returns output before you continue.

Use the `call_python_repl` tool to execute Python code. The REPL maintains state across calls. See the tool description for available variables and functions.

## Filesystem Context
{filesystem_summary}

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
"""


class BaseRLMExecutor:
    def __init__(self, env: "RLMEnv") -> None:
        self.env = env

    async def setup(self, state: State) -> None:
        raise NotImplementedError

    async def execute(self, payload: dict[str, Any], state: State) -> RLMExecResult:
        raise NotImplementedError

    async def read_answer(self, state: State) -> str:
        return ""

    async def recover_from_timeout(self, state: State) -> bool:
        return False

    async def cleanup(self, state: State) -> None:
        return None

    async def teardown(self) -> None:
        return None


class LocalRLMExecutor(BaseRLMExecutor):
    def __init__(self, env: "RLMEnv") -> None:
        super().__init__(env)
        self._sessions: dict[str, LocalRLMReplSession] = {}
        self._retained_dirs: set[str] = set()

    def create_rollout_dirs(self, state: State) -> None:
        session = self._get_or_create_session(state)
        state["rlm_rollout_dir"] = session.rollout_dir
        state["rlm_fs_root"] = session.fs_root
        state["rlm_control_dir"] = session.control_dir
        state["rlm_paths"] = session.paths.to_dict()

    async def setup(self, state: State) -> None:
        session = self._get_or_create_session(state)
        venv_path = await self._ensure_venv(session)
        session.venv_path = venv_path

        await self._write_local_files(session, state)
        await self._start_worker(state, session)

    async def execute(self, payload: dict[str, Any], state: State) -> RLMExecResult:
        session = self._get_session(state)
        if session.worker_process is None:
            raise vf.SandboxError() from Exception("RLM worker process not running")
        if session.worker_process.poll() is not None:
            raise vf.SandboxError() from Exception("RLM worker process not running")

        def _do_io() -> str:
            payload_json = json.dumps(payload)
            with open(
                session.paths.command_fifo, "w", encoding="utf-8"
            ) as command_file:
                command_file.write(payload_json)
            with open(
                session.paths.response_fifo, "r", encoding="utf-8"
            ) as response_file:
                return response_file.read()

        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(_do_io),
                timeout=self.env.code_execution_timeout,
            )
        except asyncio.TimeoutError as e:
            logger.warning(
                "Code execution timed out after %ss", self.env.code_execution_timeout
            )
            raise RLMCodeExecutionTimeout from e
        except Exception as e:
            raise vf.SandboxError() from e

        return RLMExecResult(stdout=raw, stderr="")

    async def read_answer(self, state: State) -> str:
        session = self._sessions.get(state.get("rollout_id", ""))
        if not session:
            return ""
        try:
            content = Path(session.paths.answer_file).read_text(encoding="utf-8")
            return json.loads(content).get("content", "")
        except Exception:
            return ""

    async def recover_from_timeout(self, state: State) -> bool:
        session = self._sessions.get(state.get("rollout_id", ""))
        if not session:
            logger.error("Cannot recover from timeout: missing local session")
            return False
        try:
            self._stop_worker(session)
            await self._write_local_files(session, state)
            await self._start_worker(state, session)
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
        self._stop_worker(session)
        if state.get("retain_filesystem_after_rollout", False):
            self._retained_dirs.add(session.rollout_dir)
        else:
            await asyncio.to_thread(shutil.rmtree, session.rollout_dir, True)

    async def teardown(self) -> None:
        if self._sessions:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            for session in sessions:
                try:
                    self._stop_worker(session)
                finally:
                    if session.rollout_dir not in self._retained_dirs:
                        shutil.rmtree(session.rollout_dir, True)

    def _get_or_create_session(self, state: State) -> LocalRLMReplSession:
        rollout_id = state.get("rollout_id")
        if not rollout_id:
            raise ValueError("rollout_id must be set before creating local session")
        session = self._sessions.get(rollout_id)
        if session:
            return session
        rollout_dir = Path(tempfile.mkdtemp(prefix=f"rlm_rollout_{rollout_id}_"))
        fs_root = rollout_dir / "rlm_fs"
        control_dir = rollout_dir / "rlm_control"
        fs_root.mkdir(parents=True, exist_ok=True)
        control_dir.mkdir(parents=True, exist_ok=True)
        paths = _build_worker_paths(str(control_dir))
        session = LocalRLMReplSession(
            rollout_id=rollout_id,
            rollout_dir=str(rollout_dir),
            paths=paths,
            fs_root=str(fs_root),
            control_dir=str(control_dir),
        )
        self._sessions[rollout_id] = session
        return session

    def _get_session(self, state: State) -> LocalRLMReplSession:
        rollout_id = state.get("rollout_id")
        if not rollout_id or rollout_id not in self._sessions:
            raise vf.SandboxError() from Exception("Local session not initialized")
        return self._sessions[rollout_id]

    async def _ensure_venv(self, session: LocalRLMReplSession) -> str:
        venv_path = os.path.join(session.fs_root, ".venv")
        await self._create_venv(venv_path, force=True)
        await self._install_packages(venv_path)
        return venv_path

    async def _create_venv(self, venv_path: str, force: bool) -> None:
        if force and os.path.exists(venv_path):
            await asyncio.to_thread(shutil.rmtree, venv_path, True)
        args = ["uv", "venv", venv_path]
        await self._run_uv_command(args, self.env._compute_install_wait_seconds())

    async def _install_packages(self, venv_path: str) -> None:
        packages = ["requests"]
        extras = [p.strip() for p in self.env.pip_install_packages.split() if p.strip()]
        packages.extend(extras)
        if not packages:
            return
        python_path = self._venv_python(venv_path)
        args = ["uv", "pip", "install", "-q", "--python", python_path]
        args.extend(packages)
        await self._run_uv_command(args, self.env._compute_install_wait_seconds())

    async def _run_uv_command(self, args: list[str], timeout: int) -> None:
        def _run() -> subprocess.CompletedProcess:
            return subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        try:
            result = await asyncio.to_thread(_run)
        except FileNotFoundError:
            raise vf.SandboxError() from RuntimeError(
                "uv not found on PATH; local execution requires uv installed"
            )
        except subprocess.TimeoutExpired:
            raise vf.SandboxError() from RuntimeError(
                f"uv command timed out after {timeout} seconds"
            )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            raise vf.SandboxError() from Exception(
                f"uv command failed: {' '.join(args)}\nstdout: {stdout}\nstderr: {stderr}"
            )

    def _venv_python(self, venv_path: str) -> str:
        if os.name == "nt":
            return os.path.join(venv_path, "Scripts", "python.exe")
        return os.path.join(venv_path, "bin", "python")

    async def _write_local_files(
        self, session: LocalRLMReplSession, state: State
    ) -> None:
        Path(session.control_dir).mkdir(parents=True, exist_ok=True)
        allowed_paths = [
            session.paths.command_fifo,
            session.paths.response_fifo,
            session.paths.ready_flag,
            session.paths.context_file,
            session.paths.answer_file,
        ]
        context = {
            "fs_root": state.get("rlm_fs_root"),
            "fs_metadata": state.get("rlm_fs_metadata") or {},
            "allowed_paths": allowed_paths,
        }
        Path(session.paths.context_file).write_text(
            json.dumps(context), encoding="utf-8"
        )
        Path(session.paths.answer_file).write_text(
            json.dumps({"ready": False, "content": ""}), encoding="utf-8"
        )

    async def _start_worker(self, state: State, session: LocalRLMReplSession) -> None:
        if not session.venv_path:
            raise vf.SandboxError() from Exception("Local venv not initialized")
        worker_script = _render_worker_script(session.paths)
        Path(session.paths.worker_path).write_text(worker_script, encoding="utf-8")

        env_vars = os.environ.copy()
        env_vars.update(
            {
                "RLM_INTERCEPTION_URL": state["interception_url"],
                "RLM_ROOT_TOOL_URL": state.get("root_tool_url", ""),
                "RLM_ROOT_TOOL_NAMES": json.dumps(self.env.root_tool_names),
                "RLM_ROOT_TOOL_SERIALIZATION": self.env.root_tool_serialization,
                "RLM_SUB_MODEL": self.env.sub_model or state.get("model", ""),
                "RLM_MAX_SUB_LLM_PARALLELISM": str(self.env.max_sub_llm_parallelism),
                "RLM_SUB_LLM_STAGGER_MS": str(self.env.sub_llm_stagger_ms),
                "RLM_SUB_LLM_STAGGER_JITTER_MS": str(
                    self.env.sub_llm_stagger_jitter_ms
                ),
                "RLM_SUB_LLM_TIMEOUT": str(self.env.sub_llm_timeout),
                "RLM_DISALLOWED_MODULES": self.env.disallowed_modules,
                "RLM_DISALLOWED_BUILTINS": self.env.disallowed_builtins,
            }
        )

        python_path = self._venv_python(session.venv_path)
        with open(session.paths.log_file, "a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [python_path, "-u", session.paths.worker_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env_vars,
                start_new_session=True,
            )
        session.worker_process = process

        await self._wait_for_ready(session)

    async def _wait_for_ready(self, session: LocalRLMReplSession) -> None:
        max_wait_seconds = self.env.max_startup_wait_seconds
        start = perf_counter()
        while True:
            if Path(session.paths.ready_flag).exists():
                return
            if session.worker_process and session.worker_process.poll() is not None:
                log_tail = ""
                try:
                    log_tail = Path(session.paths.log_file).read_text(encoding="utf-8")[
                        -2000:
                    ]
                except Exception:
                    pass
                raise vf.SandboxError() from Exception(
                    f"RLM worker exited before ready. Log tail:\n{log_tail}"
                )
            if perf_counter() - start > max_wait_seconds:
                raise vf.SandboxError() from Exception("RLM worker failed to start")
            await asyncio.sleep(0.1)

    def _stop_worker(self, session: LocalRLMReplSession) -> None:
        if not session.worker_process:
            return
        process = session.worker_process
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except Exception:
                pass
        session.worker_process = None


class RLMEnv(vf.StatefulToolEnv):
    """
    Recursive Language Model Environment.

    Extends StatefulToolEnv to provide a Python REPL environment where the model can:
    - Interact with large input data stored in a working directory (filesystem)
    - Make recursive sub-LLM calls via `llm_batch()`
    - Return final answers via an `answer` variable

    Architecture:
    - REPL loop runs in the framework (standard MultiTurnEnv pattern)
    - Code execution runs locally in a persistent Python worker
    - Sub-LLM calls from worker code are intercepted via HTTP proxy

    Works with any dataset that has a normal prompt. Input data can optionally
    be provided via info[context_dir_key] (directory path) or info[context_key]
    (legacy builtin data written to a file).

    Args:
        tools: List of tools shared by both the root REPL and sub-LLMs.
                   These are added first in the tool documentation order.
        root_tools: List of tools available only to the root REPL.
                   The root model can call these inside the REPL as Python functions.
        sub_tools: List of tools available only to sub-LLMs.
                   Sub-LLMs access these via standard tool calling.
        sub_model: Model to use for sub-LLM calls (defaults to same as root model)
        (Ordering) The root tool list is: fixed tools (e.g. llm_batch), then `tools`,
                   then `root_tools`. The sub-LLM tool list is: `tools`, then `sub_tools`.
                   Each list is deduplicated by tool name. If two different tools
                   share a name within a list, initialization raises an error.
        sub_tool_max_turns: Maximum tool-calling turns for sub-LLM calls (default: 5)
        max_iterations: Maximum REPL iterations before stopping (maps to max_turns)
        max_output_length: Maximum length of code execution output
        max_sub_llm_parallelism: Maximum number of concurrent sub-LLM calls
        sub_llm_stagger_ms: Optional fixed per-call stagger delay (ms) within llm_batch.
        sub_llm_stagger_jitter_ms: Optional random jitter (ms) added to stagger delay.
        context_key: Key in info containing legacy context data (default: "context")
        context_dir_key: Key in info containing directory path (default: "context_dir")
        system_prompt: Custom system prompt (default: RLM standard prompt)
        interception_host: Optional hostname/IP for interception server (default: 127.0.0.1)
        interception_port: Port for interception server (default: 8766)
        pip_install_packages: Space-separated packages to install in addition to requests
                   (default: "")
        max_startup_wait_seconds: Maximum seconds to wait for worker startup (default: 120)
        include_sub_llm_in_trajectory: Whether to include sub-LLM calls as trajectory steps.
                   When True (default), sub-LLM turns are prepended to the trajectory as
                   TrajectoryStep objects with tokens, enabling training on sub-LLM calls.
                   When False, sub-LLM calls happen but are not stored.
        context_warning_threshold: Fraction of max_seq_len at which to warn the model
                   to finish (default: 0.80). Only active if max_seq_len is set.
        code_execution_timeout: Timeout in seconds for code execution (default: 120).
                   This is longer than the default command timeout to allow for
                   llm_batch calls which can take several minutes.
        abort_on_code_timeout: If True, abort the rollout when code execution times out.
                   If False (default), return an error message to the model so it can
                   try a more efficient approach.
        retain_filesystem_after_rollout: If True, keep filesystem after rollout.
        filesystem_copy_max_bytes: Optional max bytes for context directory copy.
        disallowed_modules: Space-separated module names that user code may not import.
        disallowed_builtins: Space-separated builtin names removed from user code execution.
        **kwargs: Additional arguments passed to StatefulToolEnv
    """

    def __init__(
        self,
        tools: list[Callable] | None = None,
        root_tools: list[Callable] | None = None,
        sub_model: str | None = None,
        sub_tools: list[Callable] | None = None,
        sub_tool_max_turns: int = 5,
        max_iterations: int = 50,
        max_output_length: int = 8192,
        max_sub_llm_parallelism: int = 5,
        sub_llm_stagger_ms: int = 200,
        sub_llm_stagger_jitter_ms: int = 50,
        context_key: str = "context",
        context_dir_key: str = "context_dir",
        system_prompt: str | None = None,
        interception_host: str | None = None,
        interception_port: int = 8766,
        pip_install_packages: str = "",
        max_startup_wait_seconds: int = 120,
        include_sub_llm_in_trajectory: bool = True,
        context_warning_threshold: float = 0.80,
        code_execution_timeout: int = 120,
        abort_on_code_timeout: bool = False,
        retain_filesystem_after_rollout: bool = False,
        filesystem_copy_max_bytes: int | None = 1_000_000_000,
        disallowed_modules: str = "",
        disallowed_builtins: str = "",
        rubric: Rubric | None = None,
        **kwargs,
    ):
        if tools is None and "tools" in kwargs:
            tools = kwargs.pop("tools")
        elif tools is not None and "tools" in kwargs:
            raise ValueError("Tools were provided twice: use tools=... only once.")

        if root_tools is None and "root_tools" in kwargs:
            root_tools = kwargs.pop("root_tools")
        elif root_tools is not None and "root_tools" in kwargs:
            raise ValueError(
                "root_tools were provided twice: use root_tools=... only once."
            )

        if sub_tools is None and "sub_tools" in kwargs:
            sub_tools = kwargs.pop("sub_tools")
        elif sub_tools is not None and "sub_tools" in kwargs:
            raise ValueError(
                "sub_tools were provided twice: use sub_tools=... only once."
            )

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
        self.pip_install_packages = pip_install_packages
        self.max_startup_wait_seconds = max_startup_wait_seconds
        self.include_sub_llm_in_trajectory = include_sub_llm_in_trajectory
        self.context_warning_threshold = context_warning_threshold
        self.code_execution_timeout = code_execution_timeout
        self.abort_on_code_timeout = abort_on_code_timeout
        self.disallowed_modules = disallowed_modules
        self.disallowed_builtins = disallowed_builtins
        self.retain_filesystem_after_rollout = retain_filesystem_after_rollout
        self.filesystem_copy_max_bytes = filesystem_copy_max_bytes
        self._interception_bind_host = self.interception_host
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
        self.sub_oai_tools = [convert_func_to_oai_tool(tool) for tool in self.sub_tools]
        self.root_tool_doc_funcs: list[Callable] = []
        for tool in self.root_tools:
            name = _tool_display_name(tool)
            if name in _FIXED_REPL_TOOL_NAMES:
                self.root_tool_doc_funcs.append(
                    self._build_fixed_root_tool_schema(name)
                )
            else:
                self.root_tool_doc_funcs.append(tool)
        self.root_oai_tools = [
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

        # Active rollout tracking for sub-LLM request routing
        self.active_rollouts: dict[str, dict[str, Any]] = {}

        super().__init__(
            tools=[],
            max_turns=max_iterations,
            rubric=rubric,
            **kwargs,
        )
        self.add_rubric(RLMMonitorRubric(root_tool_names=self.root_tool_names))
        self._executor = LocalRLMExecutor(self)

        # Add the Python REPL tool (state is injected via update_tool_args)
        self.add_tool(self.call_python_repl, args_to_skip=["state"])

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
                "code_execution_timeout=%s is low; sub-LLM calls may be unreliable",
                code_timeout,
            )

        return api_timeout, worker_timeout

    def _build_fixed_root_tools(self) -> list[Callable]:
        """Return the fixed root REPL tools (non-overridable)."""

        async def llm_batch(prompts: list[str]) -> list[str]:
            """
            Make multiple sub-LLM calls in parallel.

            Args:
                prompts: List of prompt strings (recommended). Message dicts or lists
                    of message dicts are also accepted for compatibility.

            Returns:
                List of response contents in the same order as the input prompts.
            """
            context = self._root_tool_context_var.get()
            if context is None:
                raise RuntimeError(
                    "llm_batch called outside of a tool request context."
                )
            results, _ = await self._root_llm_batch(context, prompts)
            return results

        llm_batch.__name__ = "llm_batch"
        return [llm_batch]

    def _build_fixed_root_tool_schema(self, name: str) -> Callable:
        """Return a schema-only stub for fixed root tools."""
        if name == "llm_batch":

            def llm_batch(prompts: list[str]) -> list[str]:
                """Make multiple sub-LLM calls in parallel."""
                raise RuntimeError("llm_batch schema stub should not be executed.")

            llm_batch.__name__ = "llm_batch"
            return llm_batch
        raise ValueError(f"Unsupported fixed tool schema: {name}")

    def _compute_install_wait_seconds(self) -> int:
        """Estimate how long to wait for pip installs based on package count."""
        packages = [p.strip() for p in self.pip_install_packages.split() if p.strip()]
        package_count = len(packages) + 1  # Always includes requests
        estimated_seconds = 30 * package_count
        return max(self.max_startup_wait_seconds, estimated_seconds)

    def _generate_packages_documentation(self) -> str:
        """Generate documentation for installed packages to include in system prompt."""
        if not self.pip_install_packages:
            return ""

        # Parse package names from pip_install_packages string
        packages = [p.strip() for p in self.pip_install_packages.split() if p.strip()]
        if not packages:
            return ""

        lines = ["\n## Installed Packages\n"]
        lines.append(
            "The following Python packages are pre-installed in the REPL environment:\n"
        )
        for pkg in packages:
            lines.append(f"- `{pkg}`")
        lines.append("")
        lines.append("You can import and use these packages directly in your code.\n")

        return "\n".join(lines)

    def _generate_sub_tools_documentation(self) -> str:
        """Generate documentation for sub-agent tools to include in system prompt."""
        if not self.sub_tools:
            return ""

        lines = ["\n## Sub-LLM Tools\n"]
        lines.append(
            "The sub-LLMs called via `llm_batch()` have access to the following tools:\n"
        )

        for oai_tool in self.sub_oai_tools:
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
        lines.append(
            "The root model can call the following tools inside the Python REPL:\n"
        )

        for oai_tool in self.root_oai_tools:
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

        lines.append(
            "These tools run on the host and are only accessible from within the REPL."
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

    def _generate_filesystem_summary(
        self, *, fs_root: str, metadata: dict[str, Any], has_data: bool
    ) -> str:
        """Generate a concise summary of filesystem context for the system prompt."""
        lines = [f"Working directory: {fs_root}"]
        if has_data:
            file_count = metadata.get("file_count")
            total_size = metadata.get("total_size", metadata.get("total_bytes"))
            if file_count is not None:
                lines.append(f"File count: {file_count}")
            if total_size is not None:
                lines.append(f"Total size (bytes): {total_size}")
        else:
            lines.append(
                "No extra data was provided. The working directory exists but is empty."
            )
            lines.append("You can still use this directory for any files you create.")
        lines.append("Never access files or directories outside the working directory.")
        return "\n".join(lines)

    @staticmethod
    def _extract_tokens(response: Any) -> tuple[int, int]:
        """Extract prompt and completion tokens from response usage."""
        usage = getattr(response, "usage", None)
        if not usage:
            return 0, 0
        return (
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    @staticmethod
    def _normalize_sampling_args(sampling_args: dict[str, Any]) -> dict[str, Any]:
        """Normalize sampling args to match main model behavior."""
        if "max_tokens" in sampling_args:
            if sampling_args["max_tokens"] is None:
                sampling_args.pop("max_tokens")
            else:
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
        if (
            "max_completion_tokens" in sampling_args
            and sampling_args["max_completion_tokens"] is None
        ):
            sampling_args.pop("max_completion_tokens")
        return {k: v for k, v in sampling_args.items() if v is not None}

    def _prepare_sub_llm_sampling_args(
        self, state: State, *, interleaved: bool
    ) -> dict[str, Any]:
        sampling_args = dict(state.get("sampling_args") or {})
        extra_body = sampling_args.get("extra_body")
        if isinstance(extra_body, dict):
            sampling_args["extra_body"] = dict(extra_body)
        sampling_args = self._normalize_sampling_args(sampling_args)
        if interleaved:
            return prepare_sampling_args_for_token_prompts(sampling_args)
        return sampling_args

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
            return {
                "role": "tool",
                "content": f"Error: {e}",
                "tool_call_id": tool_call_id,
            }

    def _normalize_message_content(
        self, messages: ChatMessages | list[dict[str, Any]]
    ) -> ChatMessages:
        """Normalize message content fields to formats the API accepts.

        The API expects content to be: string, array of objects, or None.
        Handles several malformed cases:
        1. Content is a nested message dict (has 'role' and 'content' keys) - extract inner content
        2. Content is a content part object (has 'type' key) - wrap in array
        """
        normalized: ChatMessages = []
        for msg in messages:
            msg_copy: dict[str, Any] = cast(dict[str, Any], msg).copy()
            content = msg_copy.get("content")

            if content is not None and isinstance(content, dict):
                # Check if content is a nested message dict (has 'role' and 'content' keys)
                # This happens when model passes message dicts to llm_batch instead of strings
                if "role" in content and "content" in content:
                    msg_copy["content"] = content["content"]
                elif "type" in content:
                    # Content part object (e.g. {"type": "text", "text": "..."}) - wrap in array
                    msg_copy["content"] = [content]
                else:
                    # Unknown dict structure - try wrapping in array as fallback
                    msg_copy["content"] = [content]
            normalized.append(cast(ChatMessage, msg_copy))
        return normalized

    async def _call_sub_llm_api(
        self,
        state: State,
        client: Any,
        model: str,
        messages: ChatMessages,
        tools: list | None = None,
    ) -> Any | None:
        """Make a single sub-LLM API call matching main-model request mode."""
        normalized_messages = self._normalize_message_content(messages)
        sampling_args = self._prepare_sub_llm_sampling_args(
            state, interleaved=self.interleaved_rollouts
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
            "tools": tools,
        }

        try:
            if self.interleaved_rollouts:
                extra_body = sampling_args.pop("extra_body", {})
                prompt_ids = await tokenize_vllm(
                    client=client,
                    messages=normalized_messages,
                    tools=tools,
                    model=model,
                )
                payload = {
                    "model": model,
                    "messages": normalized_messages,
                    "tools": tools,
                    "tokens": prompt_ids,
                    **sampling_args,
                    **extra_body,
                }
                return await asyncio.wait_for(
                    client.post(
                        "/chat/completions/tokens",
                        body=payload,
                        cast_to=ChatCompletion,
                    ),
                    timeout=self.sub_llm_api_timeout,
                )
            payload = {
                "model": model,
                "messages": normalized_messages,
                "tools": tools,
                **sampling_args,
            }
            return await asyncio.wait_for(
                client.chat.completions.create(
                    **payload,
                ),
                timeout=self.sub_llm_api_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Sub-LLM API call timed out after {self.sub_llm_api_timeout}s"
            )
            return None
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

            prompt_tokens, completion_tokens = self._extract_tokens(response)
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
                state, client, model, current_messages, tools
            )
            if response is None:
                return self._make_timeout_result(
                    turns,
                    total_prompt_tokens,
                    total_completion_tokens,
                    tool_call_count,
                    num_turns,
                )

            prompt_tokens, completion_tokens = self._extract_tokens(response)
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
        response = await self._call_sub_llm_api(state, client, model, current_messages)
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
        prompt_tokens, completion_tokens = self._extract_tokens(response)

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
            if isinstance(prompt, dict):
                if "role" in prompt and "content" in prompt:
                    return [cast(ChatMessage, prompt)]
                raise ValueError(
                    "llm_batch prompt at index "
                    + str(index)
                    + " must be a string or message dict with 'role' and 'content'."
                )
            if isinstance(prompt, (list, tuple)):
                if all(isinstance(item, dict) for item in prompt):
                    return [cast(ChatMessage, item) for item in prompt]
                raise ValueError(
                    "llm_batch prompt at index "
                    + str(index)
                    + " must be a list of message dicts."
                )
            raise ValueError(
                "llm_batch prompt at index "
                + str(index)
                + " must be a string, message dict, or list of message dicts."
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
            tokens = meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0)
            tool_calls = meta.get("tool_call_count", 0)
            max_turns = meta.get("max_turns_reached", False)
            status = "⚠ max turns" if max_turns else "✓"
            summary_lines.append(
                f"  [{index}]: {tokens} tokens, {tool_calls} tool calls, "
                f"{elapsed:.2f}s {status}"
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
            cast(ChatMessage, {"role": "system", "content": _SUB_LLM_SYSTEM_PROMPT}),
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

    @vf.teardown
    async def teardown_executor(self):
        """Cleanup executor-level resources (e.g., local venv)."""
        await self._executor.teardown()

    # =========================================================================
    # State Management
    # =========================================================================

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict[str, Any]:
        """Inject state into call_python_repl tool args."""
        if tool_name == "call_python_repl":
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

        interception_url = (
            f"http://{self.interception_host}:{self.interception_port}"
            f"/rollout/{rollout_id}/v1/chat/completions"
        )
        root_tool_url = (
            f"http://{self.interception_host}:{self.interception_port}"
            f"/rollout/{rollout_id}/v1/rlm/tools"
        )

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

        fs_metadata = self._compute_fs_metadata(fs_root)
        state["rlm_fs_root"] = fs_root
        state["rlm_fs_source"] = fs_source
        state["rlm_fs_metadata"] = fs_metadata
        state["rlm_fs_has_data"] = fs_has_data
        state["retain_filesystem_after_rollout"] = self.retain_filesystem_after_rollout

        filesystem_summary = self._generate_filesystem_summary(
            fs_root=fs_root,
            metadata=fs_metadata,
            has_data=fs_has_data,
        )
        base_system_prompt = self.custom_system_prompt or _RLM_SYSTEM_PROMPT
        if "{filesystem_summary}" in base_system_prompt:
            # Use replace instead of format to avoid conflict with curly braces from Python code
            base_system_prompt = base_system_prompt.replace(
                "{filesystem_summary}", filesystem_summary
            )
        else:
            # If custom prompt doesn't have placeholder, prepend it
            base_system_prompt = f"{filesystem_summary}\n\n{base_system_prompt}"

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
        state["rlm_root_tools"] = [_tool_display_name(tool) for tool in self.root_tools]
        state["rlm_sub_tools"] = [_tool_display_name(tool) for tool in self.sub_tools]

        # 4. Prepare backend and start worker
        await self._executor.setup(state)

        state["rlm_worker_ready"] = True

        # Initialize context warning flag (feature enabled if max_seq_len is set)
        state["context_warning_sent"] = False

        # Initialize FIFO sequence counter for detecting stale responses
        state["_exec_seq"] = 0

        _ensure_rlm_metric_state(state)

        return state

    # =========================================================================
    # Code Execution
    # =========================================================================

    async def _recover_from_code_timeout(self, state: State) -> bool:
        """Attempt to recover from a code execution timeout via the active backend."""
        return await self._executor.recover_from_timeout(state)

    async def _execute_code(self, code: str, state: State) -> dict[str, Any]:
        """Execute code in worker and return result."""
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
                # Abort rollout immediately on timeout
                raise vf.SandboxError() from e
            recovered = await self._recover_from_code_timeout(state)
            recovery_note = (
                " The worker was restarted and the REPL state was reset."
                if recovered
                else " Failed to restart the worker; the REPL may be unusable."
            )
            # Return error to model so it can try more efficient code
            return {
                "status": "error",
                "stdout": "",
                "stderr": "",
                "result": (
                    f"Code execution timed out after {self.code_execution_timeout} seconds."
                    f"{recovery_note} Your code may be too slow - consider a more "
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

    # =========================================================================
    # REPL Tool
    # =========================================================================

    async def call_python_repl(self, code: str, state: Any) -> str:
        """
        Execute Python code in a persistent REPL environment.

        The REPL maintains state across calls and provides access to:

        - Files in the working directory (current working directory is the context root).
        - `extra_data`: The working directory path (string) for convenience.
        - `fs_metadata`: Metadata about the filesystem context (file_count, total_size).

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
        # Update current turn in rollout context for sub-LLM call tracking
        rollout_id = state.get("rollout_id")
        if rollout_id and rollout_id in self.active_rollouts:
            self.active_rollouts[rollout_id]["current_turn"] = state.get("turn", 0)
        # Time the full tool call execution
        execution_start = perf_counter()
        result = await self._execute_code(code, state)
        execution_time = perf_counter() - execution_start
        output = self._format_execution_output(result)

        # Track timing in state for metrics
        state.setdefault("tool_call_timings", []).append(
            {
                "turn": state.get("turn", 0),
                "execution_seconds": execution_time,
            }
        )
        _update_rlm_repl_metrics(state, execution_time)

        # Append execution time to output
        output += f"\n[Execution time: {execution_time:.2f}s]"

        # Check if answer is ready
        answer = result.get("answer", {})
        if answer.get("ready", False):
            state["final_answer"] = answer.get("content", "")
            logger.debug(f"Answer ready: {state['final_answer'][:100]}...")

        # Inject context limit warning if approaching limit
        if self.max_seq_len and not state.get("context_warning_sent"):
            # Get prompt token count from latest main-model trajectory response
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
                    f"{self.max_seq_len:,} tokens ({pct:.0%}). Please finalize your answer "
                    "soon by setting answer['ready'] = True."
                )

        return output

    async def add_trajectory_step(self, state: State, trajectory_step: TrajectoryStep):
        update_rlm_metrics_from_step(state, trajectory_step)
        await super().add_trajectory_step(state, trajectory_step)

    # =========================================================================
    # MultiTurnEnv Interface
    # =========================================================================

    async def get_prompt_messages(self, state: State) -> Messages:
        """Build prompt messages, adding system prompt with tool docs on first turn."""
        if len(state["trajectory"]) == 0:
            # First turn: add system prompt
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

            messages = list(prompt)
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                # Append packages and tool docs to existing system prompt
                messages[0] = {
                    "role": "system",
                    "content": (
                        messages[0]["content"]
                        + packages_docs
                        + root_tools_docs
                        + sub_tools_docs
                    ),
                }
            return cast(Messages, messages)
        else:
            # Subsequent turns: use parent implementation
            return await super().get_prompt_messages(state)

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
