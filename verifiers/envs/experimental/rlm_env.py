"""
Recursive Language Model (RLM) Environment.

Implements the RLM inference strategy where language models can decompose and
recursively interact with input data of unbounded length through REPL environments.

Based on: https://www.alexzhang.dev/blog/recursive-language-models

Architecture:
- REPL loop runs in the framework (MultiTurnEnv pattern)
- Sandbox is used only for code execution (persistent Python worker)
- Sub-LLM calls from sandbox code are intercepted via HTTP proxy

Key features:
- Works with any dataset that has a normal prompt
- Optional large input data can be provided in info["context"]
- Root model only sees query, not full input data (unless it peeks via code)
- Model can make recursive sub-LLM calls via llm_batch() function
- Final answer returned via answer variable
"""

import asyncio
import base64
import json
import logging
import sys
import textwrap
import time
import uuid
from time import perf_counter
from typing import Any, Callable, cast

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from aiohttp import web
from prime_sandboxes import CommandTimeoutError

import verifiers as vf
from verifiers.envs.sandbox_env import (
    SandboxCreationError,
    SandboxEnv,
    SandboxNotReadyError,
)
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, ModelResponse, State, TrajectoryStep
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.data_utils import extract_boxed_answer
from verifiers.utils.response_utils import (
    parse_is_truncated,
    parse_response_messages,
    parse_response_tokens,
)
from verifiers.utils.tool_utils import convert_func_to_oai_tool
from verifiers.utils.tunnel_utils import TunnelPool

logger = logging.getLogger(__name__)


class SubLLMTurn(TypedDict):
    """A single turn in a sub-LLM call (used by RLMEnv)."""

    prompt_messages: list[dict]  # Messages before this LLM call
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


# Worker script that runs inside the sandbox - handles code execution only
# The REPL loop is managed by the framework, not this script
_RLM_WORKER_SCRIPT = textwrap.dedent(
    '''
    import ast
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    import requests

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
    SANDBOX_TIMEOUT = int(os.environ.get("RLM_SANDBOX_TIMEOUT", "120"))

    def ensure_fifo(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        os.mkfifo(path)

    for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):
        ensure_fifo(fifo_path)

    # Load extra_data from file (written by setup_state)
    extra_data = None
    if Path(CONTEXT_FILE).exists():
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            _full_context = json.load(f)
            extra_data = _full_context.get("input_data")

    # Initialize answer structure
    answer = {{"ready": False, "content": ""}}
    if Path(ANSWER_FILE).exists():
        with open(ANSWER_FILE, "r", encoding="utf-8") as f:
            answer = json.load(f)

    def _single_llm_call(prompt: str, batch_id: str, **kwargs) -> dict:
        """Make a single sub-LLM call via interception server.
        
        Returns a dict with 'content' and 'metadata' keys (including 'elapsed_seconds').
        """
        from time import perf_counter
        import uuid
        start_time = perf_counter()
        
        if not INTERCEPTION_URL:
            return {{
                "content": "Error: Sub-LLM interception URL not configured",
                "metadata": {{"error": True, "elapsed_seconds": 0.0}},
            }}
        
        try:
            request_id = uuid.uuid4().hex[:8]
            payload = {{
                "model": SUB_MODEL or "default",
                "messages": [{{"role": "user", "content": prompt}}],
                "_batch_id": batch_id,
                "_request_id": request_id,
            }}
            # Add any extra kwargs
            for k, v in kwargs.items():
                if k not in ("model", "messages", "_batch_id", "_request_id"):
                    payload[k] = v
            
            resp = requests.post(
                INTERCEPTION_URL,
                json=payload,
                timeout=SUB_LLM_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{{}}])[0].get("message", {{}}).get("content", "")
            metadata = data.get("_rlm_metadata", {{}})
            elapsed = perf_counter() - start_time
            metadata["elapsed_seconds"] = elapsed
            return {{"content": content, "metadata": metadata}}
        except Exception as e:
            elapsed = perf_counter() - start_time
            return {{
                "content": f"Error in sub-LLM call: {{e}}",
                "metadata": {{"error": True, "elapsed_seconds": elapsed}},
            }}

    def llm_batch(prompts: list, **kwargs) -> list:
        """
        Make multiple sub-LLM calls in parallel.
        
        Prints a summary of each call's metadata (including timing), then returns the list of responses.
        
        Parallelism is controlled by RLM_MAX_SUB_LLM_PARALLELISM.
        Sandbox timeout is available via SANDBOX_TIMEOUT env var.
        
        Args:
            prompts: List of prompts for the sub-LLMs
            **kwargs: Additional arguments applied to all calls
        
        Returns:
            List of response contents in the same order as the input prompts
        """
        from time import perf_counter
        import uuid
        batch_start = perf_counter()
        batch_id = uuid.uuid4().hex[:8]
        with ThreadPoolExecutor(max_workers=MAX_SUB_LLM_PARALLELISM) as executor:
            futures = [executor.submit(_single_llm_call, p, batch_id, **kwargs) for p in prompts]
            results = [f.result() for f in futures]
        batch_elapsed = perf_counter() - batch_start
        
        # Print metadata summary with timing
        print(f"llm_batch: {{len(results)}} call(s) in {{batch_elapsed:.2f}}s")
        for i, r in enumerate(results):
            meta = r.get("metadata", {{}})
            elapsed = meta.get("elapsed_seconds", 0.0)
            if meta.get("error"):
                print(f"  [{{i}}]: error ({{elapsed:.2f}}s)")
            else:
                tokens = meta.get("prompt_tokens", 0) + meta.get("completion_tokens", 0)
                tool_calls = meta.get("tool_call_count", 0)
                max_turns = meta.get("max_turns_reached", False)
                status = "⚠ max turns" if max_turns else "✓"
                print(f"  [{{i}}]: {{tokens}} tokens, {{tool_calls}} tool calls, {{elapsed:.2f}}s {{status}}")
        
        # Return just the content
        return [r.get("content", "") for r in results]

    # Persistent execution namespace
    namespace: dict[str, object] = {{
        "__name__": "__main__",
        "extra_data": extra_data,
        "answer": answer,
        "llm_batch": llm_batch,
    }}

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
    '''
)


_RLM_START_COMMAND_TEMPLATE = textwrap.dedent(
    """
    bash -lc '
    set -euo pipefail

    command_fifo="{command_fifo}"
    response_fifo="{response_fifo}"
    ready_flag="{ready_flag}"
    worker_path="{worker_path}"

    rm -f "$command_fifo" "$response_fifo" "$ready_flag"

    pip install -q requests {pip_install_packages}

    # Write worker script but do NOT start it yet
    # Worker will be started by setup_state after context/env vars are set
    python - <<'PY'
import base64
from pathlib import Path

Path("{worker_path}").write_bytes(base64.b64decode("{worker_b64}"))
PY

    tail -f /dev/null
    '
    """
)


def _make_ready_wait_script(ready_flag: str, max_wait_seconds: int) -> str:
    """Generate a ready wait script with configurable timeout."""
    # Each iteration sleeps 0.05 seconds, so calculate iterations needed
    iterations = max(1, int(max_wait_seconds / 0.05))
    return textwrap.dedent(
        f"""
        bash -lc '
        for i in $(seq 1 {iterations}); do
          if [ -f "{ready_flag}" ]; then
            exit 0
          fi
          sleep 0.05
        done
        echo "RLM worker failed to start" >&2
        exit 1
        '
        """
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

## Input Data Metadata
{metadata_summary}

## Workflow

**Step 1: Explore the data**
```python
print(type(extra_data))
print(extra_data[:500] if isinstance(extra_data, str) else extra_data[:3])
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


class RLMEnv(SandboxEnv):
    """
    Recursive Language Model Environment.

    Extends SandboxEnv to provide a Python REPL environment where the model can:
    - Interact with large input data stored as a variable (`extra_data`)
    - Make recursive sub-LLM calls via `llm_batch()`
    - Return final answers via an `answer` variable

    Architecture:
    - REPL loop runs in the framework (standard MultiTurnEnv pattern)
    - Sandbox is used only for code execution (persistent Python worker)
    - Sub-LLM calls from sandbox code are intercepted via HTTP proxy

    Works with any dataset that has a normal prompt. Input data can optionally
    be provided in info[context_key] for large data that shouldn't be in the prompt.

    Args:
        sub_model: Model to use for sub-LLM calls (defaults to same as root model)
        sub_tools: List of Python functions that sub-LLMs can use as tools.
                   These tools are NOT available to the root model.
        sub_tool_max_turns: Maximum tool-calling turns for sub-LLM calls (default: 5)
        max_iterations: Maximum REPL iterations before stopping (maps to max_turns)
        max_output_length: Maximum length of code execution output
        max_sub_llm_parallelism: Maximum number of concurrent sub-LLM calls
        context_key: Key in info containing optional input data (default: "context")
        system_prompt: Custom system prompt (default: RLM standard prompt)
        interception_host: Optional hostname/IP for interception server (auto-tunneled if not set)
        interception_port: Port for interception server (default: 8766)
        pip_install_packages: Space-separated packages to install (default: "requests")
        max_startup_wait_seconds: Maximum seconds to wait for worker startup (default: 30)
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
        **kwargs: Additional arguments passed to SandboxEnv
    """

    # Worker file paths
    _WORKER_PATH = "/tmp/rlm_worker.py"
    _COMMAND_FIFO = "/tmp/rlm_cmd"
    _RESPONSE_FIFO = "/tmp/rlm_res"
    _READY_FLAG = "/tmp/rlm_ready"
    _CONTEXT_FILE = "/tmp/rlm_context.json"
    _ANSWER_FILE = "/tmp/rlm_answer.json"

    def __init__(
        self,
        sub_model: str | None = None,
        sub_tools: list[Callable] | None = None,
        sub_tool_max_turns: int = 5,
        max_iterations: int = 50,
        max_output_length: int = 8192,
        max_sub_llm_parallelism: int = 5,
        context_key: str = "context",
        system_prompt: str | None = None,
        interception_host: str | None = None,
        interception_port: int = 8766,
        pip_install_packages: str = "",
        max_startup_wait_seconds: int = 120,
        include_sub_llm_in_trajectory: bool = True,
        context_warning_threshold: float = 0.80,
        code_execution_timeout: int = 120,
        abort_on_code_timeout: bool = False,
        rubric: Rubric | None = None,
        **kwargs,
    ):
        self.sub_model = sub_model
        self.sub_tools = sub_tools or []
        self.sub_tool_max_turns = sub_tool_max_turns
        self.max_iterations = max_iterations
        self.max_output_length = max_output_length
        self.max_sub_llm_parallelism = max_sub_llm_parallelism
        self.context_key = context_key
        self.custom_system_prompt = system_prompt
        self.interception_host = interception_host
        self.interception_port = interception_port
        self.pip_install_packages = pip_install_packages
        self.max_startup_wait_seconds = max_startup_wait_seconds
        self.include_sub_llm_in_trajectory = include_sub_llm_in_trajectory
        self.context_warning_threshold = context_warning_threshold
        self.code_execution_timeout = code_execution_timeout
        self.abort_on_code_timeout = abort_on_code_timeout
        # Server-side timeout for LLM API calls (shorter than sandbox HTTP timeout)
        # This ensures server responds before sandbox worker's HTTP request times out
        self.sub_llm_api_timeout = max(10, int(self.code_execution_timeout * 0.8))

        # Convert sub_tools to OAI format (reusing existing infrastructure)
        self.sub_oai_tools = [convert_func_to_oai_tool(tool) for tool in self.sub_tools]
        self.sub_tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool
            for tool in self.sub_tools
        }

        # Build worker script
        worker_script = _RLM_WORKER_SCRIPT.format(
            command_fifo=self._COMMAND_FIFO,
            response_fifo=self._RESPONSE_FIFO,
            ready_flag=self._READY_FLAG,
            context_file=self._CONTEXT_FILE,
            answer_file=self._ANSWER_FILE,
        )
        worker_b64 = base64.b64encode(worker_script.encode("utf-8")).decode("utf-8")

        start_command = _RLM_START_COMMAND_TEMPLATE.format(
            command_fifo=self._COMMAND_FIFO,
            response_fifo=self._RESPONSE_FIFO,
            ready_flag=self._READY_FLAG,
            worker_path=self._WORKER_PATH,
            worker_b64=worker_b64,
            pip_install_packages=pip_install_packages,
        )

        # Interception server state (shared across rollouts)
        self._interception_server: Any = None
        self._server_lock = asyncio.Lock()
        self._server_runner: Any = None
        self._server_site: Any = None

        # Tunnel pool for exposing interception server to sandboxes
        self._tunnel_pool: TunnelPool | None = (
            TunnelPool(port=interception_port) if interception_host is None else None
        )

        # Active rollout tracking for sub-LLM request routing
        self.active_rollouts: dict[str, dict[str, Any]] = {}

        # Logprobs support detection (None = not tested yet)
        self._sub_llm_supports_logprobs: bool | None = None

        super().__init__(
            sandbox_name="rlm-env",
            start_command=start_command,
            max_turns=max_iterations,
            rubric=rubric,
            **kwargs,
        )

        # Remove bash tool from parent - we use our own REPL tool
        if hasattr(self, "tool_map") and "bash" in self.tool_map:
            self.remove_tool(self.bash)

        # Add the Python REPL tool (sandbox_id and state are injected via update_tool_args)
        self.add_tool(self.call_python_repl, args_to_skip=["sandbox_id", "state"])

    # =========================================================================
    # Sub-Agent Tool Infrastructure
    # =========================================================================

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

        lines = ["\n## Sub-Agent Tools\n"]
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

    def _generate_metadata_documentation(self, metadata: dict[str, Any]) -> str:
        """Generate a concise summary of input data metadata for the system prompt."""
        if not metadata:
            return "No input data metadata available."

        lines = ["The environment contains the following input data in `extra_data`:"]
        for key, value in metadata.items():
            # Format key for better readability
            display_key = key.replace("_", " ").title()
            lines.append(f"- **{display_key}**: `{value}`")
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

    async def _test_logprobs_support(self, client: Any, model: str) -> bool:
        """Test if model supports logprobs with a minimal API call."""
        try:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                logprobs=True,
            )
            return True
        except Exception as e:
            error_text = str(e).lower()
            if "403" in error_text and "logprob" in error_text:
                return False
            raise  # Re-raise other errors

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

    async def _call_sub_llm_api(
        self, client: Any, model: str, messages: list[dict], tools: list | None = None
    ) -> Any | None:
        """Make a single sub-LLM API call with timeout. Returns None on timeout."""
        try:
            return await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    logprobs=self._sub_llm_supports_logprobs or None,
                ),
                timeout=self.sub_llm_api_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Sub-LLM API call timed out after {self.sub_llm_api_timeout}s"
            )
            return None

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
        self, client: Any, model: str, messages: list[dict]
    ) -> SubLLMResult:
        """Run a sub-LLM call, with optional tool-calling loop."""
        # Fast path: no tools configured - single LLM call
        if not self.sub_tools:
            response = await self._call_sub_llm_api(client, model, messages)
            if response is None:
                return self._make_timeout_result([], 0, 0, 0, 0)

            prompt_tokens, completion_tokens = self._extract_tokens(response)
            return SubLLMResult(
                final_content=response.choices[0].message.content or "",
                turns=[
                    SubLLMTurn(
                        prompt_messages=[dict(m) for m in messages],
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
            prompt_snapshot = [dict(m) for m in current_messages]

            response = await self._call_sub_llm_api(
                client, model, current_messages, tools
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

            current_messages.append(assistant_message.model_dump())

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                tool_result = await self._call_sub_tool(
                    tool_name, tool_args, tool_call.id
                )
                current_messages.append(tool_result)

        # Max turns reached - add prompt for final answer and make call without tools
        num_turns += 1
        current_messages.append(
            {
                "role": "user",
                "content": "You've reached the maximum number of tool calls. "
                "Based on the information gathered, provide your final answer inside \\boxed{}.",
            }
        )

        prompt_snapshot = [dict(m) for m in current_messages]
        response = await self._call_sub_llm_api(client, model, current_messages)
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

    # =========================================================================
    # Interception Server (for sub-LLM calls from sandbox code)
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

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.interception_port)
            await site.start()

            self._interception_server = app
            self._server_runner = runner
            self._server_site = site

            logger.debug(
                f"Started RLM interception server on port {self.interception_port}"
            )

    async def _handle_sub_llm_request(self, request: Any) -> Any:
        """Handle sub-LLM requests from sandbox code."""
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

        messages = request_body.get("messages", [])
        batch_id = request_body.get("_batch_id", "")
        request_id = request_body.get("_request_id", "")

        # Prepend system message with \boxed{} instruction
        messages_with_system = [
            {"role": "system", "content": _SUB_LLM_SYSTEM_PROMPT},
            *messages,
        ]

        try:
            # Run sub-LLM call (handles both with-tools and no-tools cases)
            result = await self._run_sub_llm(client, sub_model, messages_with_system)
            final_content = result["final_content"]
            prompt_tokens = result["total_prompt_tokens"]
            completion_tokens = result["total_completion_tokens"]
            tool_call_count = result["tool_call_count"]
            num_turns = result["num_turns"]
            max_turns_reached = result["max_turns_reached"]
            turns = result["turns"]

            # Extract boxed answer for response to sandbox
            boxed_content = extract_boxed_answer(final_content)

            # Build TrajectorySteps if enabled
            if self.include_sub_llm_in_trajectory:
                parent_turn = context.get("current_turn", 0)
                timestamp = time.time()

                total_sub_turns = len(turns)
                for sub_turn_index, turn in enumerate(turns):
                    # Parse tokens from response
                    tokens = await parse_response_tokens(
                        turn["response"], "chat", self.max_seq_len
                    )
                    # Parse completion messages
                    completion_messages = await parse_response_messages(
                        turn["response"], "chat"
                    )
                    # Check if response was truncated
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
                        extras={
                            "is_sub_llm_call": True,
                            "parent_turn": parent_turn,
                            "batch_id": batch_id,
                            "request_id": request_id,
                            "sub_turn_index": sub_turn_index,
                            "total_sub_turns": total_sub_turns,
                            "timestamp": timestamp,
                            "tool_call_count": turn["tool_call_count"],
                        },
                    )
                    context.setdefault("sub_llm_trajectory_steps", []).append(
                        trajectory_step
                    )

            # Build response dict for sandbox
            response_dict = {
                "choices": [{"message": {"content": boxed_content}}],
                "_rlm_metadata": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tool_call_count": tool_call_count,
                    "num_turns": num_turns,
                    "max_turns_reached": max_turns_reached,
                },
            }

            return web.json_response(response_dict)
        except Exception as e:
            logger.error(f"Sub-LLM call failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @vf.teardown
    async def teardown_tunnels(self):
        """Stop all cloudflared tunnel processes."""
        if self._tunnel_pool:
            self._tunnel_pool.teardown()

    # =========================================================================
    # State Management
    # =========================================================================

    async def _execute_command_with_retry(
        self, sandbox_id: str, command: str, timeout: int | None = None
    ):
        """Execute command with retry logic for transient sandbox errors."""
        effective_timeout = timeout or self.timeout_per_command_seconds
        start = perf_counter()
        logger.debug(f"Executing command in sandbox {sandbox_id}: {command[:100]}...")
        try:
            result = await self.with_retry(self.sandbox_client.execute_command)(
                sandbox_id, command, timeout=effective_timeout
            )
        except CommandTimeoutError as e:
            logger.debug(
                f"Command timed out after {effective_timeout}s in sandbox {sandbox_id}"
            )
            raise vf.SandboxError() from e
        except Exception as e:
            raise vf.SandboxError() from e
        elapsed = perf_counter() - start
        logger.debug(f"Command completed in {elapsed:.1f}s")
        return result

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: Messages,
        state: State,
        **kwargs,
    ) -> dict[str, Any]:
        """Inject sandbox_id and state into call_python_repl tool args."""
        if tool_name == "call_python_repl":
            updated_args = dict(tool_args)
            updated_args["sandbox_id"] = state["sandbox_id"]
            updated_args["state"] = state
            return updated_args
        else:
            return super().update_tool_args(
                tool_name, tool_args, messages, state, **kwargs
            )

    async def _setup_interception_and_register(
        self, state: State, rollout_id: str
    ) -> State:
        """Start interception server, configure tunnel, and register rollout."""
        await self._ensure_interception_server()

        if self._tunnel_pool:
            tunnel_url = await self._tunnel_pool.get_tunnel_url(
                len(self.active_rollouts)
            )
            interception_url = f"{tunnel_url}/rollout/{rollout_id}/v1/chat/completions"
        else:
            tunnel_url = None
            interception_url = f"http://{self.interception_host}:{self.interception_port}/rollout/{rollout_id}/v1/chat/completions"

        state["interception_url"] = interception_url
        state["tunnel_url"] = tunnel_url

        self.active_rollouts[rollout_id] = {
            "client": state.get("client"),
            "model": state.get("model"),
            "sub_model": self.sub_model or state.get("model"),
        }
        return state

    async def _start_worker(self, state: State) -> None:
        """Start the Python worker process in the sandbox and wait for ready signal."""
        sandbox_id = state["sandbox_id"]
        interception_url = state["interception_url"]

        sub_llm_timeout = max(10, int(self.code_execution_timeout * 0.9))
        # Calculate max wait iterations for worker script (written AFTER pip install completes)
        # This can take 30-60+ seconds for heavy packages like numpy, sympy, scipy
        script_wait_iterations = max(1, int(self.max_startup_wait_seconds / 0.1))
        start_worker_cmd = f"""
export RLM_INTERCEPTION_URL="{interception_url}"
export RLM_SUB_MODEL="{self.sub_model or state.get("model", "")}"
export RLM_MAX_SUB_LLM_PARALLELISM="{self.max_sub_llm_parallelism}"
export RLM_SUB_LLM_TIMEOUT="{sub_llm_timeout}"
export RLM_SANDBOX_TIMEOUT="{self.code_execution_timeout}"

# Wait for worker script to exist (written AFTER pip install completes in start_command)
# This can take 30-60+ seconds for heavy packages like numpy, sympy, scipy
sync 2>/dev/null || true
for i in $(seq 1 {script_wait_iterations}); do
    if [ -f "{self._WORKER_PATH}" ]; then
        break
    fi
    sleep 0.1
done

if [ ! -f "{self._WORKER_PATH}" ]; then
    echo "Worker script not found - pip install may have failed or timed out" >&2
    exit 1
fi

# Small delay to ensure filesystem is fully synced before reading script
sleep 0.2

# Start the worker
nohup python -u {self._WORKER_PATH} >> /tmp/rlm_worker.log 2>&1 &
"""
        # Timeout needs to account for pip install wait + worker startup
        start_worker_timeout = self.max_startup_wait_seconds + 30
        await self._execute_command_with_retry(
            sandbox_id, start_worker_cmd, timeout=start_worker_timeout
        )
        await self._wait_for_worker_ready(sandbox_id)

    async def _prepare_sandbox_and_start_worker(
        self, state: State, context_dict: dict[str, Any]
    ) -> None:
        """Write files to sandbox and start the worker process."""
        sandbox_id = state["sandbox_id"]
        try:
            await self.sandbox_client.wait_for_creation(sandbox_id)
        except Exception as e:
            raise SandboxNotReadyError(e)
        await self._write_json_to_sandbox(
            sandbox_id, context_dict, self._CONTEXT_FILE, "rlm_context.json"
        )
        await self._write_json_to_sandbox(
            sandbox_id,
            {"ready": False, "content": ""},
            self._ANSWER_FILE,
            "rlm_answer.json",
        )
        await self._start_worker(state)

    async def _recreate_sandbox(self, state: State) -> State:
        """Delete the current sandbox and create a fresh one."""
        old_sandbox_id = state.get("sandbox_id")
        if old_sandbox_id:
            # Remove from active sandboxes and delete
            self.active_sandboxes.discard(old_sandbox_id)
            try:
                await self.sandbox_client.delete(old_sandbox_id)
            except Exception as e:
                logger.warning(f"Failed to delete broken sandbox {old_sandbox_id}: {e}")

        # Wait a second to make absolutely sure that the sandbox is deleted
        await asyncio.sleep(1)

        # Create new sandbox via parent's parent (SandboxEnv.setup_state)
        # We need to call the grandparent to avoid re-running RLM setup
        try:
            sandbox = await self.with_retry(self.sandbox_client.create)(
                self.sandbox_request
            )
        except Exception as e:
            raise SandboxCreationError(e)
        self.active_sandboxes.add(sandbox.id)
        logger.debug(f"Created replacement sandbox {sandbox.id}")
        state["sandbox_id"] = sandbox.id
        return state

    async def setup_state(self, state: State, **kwargs) -> State:
        """Setup sandbox with context and worker, plus interception for sub-LLM calls."""
        # 1. Create sandbox via parent
        state = await super().setup_state(state, **kwargs)
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            raise vf.SandboxError() from Exception("Sandbox ID not set")

        rollout_id = f"rlm_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        # 2. Setup interception, tunnels, and register rollout
        state = await self._setup_interception_and_register(state, rollout_id)

        # 3. Test logprobs support on first rollout
        if self._sub_llm_supports_logprobs is None:
            client = state.get("client")
            sub_model = self.sub_model or state.get("model")
            if client and sub_model:
                self._sub_llm_supports_logprobs = await self._test_logprobs_support(
                    client, sub_model
                )

        # 5. Build context
        info = state.get("info", {})
        context_data = info.get(self.context_key, None)
        metadata: dict[str, str | int] = {"type": str(type(context_data))}
        if context_data is None:
            metadata["size"] = 0
        elif hasattr(context_data, "__len__"):
            metadata["size"] = len(context_data)
        else:
            metadata["size"] = "unknown"
        context_dict = {"input_data": context_data, "input_data_metadata": metadata}
        state["rlm_context"] = context_dict

        # 6. Prepare sandbox and start worker (with retry using fresh sandbox)
        max_sandbox_retries = 5

        for attempt in range(max_sandbox_retries):
            try:
                await self._prepare_sandbox_and_start_worker(state, context_dict)
                break  # Success
            except vf.SandboxError as e:
                if (
                    "worker failed to start" in str(e.__cause__)
                    and attempt < max_sandbox_retries - 1
                ):
                    logger.warning(
                        f"Worker startup failed (attempt {attempt + 1}/{max_sandbox_retries}), "
                        f"recreating sandbox..."
                    )
                    state = await self._recreate_sandbox(state)
                else:
                    raise

        state["rlm_worker_ready"] = True

        # Initialize context warning flag (feature enabled if max_seq_len is set)
        state["context_warning_sent"] = False

        # Initialize FIFO sequence counter for detecting stale responses
        state["_exec_seq"] = 0

        return state

    async def _write_json_to_sandbox(
        self, sandbox_id: str, data: dict, file_path: str, filename: str
    ) -> None:
        """Write JSON data to sandbox file using direct file upload."""
        data_bytes = json.dumps(data).encode("utf-8")
        await self.with_retry(self.sandbox_client.upload_bytes)(
            sandbox_id, file_path=file_path, file_bytes=data_bytes, filename=filename
        )

    async def _wait_for_worker_ready(self, sandbox_id: str) -> None:
        """Wait for worker to signal ready."""
        wait_script = _make_ready_wait_script(
            self._READY_FLAG, self.max_startup_wait_seconds
        )
        # Use max_startup_wait_seconds as timeout (+ buffer for script overhead)
        timeout = self.max_startup_wait_seconds + 10
        result = await self._execute_command_with_retry(
            sandbox_id, wait_script, timeout=timeout
        )
        if "failed to start" in result.stdout or "failed to start" in (
            result.stderr or ""
        ):
            # Debug: get more info about why it failed
            debug_result = await self._execute_command_with_retry(
                sandbox_id,
                "ls -la /tmp/rlm* 2>&1; echo '---LOG---'; cat /tmp/rlm_worker.log 2>&1 || echo 'no log'; echo '---PS---'; ps aux 2>&1",
            )
            logger.error(
                f"RLM worker failed to start. Debug info:\n{debug_result.stdout}"
            )
            raise vf.SandboxError() from Exception(
                f"RLM worker failed to start: {debug_result.stdout[:500]}"
            )

    # =========================================================================
    # Code Execution
    # =========================================================================

    async def _execute_code(
        self, sandbox_id: str, code: str, state: State
    ) -> dict[str, Any]:
        """Execute code in sandbox worker and return result."""
        # Increment and track sequence number for this execution
        seq = state.get("_exec_seq", 0) + 1
        state["_exec_seq"] = seq

        payload = {"code": code, "seq": seq}
        payload_json = json.dumps(payload)
        payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("utf-8")

        command = textwrap.dedent(
            f"""
            python3 - <<'PY'
import base64
import json
import sys

data = base64.b64decode('{payload_b64}').decode('utf-8')
with open('{self._COMMAND_FIFO}', 'w', encoding='utf-8') as command_file:
    command_file.write(data)
with open('{self._RESPONSE_FIFO}', 'r', encoding='utf-8') as response_file:
    sys.stdout.write(response_file.read())
PY
            """
        )

        try:
            # No retry for code execution - if code times out or fails, return error to model
            result = await self.sandbox_client.execute_command(
                sandbox_id, command, timeout=self.code_execution_timeout
            )
        except CommandTimeoutError as e:
            logger.warning(
                f"Code execution timed out after {self.code_execution_timeout}s"
            )
            if self.abort_on_code_timeout:
                # Abort rollout immediately on timeout
                raise vf.SandboxError() from e
            # Return error to model so it can try more efficient code
            return {
                "status": "error",
                "stdout": "",
                "stderr": "",
                "result": f"Code execution timed out after {self.code_execution_timeout} seconds. "
                "Your code may be too slow - consider a more efficient algorithm or "
                "breaking the computation into smaller steps.",
                "answer": {"ready": False, "content": ""},
            }
        except Exception as e:
            # Other errors - abort the rollout
            logger.error(f"Sandbox error during code execution: {e}")
            raise vf.SandboxError() from e

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

    async def call_python_repl(self, code: str, sandbox_id: str, state: Any) -> str:
        """
        Execute Python code in a persistent REPL environment.

        The REPL maintains state across calls and provides access to:

        - `extra_data`: The actual input data you need to process.

        - `answer`: A dictionary for your final answer:
          - `answer["content"]`: Your answer (string) - update this as you work
          - `answer["ready"]`: Set to `True` to finish (terminates execution immediately)

        - `llm_batch(prompts, **kwargs)`: Make sub-LLM calls for help with subtasks
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
        result = await self._execute_code(sandbox_id, code, state)
        execution_time = perf_counter() - execution_start
        output = self._format_execution_output(result)

        # Track timing in state for metrics
        state.setdefault("tool_call_timings", []).append(
            {
                "turn": state.get("turn", 0),
                "execution_seconds": execution_time,
            }
        )

        # Append execution time to output
        output += f"\n[Execution time: {execution_time:.2f}s]"

        # Check if answer is ready
        answer = result.get("answer", {})
        if answer.get("ready", False):
            state["final_answer"] = answer.get("content", "")
            logger.debug(f"Answer ready: {state['final_answer'][:100]}...")

        # Inject context limit warning if approaching limit
        if self.max_seq_len and not state.get("context_warning_sent"):
            # Get prompt token count from latest trajectory response
            trajectory = state.get("trajectory", [])
            response = trajectory[-1].get("response") if trajectory else None
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

            # Build system prompt with metadata, packages and sub-tool documentation
            metadata = state.get("rlm_context", {}).get("input_data_metadata", {})
            metadata_summary = self._generate_metadata_documentation(metadata)

            base_system_prompt = self.custom_system_prompt or _RLM_SYSTEM_PROMPT
            if "{metadata_summary}" in base_system_prompt:
                # Use replace instead of format to avoid conflict with curly braces from Python code
                base_system_prompt = base_system_prompt.replace(
                    "{metadata_summary}", metadata_summary
                )
            else:
                # If custom prompt doesn't have placeholder, prepend it
                base_system_prompt = f"{metadata_summary}\n\n{base_system_prompt}"

            packages_docs = self._generate_packages_documentation()
            sub_tools_docs = self._generate_sub_tools_documentation()
            system_prompt = base_system_prompt + packages_docs + sub_tools_docs

            messages = list(prompt)
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                # Append packages and tool docs to existing system prompt
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + packages_docs + sub_tools_docs,
                }
            return cast(Messages, messages)
        else:
            # Subsequent turns: use parent implementation
            return await super().get_prompt_messages(state)

    # =========================================================================
    # Stop Conditions
    # =========================================================================

    async def _ensure_final_answer(self, state: State) -> None:
        """Read final answer from sandbox if not already set."""
        if "final_answer" in state:
            return
        state["final_answer"] = ""
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            try:
                result = await self._execute_command_with_retry(
                    sandbox_id,
                    f'cat {self._ANSWER_FILE} 2>/dev/null || echo \'{{"content": ""}}\'',
                )
                state["final_answer"] = json.loads(result.stdout.strip()).get(
                    "content", ""
                )
            except Exception:
                pass

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
        from collections import Counter

        main_trajectory_len = len(state.get("trajectory", []))
        rollout_id = state.get("rollout_id")

        if rollout_id and rollout_id in self.active_rollouts:
            context = self.active_rollouts[rollout_id]
            sub_steps = context.get("sub_llm_trajectory_steps", [])

            if sub_steps:
                # Extract unique (batch_id, request_id) pairs and count per batch
                pairs = {
                    (e.get("batch_id"), e.get("request_id"))
                    for s in sub_steps
                    if (e := s.get("extras", {})).get("batch_id")
                }
                batch_sizes = list(Counter(b for b, _ in pairs).values())

                state["sub_llm_call_count"] = len(pairs)
                state["sub_llm_prompt_tokens"] = sum(
                    self._extract_tokens(s.get("response"))[0] for s in sub_steps
                )
                state["sub_llm_completion_tokens"] = sum(
                    self._extract_tokens(s.get("response"))[1] for s in sub_steps
                )
                state["sub_llm_total_tool_calls"] = sum(
                    s.get("extras", {}).get("tool_call_count", 0) or 0
                    for s in sub_steps
                )
                state["sub_llm_total_turns"] = len(sub_steps)
                state["sub_llm_batch_count"] = len(batch_sizes)
                state["sub_llm_max_batch_size"] = max(batch_sizes, default=0)
                state["sub_llm_mean_batch_size"] = (
                    sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0
                )

            if self.include_sub_llm_in_trajectory and sub_steps:
                sub_steps_sorted = sorted(
                    sub_steps, key=lambda s: s["extras"].get("timestamp", 0)
                )
                state["trajectory"] = sub_steps_sorted + state["trajectory"]

            del self.active_rollouts[rollout_id]

        # Compute main RLM metrics (excluding sub-LLM steps)
        state["main_rlm_turns"] = main_trajectory_len
        main_steps = [
            s
            for s in state.get("trajectory", [])
            if not s.get("extras", {}).get("is_sub_llm_call")
        ]
        state["main_rlm_prompt_tokens"] = sum(
            self._extract_tokens(s.get("response"))[0] for s in main_steps
        )
        state["main_rlm_completion_tokens"] = sum(
            self._extract_tokens(s.get("response"))[1] for s in main_steps
        )

        # Release tunnel
        if (tunnel_url := state.get("tunnel_url")) and self._tunnel_pool:
            await self._tunnel_pool.release_tunnel(tunnel_url)

    async def post_rollout(self, state: State):
        """Read final answer from sandbox if not already set."""
        await self._ensure_final_answer(state)
