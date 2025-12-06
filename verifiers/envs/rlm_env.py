"""
Recursive Language Model (RLM) Environment.

Implements the RLM inference strategy where language models can decompose and
recursively interact with input context of unbounded length through REPL environments.

Based on: https://www.alexzhang.dev/blog/recursive-language-models

Architecture:
- REPL loop runs in the framework (MultiTurnEnv pattern)
- Sandbox is used only for code execution (persistent Python worker)
- Sub-LLM calls from sandbox code are intercepted via HTTP proxy

Key features:
- Works with any dataset that has a normal prompt
- Optional large context can be provided in info["context"]
- Root model only sees query, not full context (unless it peeks via code)
- Model can make recursive sub-LLM calls via llm_batch() function
- Final answer returned via answer variable
"""

import asyncio
import base64
import json
import logging
import re
import subprocess
import textwrap
import time
import uuid
from typing import Any, Callable

from aiohttp import web

import verifiers as vf
from verifiers.envs.sandbox_env import SandboxEnv
from verifiers.types import Messages, State
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.tool_utils import convert_func_to_oai_tool

logger = logging.getLogger(__name__)


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

    def ensure_fifo(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        os.mkfifo(path)

    for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):
        ensure_fifo(fifo_path)

    # Load context from file (written by setup_state)
    context = {{"input_data": None, "input_data_metadata": {{"type": "none", "size": 0}}}}
    if Path(CONTEXT_FILE).exists():
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            context = json.load(f)

    # Initialize answer structure
    answer = {{"ready": False, "content": ""}}
    if Path(ANSWER_FILE).exists():
        with open(ANSWER_FILE, "r", encoding="utf-8") as f:
            answer = json.load(f)

    def _single_llm_call(prompt: str, **kwargs) -> str:
        """Make a single sub-LLM call via interception server."""
        if not INTERCEPTION_URL:
            return "Error: Sub-LLM interception URL not configured"
        
        try:
            payload = {{
                "model": SUB_MODEL or "default",
                "messages": [{{"role": "user", "content": prompt}}],
            }}
            # Add any extra kwargs
            for k, v in kwargs.items():
                if k not in ("model", "messages"):
                    payload[k] = v
            
            resp = requests.post(
                INTERCEPTION_URL,
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{{}}])[0].get("message", {{}}).get("content", "")
        except Exception as e:
            return f"Error in sub-LLM call: {{e}}"

    def llm_batch(prompts: list, **kwargs) -> list:
        """
        Make multiple sub-LLM calls in parallel.
        
        Parallelism is controlled by RLM_MAX_SUB_LLM_PARALLELISM.
        
        Args:
            prompts: List of prompts for the sub-LLMs
            **kwargs: Additional arguments applied to all calls
        
        Returns:
            List of responses in the same order as the input prompts
        """
        with ThreadPoolExecutor(max_workers=MAX_SUB_LLM_PARALLELISM) as executor:
            futures = [executor.submit(_single_llm_call, p, **kwargs) for p in prompts]
            return [f.result() for f in futures]

    # Persistent execution namespace
    namespace: dict[str, object] = {{
        "__name__": "__main__",
        "context": context,
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
        execution_count += 1
        
        result = {{
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": None,
            "execution_count": execution_count,
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

    pip install -q requests

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


_RLM_READY_WAIT_SCRIPT = textwrap.dedent(
    """
    bash -lc '
    for i in $(seq 1 200); do
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


# System prompt for RLM
_RLM_SYSTEM_PROMPT = """You are operating in a Recursive Language Model (RLM) environment - an iterative Python REPL where you explore data step by step.

## Critical: This is an ITERATIVE environment

You will write code, see its output, then write more code based on what you learned. **Do NOT try to solve everything in one code block.** Each code block executes and returns output before you continue.

**You MUST wrap all code in markdown code blocks** using triple backticks with `python`:
```python
# your code here
```
Only code inside these blocks will be executed. Text outside code blocks is for your reasoning.

## Available Variables and Functions

- `context`: A dictionary containing:
  - `context["input_data_metadata"]`: Metadata about the input (type, size, structure, etc.)
  - `context["input_data"]`: The actual input data

- `answer`: A dictionary for your final answer:
  - `answer["content"]`: Your answer (string) - write and update this throughout execution
  - `answer["ready"]`: Set to `True` to finish - **this immediately terminates execution**

- `llm_batch(prompts, **kwargs)`: Make sub-LLM calls for help with subtasks
  - Takes a list of prompts, returns a list of responses (same order)
  - Useful for semantic understanding, summarization, complex reasoning
  - Include any context you need directly in the prompt strings
  - Parallelism is automatically rate-limited
  - Examples:
    ```python
    # Single sub-LLM call
    result = llm_batch(["What is the main theme of this text?"])[0]
    print(result)
    
    # Multiple parallel sub-LLM calls
    results = llm_batch([
        "Summarize section 1: " + section1,
        "Summarize section 2: " + section2,
        "Summarize section 3: " + section3,
    ])
    # results[0] = response for section 1, results[1] = response for section 2, etc.
    ```

## Workflow

**Step 1: Inspect metadata first**
```python
print(context["input_data_metadata"])
```
Wait for output. This tells you the data type, size, and structure before you look at the actual data.

**Step 2: Explore the data based on what you learned**
```python
# Look at a sample of the actual data
data = context["input_data"]
print(type(data))
print(data[:500] if isinstance(data, str) else data[:3])  # First part only
```
Wait for output. Now you know the actual format.

**Step 3: Process and build your answer**
```python
# Based on what you've seen, write code to solve the task
# ...
answer["content"] = "your current best answer"
```
You can update `answer["content"]` multiple times as you refine your solution.

**Step 4: Verify and finalize (only after reviewing output)**
```python
print(f"My answer: {answer['content']}")
# Only after confirming this looks correct:
answer["ready"] = True
```

## Important Rules

1. **Always use markdown code blocks** - wrap code in triple backticks with `python`, or it won't execute
2. **NEVER set `answer["ready"] = True` until you have seen execution output** - you need feedback first
3. **Start with metadata** - always run `print(context["input_data_metadata"])` before accessing `input_data`
4. **One step at a time** - write small code blocks, see output, then continue
5. **Use `llm_batch()` for semantic tasks** - summarization, understanding text, classification, etc.
6. You can think in natural language between code blocks - reasoning and planning are encouraged

The environment executes your code and shows you the output. Use that feedback to iterate toward the correct answer.
"""


class RLMEnv(SandboxEnv):
    """
    Recursive Language Model Environment.

    Extends SandboxEnv to provide a Python REPL environment where the model can:
    - Interact with large context stored as a variable
    - Make recursive sub-LLM calls via llm_batch()
    - Return final answers via an answer variable

    Architecture:
    - REPL loop runs in the framework (standard MultiTurnEnv pattern)
    - Sandbox is used only for code execution (persistent Python worker)
    - Sub-LLM calls from sandbox code are intercepted via HTTP proxy

    Works with any dataset that has a normal prompt. Context can optionally
    be provided in info[context_key] for large data that shouldn't be in the prompt.

    Args:
        sub_model: Model to use for sub-LLM calls (defaults to same as root model)
        sub_tools: List of Python functions that sub-LLMs can use as tools.
                   These tools are NOT available to the root model.
        sub_tool_max_turns: Maximum tool-calling turns for sub-LLM calls (default: 5)
        max_iterations: Maximum REPL iterations before stopping (maps to max_turns)
        max_output_length: Maximum length of code execution output
        max_sub_llm_parallelism: Maximum number of concurrent sub-LLM calls
        context_key: Key in info containing optional context data (default: "context")
        system_prompt: Custom system prompt (default: RLM standard prompt)
        interception_host: Optional hostname/IP for interception server (auto-tunneled if not set)
        interception_port: Port for interception server (default: 8766)
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
        )

        # Interception server state (shared across rollouts)
        self._interception_server: Any = None
        self._server_lock = asyncio.Lock()
        self._server_runner: Any = None
        self._server_site: Any = None
        self._tunnels: list[dict[str, Any]] = []
        self._tunnel_lock = asyncio.Lock()
        self._tunnel_round_robin_index = 0

        # Active rollout tracking for sub-LLM request routing
        self.active_rollouts: dict[str, dict[str, Any]] = {}

        super().__init__(
            sandbox_name="rlm-env",
            start_command=start_command,
            max_turns=max_iterations,
            **kwargs,
        )

        # Remove bash tool from parent - we use our own code execution
        if hasattr(self, "tool_map") and "bash" in self.tool_map:
            self.remove_tool(self.bash)

    # =========================================================================
    # Sub-Agent Tool Infrastructure
    # =========================================================================

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
            params = func_def.get("parameters", {}).get("properties", {})

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

    async def _call_sub_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str
    ) -> dict:
        """Execute a sub-agent tool call. Returns tool message dict."""
        try:
            tool_func = self.sub_tool_map[tool_name]
            result = await maybe_await(tool_func, **tool_args)
            return {"role": "tool", "content": str(result), "tool_call_id": tool_call_id}
        except Exception as e:
            return {"role": "tool", "content": f"Error: {e}", "tool_call_id": tool_call_id}

    async def _run_sub_llm_with_tools(
        self, client: Any, model: str, messages: list[dict]
    ) -> dict:
        """
        Run a sub-LLM call with tool-calling loop.

        Returns the final response dict in OAI format.
        """
        current_messages = list(messages)

        for _ in range(self.sub_tool_max_turns):
            # Make LLM call with tools
            response = await client.chat.completions.create(
                model=model,
                messages=current_messages,
                tools=self.sub_oai_tools if self.sub_oai_tools else None,
            )

            assistant_message = response.choices[0].message
            tool_calls = getattr(assistant_message, "tool_calls", None)

            # If no tool calls, we're done
            if not tool_calls:
                return (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else dict(response)
                )

            # Add assistant message with tool calls to conversation
            current_messages.append(assistant_message.model_dump())

            # Execute all tool calls
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

        # Max turns reached - make final call without tools
        response = await client.chat.completions.create(
            model=model,
            messages=current_messages,
        )
        return (
            response.model_dump() if hasattr(response, "model_dump") else dict(response)
        )

    # =========================================================================
    # Interception Server (for sub-LLM calls from sandbox code)
    # =========================================================================

    def _ensure_cloudflared_installed(self) -> str:
        """Install cloudflared if not already installed. Returns path to cloudflared binary."""
        import platform
        import shutil

        cloudflared_path = shutil.which("cloudflared")
        if cloudflared_path:
            return cloudflared_path

        logger.info("Installing cloudflared...")
        system = platform.system()

        if system == "Darwin":  # macOS
            result = subprocess.run(
                ["brew", "install", "cloudflare/cloudflare/cloudflared"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to install cloudflared via Homebrew: {result.stderr}"
                )
            cloudflared_path = shutil.which("cloudflared")
            if not cloudflared_path:
                raise RuntimeError("cloudflared installed but not found in PATH")
            return cloudflared_path
        elif system == "Linux":
            install_script = "curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared.deb && rm cloudflared.deb"
            result = subprocess.run(
                ["bash", "-c", install_script],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install cloudflared: {result.stderr}")
            cloudflared_path = shutil.which("cloudflared")
            if not cloudflared_path:
                raise RuntimeError("cloudflared installed but not found in PATH")
            return cloudflared_path
        else:
            raise RuntimeError(
                f"Unsupported platform: {system}. "
                "Please install cloudflared manually."
            )

    def _extract_tunnel_url_from_line(self, line: str) -> str | None:
        """Extract tunnel URL from a line of cloudflared output."""
        if ".trycloudflare.com" not in line:
            return None
        start_idx = line.find("https://")
        if start_idx == -1:
            return None
        url_end = start_idx + 8
        while url_end < len(line) and not line[url_end].isspace():
            url_end += 1
        url = line[start_idx:url_end].rstrip("/")
        if ".trycloudflare.com" in url:
            return url
        return None

    def _start_cloudflared_tunnel(self) -> tuple[str, subprocess.Popen]:
        """Start cloudflared tunnel and return (URL, process)."""
        cloudflared_path = self._ensure_cloudflared_installed()
        tunnel_process = subprocess.Popen(
            [
                cloudflared_path,
                "tunnel",
                "--url",
                f"http://localhost:{self.interception_port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stderr_lines = []
        max_wait_seconds = 30
        check_interval = 0.5
        max_iterations = int(max_wait_seconds / check_interval)

        for _ in range(max_iterations):
            if tunnel_process.poll() is not None:
                if tunnel_process.stderr:
                    remaining = tunnel_process.stderr.read()
                    stderr_lines.append(remaining)
                error_output = "".join(stderr_lines)
                raise RuntimeError(
                    f"cloudflared tunnel failed to start: {error_output}"
                )
            if tunnel_process.stderr:
                line = tunnel_process.stderr.readline()
                if line:
                    stderr_lines.append(line)
                    url = self._extract_tunnel_url_from_line(line)
                    if url:
                        logger.info(f"Cloudflare tunnel started: {url}")
                        return url, tunnel_process
            time.sleep(check_interval)

        raise RuntimeError(
            f"Failed to get tunnel URL from cloudflared after {max_wait_seconds} seconds."
        )

    async def _get_tunnel_url(self) -> str:
        """Get tunnel URL, creating new tunnel if needed."""
        async with self._tunnel_lock:
            total_active = len(self.active_rollouts)
            required_tunnels = max(1, (total_active + 49) // 50)

            while len(self._tunnels) < required_tunnels:
                try:
                    url, process = self._start_cloudflared_tunnel()
                    self._tunnels.append({
                        "url": url,
                        "process": process,
                        "active_rollouts": 0,
                    })
                except Exception as e:
                    logger.error(f"Failed to create tunnel: {e}")
                    raise

            tunnel = self._tunnels[self._tunnel_round_robin_index % len(self._tunnels)]
            self._tunnel_round_robin_index += 1
            tunnel["active_rollouts"] += 1
            return tunnel["url"]

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

            logger.debug(f"Started RLM interception server on port {self.interception_port}")

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

        messages = request_body.get("messages", [])

        try:
            # Use tool-calling loop if sub_tools are configured
            if self.sub_tools:
                response_dict = await self._run_sub_llm_with_tools(
                    client, sub_model, messages
                )
            else:
                # Original simple path
                response = await client.chat.completions.create(
                    model=sub_model,
                    messages=messages,
                )
                response_dict = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else dict(response)
                )

            return web.json_response(response_dict)
        except Exception as e:
            logger.error(f"Sub-LLM call failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    @vf.teardown
    async def teardown_tunnels(self):
        """Stop all cloudflared tunnel processes."""
        async with self._tunnel_lock:
            for tunnel in self._tunnels:
                process = tunnel.get("process")
                if process:
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except Exception as e:
                        logger.warning(f"Error stopping tunnel: {e}")
                        try:
                            process.kill()
                        except Exception:
                            pass
            self._tunnels.clear()

    # =========================================================================
    # State Management
    # =========================================================================

    async def setup_state(self, state: State) -> State:
        """Setup sandbox with context and worker, plus interception for sub-LLM calls."""
        # Create sandbox via parent
        state = await super().setup_state(state)
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            raise RuntimeError("Sandbox ID not set")

        rollout_id = f"rlm_{uuid.uuid4().hex[:8]}"
        state["rollout_id"] = rollout_id

        # Start interception server for sub-LLM calls
        await self._ensure_interception_server()

        # Get tunnel URL for sandbox to call back
        if self.interception_host is None:
            tunnel_url = await self._get_tunnel_url()
            interception_url = f"{tunnel_url}/rollout/{rollout_id}/v1/chat/completions"
        else:
            interception_url = f"http://{self.interception_host}:{self.interception_port}/rollout/{rollout_id}/v1/chat/completions"

        state["interception_url"] = interception_url
        state["tunnel_url"] = tunnel_url if self.interception_host is None else None

        # Register rollout for sub-LLM routing
        self.active_rollouts[rollout_id] = {
            "client": state.get("client"),
            "model": state.get("model"),
            "sub_model": self.sub_model or state.get("model"),
        }

        # Build context dict
        info = state.get("info", {})
        context_data = info.get(self.context_key, None)
        context_dict = self._build_context_dict(context_data)
        state["rlm_context"] = context_dict

        # Wait for sandbox to be ready
        await self.sandbox_client.wait_for_creation(sandbox_id)

        # Write context file to sandbox BEFORE starting worker
        await self._write_context_to_sandbox(sandbox_id, context_dict)

        # Write initial answer file BEFORE starting worker
        await self._write_answer_to_sandbox(sandbox_id, {"ready": False, "content": ""})

        # Start worker with environment variables set
        # This must happen AFTER writing context/answer files
        start_worker_cmd = f"""
export RLM_INTERCEPTION_URL="{interception_url}"
export RLM_SUB_MODEL="{self.sub_model or state.get('model', '')}"
export RLM_MAX_SUB_LLM_PARALLELISM="{self.max_sub_llm_parallelism}"
nohup python -u {self._WORKER_PATH} > /tmp/rlm_worker.log 2>&1 &
"""
        await self.sandbox_client.execute_command(sandbox_id, start_worker_cmd)

        # Wait for worker to be ready
        await self._wait_for_worker_ready(sandbox_id)

        # Initialize worker state (ready flag indicates worker is running)
        state["rlm_worker_ready"] = True

        return state

    def _build_context_dict(self, context_data: Any) -> dict[str, Any]:
        """Build context dictionary with metadata."""
        if context_data is not None:
            return {
                "input_data": context_data,
                "input_data_metadata": self._build_context_metadata(context_data),
            }
        else:
            return {
                "input_data": None,
                "input_data_metadata": {"type": "none", "size": 0},
            }

    def _build_context_metadata(self, context_data: Any) -> dict[str, Any]:
        """Build minimal metadata dictionary for the context."""
        metadata: dict[str, Any] = {}
        metadata["type"] = str(type(context_data))
        if context_data is None:
            metadata["size"] = 0
        elif hasattr(context_data, "__len__"):
            metadata["size"] = len(context_data)
        else:
            metadata["size"] = "unknown"
        return metadata

    async def _write_context_to_sandbox(self, sandbox_id: str, context_dict: dict) -> None:
        """Write context to sandbox file."""
        context_json = json.dumps(context_dict)
        context_b64 = base64.b64encode(context_json.encode("utf-8")).decode("utf-8")
        cmd = f"""python3 -c "
import base64
from pathlib import Path
Path('{self._CONTEXT_FILE}').write_bytes(base64.b64decode('{context_b64}'))
" """
        await self.sandbox_client.execute_command(sandbox_id, cmd)

    async def _write_answer_to_sandbox(self, sandbox_id: str, answer: dict) -> None:
        """Write answer to sandbox file."""
        answer_json = json.dumps(answer)
        answer_b64 = base64.b64encode(answer_json.encode("utf-8")).decode("utf-8")
        cmd = f"""python3 -c "
import base64
from pathlib import Path
Path('{self._ANSWER_FILE}').write_bytes(base64.b64decode('{answer_b64}'))
" """
        await self.sandbox_client.execute_command(sandbox_id, cmd)

    async def _wait_for_worker_ready(self, sandbox_id: str) -> None:
        """Wait for worker to signal ready."""
        wait_script = _RLM_READY_WAIT_SCRIPT.format(ready_flag=self._READY_FLAG)
        result = await self.sandbox_client.execute_command(sandbox_id, wait_script)
        if "failed to start" in result.stdout or "failed to start" in (result.stderr or ""):
            # Debug: get more info about why it failed
            debug_result = await self.sandbox_client.execute_command(
                sandbox_id,
                "ls -la /tmp/rlm* 2>&1; echo '---LOG---'; cat /tmp/rlm_worker.log 2>&1 || echo 'no log'; echo '---PS---'; ps aux 2>&1"
            )
            logger.error(f"RLM worker failed to start. Debug info:\n{debug_result.stdout}")
            raise RuntimeError(f"RLM worker failed to start: {debug_result.stdout[:500]}")

    # =========================================================================
    # Code Execution
    # =========================================================================

    def _extract_code_blocks(self, text: str) -> list[str]:
        """Extract Python code blocks from model output."""
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        return re.findall(pattern, text, re.DOTALL)

    async def _execute_code(self, sandbox_id: str, code: str) -> dict[str, Any]:
        """Execute code in sandbox worker and return result."""
        payload = {"code": code}
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

        result = await self.sandbox_client.execute_command(sandbox_id, command)
        if not result.stdout:
            return {
                "status": "error",
                "stdout": "",
                "stderr": result.stderr or "",
                "result": "Worker returned no output",
                "answer": {"ready": False, "content": ""},
            }
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "stdout": result.stdout,
                "stderr": result.stderr or "",
                "result": f"Failed to parse worker response: {e}",
                "answer": {"ready": False, "content": ""},
            }

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
            output = output[:self.max_output_length] + "\n... [output truncated]"

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

            # Build system prompt with sub-tool documentation
            base_system_prompt = self.custom_system_prompt or _RLM_SYSTEM_PROMPT
            sub_tools_docs = self._generate_sub_tools_documentation()
            system_prompt = base_system_prompt + sub_tools_docs

            messages = list(prompt)
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                # Append tool docs to existing system prompt
                messages[0] = {
                    "role": "system",
                    "content": messages[0]["content"] + sub_tools_docs,
                }
            return messages
        else:
            # Subsequent turns: use parent implementation
            return await super().get_prompt_messages(state)

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        """
        Parse code blocks from LLM response and execute in sandbox.
        Returns execution output as user message.
        """
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return [{"role": "user", "content": "Error: Sandbox not available"}]

        # Get the last assistant message
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_content = msg.get("content", "")
                break

        # Extract code blocks
        code_blocks = self._extract_code_blocks(last_content)

        if not code_blocks:
            # No code blocks - prompt for code
            if "answer" in last_content.lower() and "ready" in last_content.lower():
                return [{
                    "role": "user",
                    "content": 'Please set your answer using a Python code block:\n```python\nanswer["ready"] = True\nanswer["content"] = "your answer"\n```'
                }]
            else:
                return [{
                    "role": "user",
                    "content": "Please provide Python code in markdown code blocks to explore the context or set your answer."
                }]

        # Execute all code blocks
        all_outputs = []
        for i, code in enumerate(code_blocks):
            result = await self._execute_code(sandbox_id, code)
            output = self._format_execution_output(result)
            all_outputs.append(f"[Code block {i + 1}]\n{output}")

            # Check if answer is ready
            answer = result.get("answer", {})
            if answer.get("ready", False):
                state["final_answer"] = answer.get("content", "")
                logger.debug(f"Answer ready: {state['final_answer'][:100]}...")

        combined_output = "\n\n".join(all_outputs)
        return [{"role": "user", "content": f"Execution output:\n{combined_output}"}]

    # =========================================================================
    # Stop Conditions
    # =========================================================================

    @vf.stop
    async def answer_ready(self, state: State) -> bool:
        """Stop when model sets answer['ready'] = True."""
        return "final_answer" in state

    async def no_tools_called(self, state: State) -> bool:
        """Override ToolEnv's stop condition - RLMEnv uses code blocks, not OAI tools."""
        # Never stop due to "no tools called" - we use markdown code blocks instead
        return False

    # =========================================================================
    # Cleanup
    # =========================================================================

    @vf.cleanup
    async def cleanup_rlm_state(self, state: State):
        """Cleanup RLM-specific state."""
        rollout_id = state.get("rollout_id")
        if rollout_id and rollout_id in self.active_rollouts:
            del self.active_rollouts[rollout_id]

        # Decrement tunnel usage
        tunnel_url = state.get("tunnel_url")
        if tunnel_url:
            async with self._tunnel_lock:
                for tunnel in self._tunnels:
                    if tunnel["url"] == tunnel_url:
                        tunnel["active_rollouts"] = max(0, tunnel["active_rollouts"] - 1)
                        break

    async def post_rollout(self, state: State):
        """Read final answer from sandbox if not already set."""
        if "final_answer" in state:
            return

        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            state["final_answer"] = ""
            return

        try:
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                f'cat {self._ANSWER_FILE} 2>/dev/null || echo \'{{"content": ""}}\''
            )
            answer = json.loads(result.stdout.strip())
            state["final_answer"] = answer.get("content", "")
        except Exception as e:
            logger.warning(f"Failed to read RLM answer: {e}")
            state["final_answer"] = ""

# TODO: Improve system prompt
# TODO: Add logging for sub-LLM calls
# TODO: Experiment with putting the user query inside the `context`
