"""
Spawning Language Model (RLM) Environment with Protocol/MultiAgentEnv support.

Demonstrates the spawning pattern:
1. Give RLM an initial task
2. It can run code in REPL (call_python_repl)
3. It has a tool to spawn sub-instances (spawn_rlm via protocol.spawn())
4. Sub-RLMs get parent context plus task
5. Scored by MultiAgentRubric for reward propagation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import tempfile
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from openai.types.chat import ChatCompletionAssistantMessageParam

import verifiers as vf
from verifiers.envs.actor import Actor
from verifiers.envs.multiagent_env import MultiAgentEnv
from verifiers.rubrics.multiagent_rubric import MultiAgentRubric
from verifiers.types import Messages, RolloutInput, State
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.tool_utils import convert_func_to_oai_tool

if TYPE_CHECKING:
    from verifiers.envs.protocol import Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class RLMWorkerPaths:
    base_dir: str
    command_fifo: str
    response_fifo: str
    ready_flag: str
    worker_path: str
    context_file: str
    answer_file: str


@dataclass
class LocalRLMSession:
    rollout_id: str
    temp_dir: tempfile.TemporaryDirectory
    paths: RLMWorkerPaths
    worker_process: subprocess.Popen | None = None


# =============================================================================
# Worker Script - Simple REPL with extra_data and answer
# =============================================================================


_RLM_WORKER_SCRIPT = textwrap.dedent(
    '''
    import ast
    import contextlib
    import io
    import json
    import os
    import traceback
    from pathlib import Path

    COMMAND_FIFO = "{command_fifo}"
    RESPONSE_FIFO = "{response_fifo}"
    READY_FLAG = "{ready_flag}"
    CONTEXT_FILE = "{context_file}"
    ANSWER_FILE = "{answer_file}"

    def ensure_fifo(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
        os.mkfifo(path)

    for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):
        ensure_fifo(fifo_path)

    # Load extra_data from context file
    extra_data = None
    if Path(CONTEXT_FILE).exists():
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            extra_data = json.load(f).get("extra_data")

    # Initialize answer
    answer = {{"ready": False, "content": ""}}
    if Path(ANSWER_FILE).exists():
        with open(ANSWER_FILE, "r", encoding="utf-8") as f:
            answer = json.load(f)

    # Execution namespace
    namespace: dict[str, object] = {{
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "extra_data": extra_data,
        "answer": answer,
    }}

    # Signal ready
    Path(READY_FLAG).write_text("ready", encoding="utf-8")

    execution_count = 0

    while True:
        with open(COMMAND_FIFO, "r", encoding="utf-8") as cmd_file:
            payload = cmd_file.read()
        if not payload:
            continue
        request = json.loads(payload)
        if request.get("shutdown"):
            break

        code = request.get("code", "")
        seq = request.get("seq", 0)
        execution_count += 1

        result = {{
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": None,
            "execution_count": execution_count,
            "seq": seq,
            "answer": namespace.get("answer", {{"ready": False, "content": ""}}),
        }}

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                module_ast = ast.parse(code, mode="exec")
                body = list(module_ast.body)
                trailing_expr = None
                if body and isinstance(body[-1], ast.Expr):
                    trailing_expr = body.pop()
                if body:
                    exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), namespace, namespace)
                if trailing_expr is not None:
                    value = eval(compile(ast.Expression(trailing_expr.value), "<cell>", "eval"), namespace, namespace)
                    if value is not None:
                        result["result"] = repr(value)
        except Exception:
            result["status"] = "error"
            result["result"] = traceback.format_exc()

        result["stdout"] = stdout_buf.getvalue()
        result["stderr"] = stderr_buf.getvalue()
        result["answer"] = namespace.get("answer", {{"ready": False, "content": ""}})

        # Persist answer
        with open(ANSWER_FILE, "w", encoding="utf-8") as f:
            json.dump(result["answer"], f)

        with open(RESPONSE_FIFO, "w", encoding="utf-8") as resp_file:
            resp_file.write(json.dumps(result))
    '''
)


# =============================================================================
# System Prompt
# =============================================================================


_RLM_SYSTEM_PROMPT = """You are an RLM (Recursive Language Model) - an iterative Python REPL where you explore data step by step.

## Tools

- `call_python_repl`: Execute Python code. State persists across calls.
- `spawn_rlm`: Spawn sub-RLMs in parallel. Pass a list of tasks, get back JSON results. Use this to break down complex tasks into smaller sub-tasks.

## Available Variables

- `extra_data`: Input data to process
- `answer`: Dict with `answer["content"]` (your answer) and `answer["ready"]` (set True to finish)

## Workflow

1. Explore: `print(type(extra_data))` - see what you have
2. Process: Work step by step, see output before continuing
3. Answer: `answer["content"] = "result"` then `answer["ready"] = True`

**Important:** Never set `answer["ready"] = True` until you've seen execution output.
"""


# =============================================================================
# Helper Functions
# =============================================================================


def _build_worker_paths(base_dir: str) -> RLMWorkerPaths:
    return RLMWorkerPaths(
        base_dir=base_dir,
        command_fifo=os.path.join(base_dir, "rlm_cmd"),
        response_fifo=os.path.join(base_dir, "rlm_res"),
        ready_flag=os.path.join(base_dir, "rlm_ready"),
        worker_path=os.path.join(base_dir, "rlm_worker.py"),
        context_file=os.path.join(base_dir, "rlm_context.json"),
        answer_file=os.path.join(base_dir, "rlm_answer.json"),
    )


def _render_worker_script(paths: RLMWorkerPaths) -> str:
    return _RLM_WORKER_SCRIPT.format(
        command_fifo=paths.command_fifo,
        response_fifo=paths.response_fifo,
        ready_flag=paths.ready_flag,
        context_file=paths.context_file,
        answer_file=paths.answer_file,
    )


# =============================================================================
# Local Executor - Subprocess with FIFO communication
# =============================================================================


class LocalRLMExecutor:
    def __init__(self, code_timeout: int = 120):
        self.code_timeout = code_timeout
        self._sessions: dict[str, LocalRLMSession] = {}

    async def setup(self, state: State, extra_data: Any) -> None:
        """Create temp dir, write context, start worker."""
        rollout_id = state["rollout_id"]
        temp_dir = tempfile.TemporaryDirectory(prefix=f"rlm_{rollout_id}_")
        paths = _build_worker_paths(temp_dir.name)

        session = LocalRLMSession(
            rollout_id=rollout_id,
            temp_dir=temp_dir,
            paths=paths,
        )
        self._sessions[rollout_id] = session

        # Write context file
        Path(paths.context_file).write_text(
            json.dumps({"extra_data": extra_data}), encoding="utf-8"
        )
        Path(paths.answer_file).write_text(
            json.dumps({"ready": False, "content": ""}), encoding="utf-8"
        )

        # Write and start worker
        worker_script = _render_worker_script(paths)
        Path(paths.worker_path).write_text(worker_script, encoding="utf-8")

        process = subprocess.Popen(
            ["python", "-u", paths.worker_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        session.worker_process = process

        # Wait for ready
        await self._wait_for_ready(session)

    async def _wait_for_ready(self, session: LocalRLMSession, timeout: int = 30) -> None:
        start = asyncio.get_event_loop().time()
        while True:
            if Path(session.paths.ready_flag).exists():
                return
            if session.worker_process and session.worker_process.poll() is not None:
                raise vf.SandboxError("Worker exited before ready")
            if asyncio.get_event_loop().time() - start > timeout:
                raise vf.SandboxError("Worker failed to start")
            await asyncio.sleep(0.1)

    async def execute(self, code: str, state: State) -> dict[str, Any]:
        """Send code to worker, get result."""
        session = self._sessions.get(state["rollout_id"])
        if not session or not session.worker_process:
            raise vf.SandboxError("Session not initialized")
        if session.worker_process.poll() is not None:
            raise vf.SandboxError("Worker process not running")

        seq = state.get("_exec_seq", 0) + 1
        state["_exec_seq"] = seq

        def _do_io() -> str:
            payload = json.dumps({"code": code, "seq": seq})
            with open(session.paths.command_fifo, "w", encoding="utf-8") as f:
                f.write(payload)
            with open(session.paths.response_fifo, "r", encoding="utf-8") as f:
                return f.read()

        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(_do_io),
                timeout=self.code_timeout,
            )
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "result": f"Code execution timed out after {self.code_timeout}s",
                "answer": {"ready": False, "content": ""},
            }

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "result": f"Failed to parse response: {raw[:200]}",
                "answer": {"ready": False, "content": ""},
            }

    async def read_answer(self, state: State) -> str:
        session = self._sessions.get(state.get("rollout_id", ""))
        if not session:
            return ""
        try:
            content = Path(session.paths.answer_file).read_text(encoding="utf-8")
            return json.loads(content).get("content", "")
        except Exception:
            return ""

    async def cleanup(self, state: State) -> None:
        rollout_id = state.get("rollout_id")
        if not rollout_id:
            return
        session = self._sessions.pop(rollout_id, None)
        if not session:
            return
        self._stop_worker(session)
        session.temp_dir.cleanup()

    def _stop_worker(self, session: LocalRLMSession) -> None:
        if not session.worker_process:
            return
        try:
            if os.name != "nt":
                os.killpg(session.worker_process.pid, signal.SIGTERM)
            else:
                session.worker_process.terminate()
            session.worker_process.wait(timeout=5)
        except Exception:
            try:
                if os.name != "nt":
                    os.killpg(session.worker_process.pid, signal.SIGKILL)
                else:
                    session.worker_process.kill()
            except Exception:
                pass
        session.worker_process = None

    async def teardown(self) -> None:
        for session in list(self._sessions.values()):
            self._stop_worker(session)
            session.temp_dir.cleanup()
        self._sessions.clear()


# =============================================================================
# MultiAgentToolEnv - Tool handling on MultiAgentEnv
# =============================================================================


class MultiAgentToolEnv(MultiAgentEnv):
    """ToolEnv that extends MultiAgentEnv."""

    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        **kwargs,
    ):
        self.tools: list[Callable] = tools or []
        self.max_turns = max_turns
        self.oai_tools = [convert_func_to_oai_tool(tool) for tool in self.tools]
        self.tool_map: dict[str, Callable] = {
            getattr(tool, "__name__", tool.__class__.__name__): tool
            for tool in self.tools
        }
        self.skipped_args: dict[str, list[str]] = {}

        super().__init__(oai_tools=self.oai_tools, max_turns=max_turns, **kwargs)

    def add_tool(self, tool: Callable, args_to_skip: list[str] | None = None):
        """Add a tool, optionally hiding arguments from the agent's view."""
        args_to_skip = args_to_skip or []
        self.tools.append(tool)

        # Build OAI tool schema, filtering skipped args
        import inspect
        sig = inspect.signature(tool)
        filtered_params = [
            p for n, p in sig.parameters.items()
            if n not in args_to_skip and n != "self"
        ]
        filtered_sig = sig.replace(parameters=filtered_params)

        def wrapper(*args, **kw):
            return tool(*args, **kw)

        wrapper.__name__ = tool.__name__
        wrapper.__doc__ = tool.__doc__
        wrapper.__signature__ = filtered_sig
        wrapper.__annotations__ = {
            k: v for k, v in getattr(tool, "__annotations__", {}).items()
            if k not in args_to_skip
        }

        oai_tool = convert_func_to_oai_tool(wrapper)
        if self.oai_tools is None:
            self.oai_tools = []
        self.oai_tools.append(oai_tool)

        tool_name = tool.__name__
        self.tool_map[tool_name] = tool
        self.skipped_args[tool_name] = args_to_skip

    def update_tool_args(
        self, tool_name: str, tool_args: dict, messages: Messages, state: State, **kwargs
    ) -> dict:
        """Override to inject state-based args."""
        return tool_args

    @vf.stop
    async def no_tools_called(self, state: State) -> bool:
        if len(state["trajectory"]) == 0:
            return False
        last_msg = state["trajectory"][-1]["completion"][-1]
        return last_msg["role"] == "assistant" and "tool_calls" not in last_msg

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> vf.Message:
        tool_func = self.tool_map[tool_name]
        result = await maybe_await(tool_func, **tool_args)
        return cast(vf.Message, {"role": "tool", "content": str(result), "tool_call_id": tool_call_id})

    async def env_response(self, messages: Messages, state: State, **kwargs) -> Messages:
        assert isinstance(messages, list) and "tool_calls" in messages[-1]
        tool_messages = []
        last_msg = cast(ChatCompletionAssistantMessageParam, messages[-1])

        for tool_call in last_msg.get("tool_calls", []):
            tool_call_id: str = tool_call.get("id", "")
            try:
                tool_name: str = tool_call.get("function", {}).get("name", "")
                tool_args: dict = json.loads(tool_call.get("function", {}).get("arguments", ""))
            except Exception as e:
                tool_messages.append(cast(vf.Message, {
                    "role": "tool", "content": str(e), "tool_call_id": tool_call_id
                }))
                continue

            tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            try:
                tool_msg = await self.call_tool(tool_name, tool_args, tool_call_id)
                tool_messages.append(tool_msg)
            except Exception as e:
                tool_messages.append(cast(vf.Message, {
                    "role": "tool", "content": str(e), "tool_call_id": tool_call_id
                }))

        return tool_messages

    def get_initial_actor(self, state: State) -> str:
        return "rlm"

    def get_next_actor(self, state: State) -> str:
        return "rlm"


# =============================================================================
# SpawningRLMEnv
# =============================================================================


class SpawningRLMEnv(MultiAgentToolEnv):
    """
    RLM environment using Protocol.spawn() for spawning sub-agents.

    Core pattern:
    - call_python_repl: Execute code in persistent REPL
    - spawn_rlm: Spawn sub-RLMs via protocol.spawn()
    - MultiAgentRubric propagates rewards through execution tree
    """

    # Multi agent specific fields
    name = "spawning_rlm"
    # We only have one actor, the RLM
    actors = ["rlm"]
    protocol: "Protocol"
    # Only have one actor, so just hardcode it as the rlm
    current_actor: str = "rlm"

    def __init__(
        self,
        max_iterations: int = 50,
        max_output_length: int = 8192,
        code_timeout: int = 120,
        system_prompt: str | None = None,
        **kwargs,
    ):
        self.max_iterations = max_iterations
        self.max_output_length = max_output_length
        self.code_timeout = code_timeout
        self.custom_system_prompt = system_prompt

        super().__init__(max_turns=max_iterations, **kwargs)

        self._executor = LocalRLMExecutor(code_timeout=code_timeout)
        self.add_tool(self.call_python_repl, args_to_skip=["state"])
        self.add_tool(self.spawn_rlm, args_to_skip=["state"])

    # =========================================================================
    # State Management
    # =========================================================================

    async def setup_state(self, state: State, **kwargs) -> State:
        state = await super().setup_state(state, **kwargs)

        state["rollout_id"] = f"rlm_{uuid.uuid4().hex[:8]}"
        state["_exec_seq"] = 0

        # Get context from RolloutInput.extra_data
        input_extra = state.get("input", {}).get("extra_data", {})
        state["rlm_context"] = input_extra.get("rlm_context", {})
        state["is_child"] = input_extra.get("is_child", False)
        state["child_states"] = []

        # Get extra_data from dataset row
        # Dataset columns are in state["input"], but State only forwards specific fields
        input_data = state.get("input", {})
        extra_data = input_data.get("context") or input_data.get("extra_data")
        if extra_data is None:
            # Fallback: check info dict
            info = state.get("info", {})
            if isinstance(info, dict):
                extra_data = info.get("context", info.get("extra_data"))

        # Build system prompt
        state["rlm_system_prompt"] = self.custom_system_prompt or _RLM_SYSTEM_PROMPT

        # Start worker
        await self._executor.setup(state, extra_data)

        return state

    def update_tool_args(
        self, tool_name: str, tool_args: dict, messages: Messages, state: State, **kwargs
    ) -> dict:
        if tool_name in ("call_python_repl", "spawn_rlm"):
            return {**tool_args, "state": state}
        return tool_args

    # =========================================================================
    # Prompt Building
    # =========================================================================

    async def get_prompt_messages(self, state: State) -> Messages:
        if len(state["trajectory"]) == 0:
            prompt = state.get("prompt", [])
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]

            messages = list(prompt)
            system_prompt = state.get("rlm_system_prompt", _RLM_SYSTEM_PROMPT)

            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})

            return cast(Messages, messages)
        return await super().get_prompt_messages(state)

    # =========================================================================
    # Tools
    # =========================================================================

    async def call_python_repl(self, code: str, state: State) -> str:
        """
        Execute Python code in a persistent REPL.

        Available:
        - `extra_data`: Input data to process
        - `answer["content"]`: Set your answer here
        - `answer["ready"]`: Set True to finish

        Args:
            code: Python code to execute
        """
        result = await self._executor.execute(code, state)
        output = self._format_output(result)

        # Check if answer ready
        answer = result.get("answer", {})
        if answer.get("ready"):
            state["final_answer"] = answer.get("content", "")

        return output

    def _format_output(self, result: dict[str, Any]) -> str:
        parts = []
        if result.get("stdout"):
            parts.append(result["stdout"].rstrip())
        if result.get("stderr"):
            parts.append(f"stderr:\n{result['stderr'].rstrip()}")

        status = result.get("status")
        res = result.get("result")
        if status == "error" and res:
            parts.append(res.rstrip())
        elif status == "ok" and res:
            parts.append(f"Out[{result.get('execution_count', 0)}]: {res}")

        output = "\n".join(parts) if parts else "(no output)"
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + "\n... [truncated]"
        return output

    async def spawn_rlm(self, tasks: list[str], state: State) -> str:
        """
        Spawn sub-RLMs to solve tasks in parallel.

        Args:
            tasks: List of task descriptions to spawn (runs concurrently)

        Returns:
            JSON array of results: [{"task": "...", "answer": "..."}, ...]
        """
        if not hasattr(self, "protocol") or self.protocol is None:
            return json.dumps([{"error": "Protocol not configured"}])

        parent_ctx = state.get("rlm_context", {})
        child_inputs = [
            RolloutInput(
                prompt=[{"role": "user", "content": t}],
                example_id=state.get("example_id", 0),
                task=self.name,
                extra_data={
                    "rlm_context": {
                        **parent_ctx,
                        "parent_task": t,
                        "depth": parent_ctx.get("depth", 0) + 1,
                    },
                    "is_child": True,
                },
            )
            for t in tasks
        ]

        try:
            child_states = await self.protocol.spawn(child_inputs, score=False)
            results = []
            for t, cs in zip(tasks, child_states):
                if cs.get("final_answer"):
                    state.setdefault("child_states", []).append(cs)
                    results.append({"task": t, "answer": str(cs["final_answer"])})
                else:
                    results.append({"task": t, "error": str(cs.get("error", "No answer"))})
            return json.dumps(results)
        except Exception as e:
            logger.error(f"spawn_rlm failed: {e}")
            return json.dumps([{"error": str(e)}])

    # =========================================================================
    # Stop Conditions
    # =========================================================================

    @vf.stop
    async def answer_ready(self, state: State) -> bool:
        return "final_answer" in state

    # =========================================================================
    # Cleanup
    # =========================================================================

    @vf.cleanup
    async def cleanup_state(self, state: State):
        await self._executor.cleanup(state)

    @vf.teardown
    async def teardown_executor(self):
        await self._executor.teardown()


# =============================================================================
# Scoring
# =============================================================================


async def exact_answer(state: State, answer: str, **_kwargs) -> float:
    """Score: 1.0 if final_answer matches expected answer."""
    final = str(state.get("final_answer") or "").strip()
    expected = str(answer).strip()
    return 1.0 if final == expected else 0.0


# =============================================================================
# Protocol Factory
# =============================================================================


def create_spawning_rlm_protocol(
    dataset=None,
    eval_dataset=None,
    system_prompt: str | None = None,
    **env_kwargs,
) -> "Protocol":
    """
    Create Protocol for spawning RLM usage.

    The dataset is registered on the Protocol, not the environment.
    This follows the multi-agent pattern where Protocol owns the dataset.

    Args:
        dataset: Training dataset (registered on Protocol)
        eval_dataset: Evaluation dataset (registered on Protocol)
        system_prompt: Custom system prompt for the RLM actor
        **env_kwargs: Additional arguments passed to SpawningRLMEnv
    """
    from verifiers.envs.protocol import Protocol

    rlm_actor = Actor(
        id="rlm",
        system_prompt=system_prompt or _RLM_SYSTEM_PROMPT,
    )

    rubric = MultiAgentRubric(funcs=[exact_answer])

    # Environment doesn't need dataset - Protocol owns it
    rlm_env = SpawningRLMEnv(
        rubric=rubric,
        system_prompt=system_prompt,
        **env_kwargs,
    )

    # Dataset registered on Protocol
    return Protocol(
        actors=[rlm_actor],
        envs=[rlm_env],
        dataset=dataset,
        eval_dataset=eval_dataset,
    )
