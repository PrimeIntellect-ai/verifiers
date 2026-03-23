"""
Simplified Recursive Language Model (SRLM) Environment.

Like RLMEnv but with single-shot sub-LLM calls (no multi-turn tool-calling loop).
Provides both ``llm(prompt)`` and ``llm_batch(prompts)`` as root REPL tools.

Based on: https://github.com/alexzhang13/rlm

Architecture:
- REPL loop runs in the framework (StatefulToolEnv / MultiTurnEnv pattern)
- Code execution runs in a sandbox via a persistent worker (inherited from RLMEnv)
- Sub-LLM calls from worker code are intercepted via HTTP proxy (inherited)
- ``llm(prompt)`` → single sub-LLM API call → response string
- ``llm_batch(prompts)`` → parallel single-shot sub-LLM calls → list of response strings
- No tool-calling loop for sub-LLMs; each call is exactly one API request

Key differences from RLMEnv:
- Sub-LLM calls are always single-shot (no tool calling for sub-LLMs)
- ``llm(prompt)`` root tool for single sub-LLM calls (in addition to ``llm_batch``)
- No ``sub_tools`` concept; sub-LLMs have no tool access

Metadata & training:
- Root model steps have the main trajectory_id (trained on)
- Sub-LLM steps get separate trajectory_ids (via ``include_sub_llm_in_trajectory``)
- All RLM metric keys are preserved for training pipeline compatibility
- ``llm()`` and ``llm_batch()`` calls flow through ``_run_sub_llm_request`` which
  handles trajectory steps, token tracking, and metric updates
"""

import logging
import re
import textwrap
import uuid
from typing import Any, Callable, Literal

from verifiers.envs.experimental.rlm_env import (
    RLMEnv,
    SubLLMResult,
    SubLLMTurn,
    _clone_messages,
    _extract_tokens_from_response,
)
from verifiers.types import (
    Messages,
    Response,
    State,
    UserMessage,
)

logger = logging.getLogger(__name__)


def _parse_text_final(text: str) -> str | None:
    """Parse FINAL(...) from model text (outside REPL code).

    Matches the RLM library's ``find_final_answer`` regex for the FINAL pattern.
    Returns the literal text content, or None if not found.
    """
    # Must be at start of line; greedy to capture nested parens.
    pattern = r"^\s*FINAL\((.*)\)\s*$"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# Both llm and llm_batch are reserved (cannot be overridden by user tools).
_SRLM_FIXED_REPL_TOOL_NAMES = frozenset({"llm", "llm_batch"})

# ---------------------------------------------------------------------------
# System prompt — adapted from https://github.com/alexzhang13/rlm
# (RLM_SYSTEM_PROMPT in rlm/utils/prompts.py)
#
# Key adaptations for SRLM sandbox environment:
#   - llm_query(prompt)          → llm(prompt)
#   - llm_query_batched(prompts) → llm_batch(prompts)
#   - rlm_query / rlm_query_batched removed (single-shot only)
#   - context variable           → filesystem (context.txt / context.json)
#   - ```repl``` blocks          → call_python_repl tool
#   - FINAL / FINAL_VAR / SHOW_VARS preserved from RLM repo
# ---------------------------------------------------------------------------
_SRLM_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are tasked with answering a query with associated context. You can access, \
transform, and analyze this context interactively in a REPL environment that can \
query sub-LLMs, which you are strongly encouraged to use as much as possible. \
You will be queried iteratively until you provide a final answer.

## REPL environment

Use the `call_python_repl` tool to execute Python code. The REPL maintains state \
across calls — variables, imports, and function definitions persist.

The environment provides:
1. **A `context` variable** that contains extremely important information about your \
query. You should check the content of the `context` variable to understand what you \
are working with. Make sure you look through it sufficiently as you answer your query.
   ```python
   print(type(context), len(context))
   print(context[:500])
   ```
   A filesystem is also available; explore it with `os.listdir(".")` as needed.

2. **`llm(prompt)`** — make a single sub-LLM call. Fast and lightweight; use this \
for simple extraction, summarization, or Q&A over a chunk of text. The sub-LLM can \
handle large context windows, so don't be afraid to pass substantial chunks.

3. **`llm_batch(prompts)`** — run multiple `llm()` calls concurrently; returns \
`list[str]` in the same order as input prompts. Much faster than sequential `llm()` \
calls for independent queries.

4. **`SHOW_VARS()`** — returns all variables you have created in the REPL. Use this \
to check what variables exist before using FINAL_VAR.

5. **`print()`** — view output from your code and continue your reasoning.

## When to use `llm()`:
- Use `llm()` for simple, one-shot tasks: extracting info from a chunk, summarizing \
text, answering a factual question, classifying content. These are fast single LLM calls.

## Strategy guidance

**Breaking down problems:** Break problems into digestible components — whether \
that means chunking or summarizing a large context, or decomposing a hard task into \
easier sub-problems and delegating them via `llm()` / `llm_batch()`. Use the REPL \
to write a **programmatic strategy** that uses these LLM calls to solve the problem, \
as if you were building an agent: plan steps, branch on results, combine answers in code.

**REPL for computation:** Use the REPL to compute programmatic steps (e.g. \
`math.sin(x)`, distances, physics formulas) and chain those results into an LLM call. \
For complex math or physics, compute intermediate quantities in code and pass the \
numbers to the LLM for interpretation or the final answer. Example:
```python
import math
v_parallel = pitch * (q * B) / (2 * math.pi * m)
v_perp = R * (q * B) / m
theta_deg = math.degrees(math.atan2(v_perp, v_parallel))
final_answer = llm(f"Computed entry angle: {theta_deg:.2f} deg. State the answer.")
```

**Use sub-LLMs for semantic analysis:** You will only see truncated outputs from \
code execution, so use `llm()` on variables you want to analyze. This is especially \
useful for analyzing the semantics of context. Use variables as buffers to build up \
your final answer.

**Explore context thoroughly:** Make sure to explicitly look through the entire \
context before answering. Break the context into digestible pieces: figure out a \
chunking strategy, query an LLM per chunk and save answers to a buffer, then query \
an LLM over the buffers to produce your final answer.

**Leverage large context windows:** Your sub-LLMs are powerful — they can handle \
very large context. A viable strategy is to feed many documents per sub-LLM query. \
Analyze your input data and see if it fits in just a few sub-LLM calls!

## Examples

**Chunking a long context and querying per chunk:**
```python
chunk_size = len(context) // 10
chunks = [context[i*chunk_size:(i+1)*chunk_size] for i in range(10)]
prompts = [f"Answer: {{query}}. Evidence:\\n{{c}}" for c in chunks]
answers = llm_batch(prompts)
for i, a in enumerate(answers):
    print(f"Chunk {i}: {a}")
```

**Aggregating sub-LLM results:**
```python
final = llm(f"Given these per-chunk answers, provide the final answer to: {query}\\n" + "\\n".join(answers))
FINAL(final)
```

**Iterative exploration:**
```python
query = "Did Gryffindor win the House Cup because they led?"
buffers = []
for i, section in enumerate(sections):
    buf = llm(f"Section {i}/{len(sections)}. Gather info for: {query}\\n{section}")
    buffers.append(buf)
    print(f"Section {i}: {buf}")
my_answer = llm(f"Based on all sections: {query}\\n" + "\\n".join(buffers))
FINAL(my_answer)
```

**Branching on results:**
```python
trend = llm(f"Analyze this dataset — up, down, or stable: {data}")
if "up" in trend.lower():
    rec = "Consider increasing exposure."
elif "down" in trend.lower():
    rec = "Consider hedging."
else:
    rec = "Hold position."
my_answer = llm(f"Trend={trend}, recommendation={rec}. One-sentence summary.")
FINAL(my_answer)
```

## Providing your final answer

IMPORTANT: When you are done with the iterative process, you MUST provide a final \
answer using the FINAL function. Do not use it unless you have completed your task. \
You have two options:
1. Use `FINAL(your_answer)` to provide the answer directly
2. Use `FINAL_VAR(variable_name)` to return a variable you created in the REPL

WARNING - COMMON MISTAKE: FINAL_VAR retrieves an EXISTING variable. You MUST create \
and assign the variable in a tool call FIRST, then call FINAL_VAR in a SEPARATE step.
- WRONG: Calling `FINAL_VAR("my_answer")` without first creating `my_answer`
- CORRECT: First create `my_answer = "the result"` in one tool call, then call \
`FINAL_VAR("my_answer")` in the next

If you're unsure what variables exist, call `SHOW_VARS()` to see all available variables.

## Critical rules

1. **One step at a time** — make small tool calls, see output, then continue.
2. **NEVER** call `FINAL()` until you have seen execution output and are confident.
3. **Use `llm_batch()`** for multiple independent prompts — it's much faster than \
sequential `llm()` calls.
4. Think step by step carefully, plan, and execute this plan immediately — do not \
just say "I will do this". Output to the REPL and sub-LLMs as much as possible. \
Remember to explicitly answer the original query in your final answer.
"""
)


class SRLMEnv(RLMEnv):
    """
    Simplified Recursive Language Model Environment.

    Extends RLMEnv with single-shot sub-LLM calls and both ``llm()`` and
    ``llm_batch()`` root tools.  All sandbox execution, interception server,
    worker scripts, metric tracking, and training trajectory support is
    inherited from RLMEnv.

    Sub-LLM calls are always single-shot: one API request per call, no
    tool-calling loop.  This means sub-LLMs cannot use tools; they simply
    receive a prompt and return a text response.

    Args:
        tools: List of tools shared by the root REPL (added after fixed tools
               in documentation order). These are available as callable
               functions inside the REPL sandbox.
        root_tools: List of additional tools available only to the root REPL.
        sub_model: Model to use for sub-LLM calls (defaults to same as root).
        sub_prompt_verbosity: Verbosity of the sub-LLM system prompt:
               "light", "medium", or "heavy".
        root_prompt_verbosity: Verbosity of the root-LLM system prompt.
        max_iterations: Maximum REPL iterations before stopping.
        max_output_length: Maximum length of code execution output.
        max_sub_llm_parallelism: Max concurrent sub-LLM calls in llm_batch.
        sub_llm_stagger_ms: Per-call stagger delay (ms) within llm_batch.
        sub_llm_stagger_jitter_ms: Random jitter (ms) added to stagger delay.
        context_key: Key in info containing legacy context data.
        context_dir_key: Key in info containing directory path.
        system_prompt: Custom system prompt (default: RLM standard prompt).
        repl_language: REPL language: "bash" or "python".
        interception_host: Hostname/IP for interception server.
        interception_port: Port for interception server (0 = ephemeral).
        interception_url: Optional base URL for interception (sandbox only).
        pip_install_packages: Space-separated packages to install in sandbox.
        include_sub_llm_in_trajectory: Whether to include sub-LLM calls as
               trajectory steps (for training on sub-calls).
        context_warning_threshold: Fraction of max_seq_len at which to warn.
        max_startup_wait_seconds: Max seconds to wait for worker startup.
        code_execution_timeout: Timeout in seconds for code execution.
        abort_on_code_timeout: If True, abort rollout on code timeout.
        retain_filesystem_after_rollout: If True, keep filesystem after rollout.
        filesystem_copy_max_bytes: Max bytes for context directory copy.
        sandbox_docker_image: Docker image for sandbox.
        sandbox_start_command: Start command for sandbox.
        sandbox_cpu_cores: Sandbox CPU cores.
        sandbox_memory_gb: Sandbox memory in GB.
        sandbox_disk_size_gb: Sandbox disk size in GB.
        sandbox_gpu_count: Sandbox GPU count.
        sandbox_timeout_minutes: Sandbox timeout in minutes.
        sandbox_environment_vars: Extra environment vars for sandbox.
        sandbox_team_id: Optional team id for sandbox.
        sandbox_advanced_configs: Optional advanced configs for sandbox.
        sandbox_labels: Optional labels for sandbox.
        sandbox_client_max_workers: Sandbox client pool size.
        sandbox_client_max_connections: Sandbox client max connections.
        sandbox_client_max_keepalive_connections: Sandbox client keepalive.
        **kwargs: Additional arguments passed to StatefulToolEnv.
    """

    def __init__(
        self,
        tools: list[Callable] | None = None,
        root_tools: list[Callable] | None = None,
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
        interception_host: str | None = None,
        interception_port: int = 0,
        interception_url: str | None = None,
        pip_install_packages: str = "",
        include_sub_llm_in_trajectory: bool = False,
        context_warning_threshold: float = 0.80,
        max_startup_wait_seconds: int = 120,
        code_execution_timeout: int = 120,
        abort_on_code_timeout: bool = False,
        retain_filesystem_after_rollout: bool = False,
        filesystem_copy_max_bytes: int | None = 1_000_000_000,
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
        **kwargs: Any,
    ):
        # Use the RLM-derived system prompt by default (not RLMEnv's built-in).
        if system_prompt is None:
            system_prompt = _SRLM_SYSTEM_PROMPT

        # SRLM always uses single-shot sub-LLM calls: no sub_tools, 1 turn.
        super().__init__(
            tools=tools,
            root_tools=root_tools,
            sub_tools=None,
            sub_tool_max_turns=1,
            sub_model=sub_model,
            sub_prompt_verbosity=sub_prompt_verbosity,
            root_prompt_verbosity=root_prompt_verbosity,
            max_iterations=max_iterations,
            max_output_length=max_output_length,
            max_sub_llm_parallelism=max_sub_llm_parallelism,
            sub_llm_stagger_ms=sub_llm_stagger_ms,
            sub_llm_stagger_jitter_ms=sub_llm_stagger_jitter_ms,
            context_key=context_key,
            context_dir_key=context_dir_key,
            system_prompt=system_prompt,
            repl_language=repl_language,
            interception_host=interception_host,
            interception_port=interception_port,
            interception_url=interception_url,
            pip_install_packages=pip_install_packages,
            include_sub_llm_in_trajectory=include_sub_llm_in_trajectory,
            context_warning_threshold=context_warning_threshold,
            max_startup_wait_seconds=max_startup_wait_seconds,
            code_execution_timeout=code_execution_timeout,
            abort_on_code_timeout=abort_on_code_timeout,
            retain_filesystem_after_rollout=retain_filesystem_after_rollout,
            filesystem_copy_max_bytes=filesystem_copy_max_bytes,
            sandbox_docker_image=sandbox_docker_image,
            sandbox_start_command=sandbox_start_command,
            sandbox_cpu_cores=sandbox_cpu_cores,
            sandbox_memory_gb=sandbox_memory_gb,
            sandbox_disk_size_gb=sandbox_disk_size_gb,
            sandbox_gpu_count=sandbox_gpu_count,
            sandbox_timeout_minutes=sandbox_timeout_minutes,
            sandbox_environment_vars=sandbox_environment_vars,
            sandbox_team_id=sandbox_team_id,
            sandbox_advanced_configs=sandbox_advanced_configs,
            sandbox_labels=sandbox_labels,
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_client_max_connections=sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=sandbox_client_max_keepalive_connections,
            **kwargs,
        )

    # =========================================================================
    # Worker Script Customization (FINAL / FINAL_VAR / SHOW_VARS)
    # =========================================================================

    def customize_worker_script(self, script: str, state: State) -> str:
        """Inject RLM-compatible helpers into the worker namespace.

        Adds:
        - ``FINAL()``, ``FINAL_VAR()``, ``SHOW_VARS()`` (matching the RLM library)
        - ``context`` variable loaded from filesystem (like ``LocalREPL.load_context``)
        - ``_restore_scaffold()`` called after every code execution to protect
          reserved names from being overwritten by model code
        """
        # -- 1. FINAL / FINAL_VAR / SHOW_VARS + context loading -----------------
        # Injected before the ready flag, after namespace + tools are set up.
        pre_ready_injection = textwrap.dedent("""\

            def _FINAL(value):
                answer["ready"] = True
                answer["content"] = str(value)
                return str(value)

            def _FINAL_VAR(name):
                if not isinstance(name, str):
                    return _FINAL(name)
                name = name.strip().strip("\\\"'")
                if name in namespace:
                    value = str(namespace[name])
                    answer["ready"] = True
                    answer["content"] = value
                    return value
                available = [k for k in namespace if not k.startswith("_")
                             and k not in _SCAFFOLD_NAMES]
                if available:
                    return (f"Error: Variable '{name}' not found. "
                            f"Available variables: {available}. "
                            f"You must create and assign a variable BEFORE calling FINAL_VAR on it.")
                return (f"Error: Variable '{name}' not found. "
                        f"No variables have been created yet. "
                        f"You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it.")

            def _SHOW_VARS():
                available = {k: type(v).__name__ for k, v in namespace.items()
                             if not k.startswith("_") and k not in _SCAFFOLD_NAMES}
                if not available:
                    return "No variables created yet."
                return f"Available variables: {available}"

            _SCAFFOLD_NAMES = {"__name__", "FINAL", "FINAL_VAR", "SHOW_VARS",
                               "answer", "extra_data", "context"}
            _SCAFFOLD_REFS = {}

            namespace["FINAL"] = _FINAL
            namespace["FINAL_VAR"] = _FINAL_VAR
            namespace["SHOW_VARS"] = _SHOW_VARS

            # Load context from filesystem into namespace (matching RLM LocalREPL.load_context).
            # The worker cwd is already set to fs_root at this point.
            _context_loaded = False
            for _ctx_name in ["context.txt", "context.json"]:
                _ctx_path = os.path.join(os.getcwd(), _ctx_name)
                if os.path.isfile(_ctx_path):
                    with open(_ctx_path, "r", encoding="utf-8") as _f:
                        if _ctx_name.endswith(".json"):
                            namespace["context"] = json.load(_f)
                        else:
                            namespace["context"] = _f.read()
                    _context_loaded = True
                    break

            # Save references to scaffold values for _restore_scaffold.
            for _sn in list(_SCAFFOLD_NAMES) + list(ROOT_TOOL_NAMES):
                if _sn in namespace:
                    _SCAFFOLD_REFS[_sn] = namespace[_sn]

            def _restore_scaffold():
                for _name, _ref in _SCAFFOLD_REFS.items():
                    namespace[_name] = _ref

        """)

        # -- 2. _restore_scaffold() after each code execution --------------------
        # The worker's exec block ends with:
        #   result["stdout"] = stdout_buffer.getvalue()
        # We inject _restore_scaffold() right after the try/except closes.
        post_exec_injection = "\n    _restore_scaffold()\n"

        # Insert FINAL/context/scaffold before ready flag.
        ready_marker = 'Path(READY_FLAG).write_text("ready"'
        if ready_marker in script:
            script = script.replace(ready_marker, pre_ready_injection + ready_marker)

        # Insert _restore_scaffold() after stdout/stderr capture.
        # The line after the except block is: result["stdout"] = stdout_buffer.getvalue()
        restore_marker = '    result["stdout"] = stdout_buffer.getvalue()'
        if restore_marker in script:
            script = script.replace(
                restore_marker,
                post_exec_injection + restore_marker,
            )

        return script

    # =========================================================================
    # REPL Tool Override (updated docstring for FINAL/FINAL_VAR)
    # =========================================================================

    async def call_python_repl(self, code: str, state: Any) -> str:
        """
        Execute Python code in a persistent REPL environment.

        The REPL maintains state across calls and provides access to:

        - `context`: Variable pre-loaded with the context data (from context.txt or context.json).
        - Files in the working directory.

        - `llm(prompt)`: Make a single sub-LLM call for help with a subtask.
        - `llm_batch(prompts)`: Make sub-LLM calls on multiple prompts in parallel.

        - `FINAL(value)`: Provide your final answer directly. Terminates execution.
        - `FINAL_VAR(variable_name)`: Return an existing REPL variable as your answer.
        - `SHOW_VARS()`: List all variables in the REPL namespace.

        Reserved names (context, llm, llm_batch, FINAL, FINAL_VAR, SHOW_VARS, answer)
        are protected and restored after each execution.

        Args:
            code: Python code to execute in the persistent REPL

        Returns:
            Execution output including stdout, stderr, and expression results
        """
        return await self._call_repl(
            code,
            state,
            ready_instruction="Please finalize your answer soon using FINAL(your_answer).",
            append_execution_time=True,
        )

    # =========================================================================
    # Text-based FINAL detection
    # =========================================================================

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ):
        """Add model response, then check text for FINAL(...) outside of REPL.

        Inside the REPL, ``FINAL(x)`` is a Python function call — ``x`` is
        evaluated as an expression (variable value, string literal, etc.).

        Outside the REPL (in the model's text response), ``FINAL(text)``
        captures the literal text content, matching the RLM library's behavior.
        """
        await super().add_model_response(state, prompt_messages, response)

        # If a REPL-based FINAL already fired, don't override it.
        if "final_answer" in state:
            return

        content = response.message.content
        if isinstance(content, str):
            final = _parse_text_final(content)
            if final is not None:
                state["final_answer"] = final
                logger.debug("FINAL detected in model text: %s...", final[:100])

    # =========================================================================
    # Root Tools: llm() and llm_batch()
    # =========================================================================

    def _build_fixed_root_tools(self) -> list[Callable]:
        """Return fixed root REPL tools: ``llm`` and ``llm_batch``.

        Both are single-shot: one API call per prompt, no tool-calling loop.
        ``llm(prompt)`` is a convenience for single calls; ``llm_batch(prompts)``
        handles parallel calls with staggering.

        These are injected into the sandbox REPL namespace as callable functions
        (Python) or shell commands (Bash).
        """

        async def llm(prompt: str) -> str:
            """
            Call the sub-LLM on a single prompt.

            - Input: a prompt string.
            - Output: the sub-LLM's response as a string.
            - Use this for one-off sub-tasks (summarization, reasoning, etc.).
            - For multiple prompts, prefer ``llm_batch()`` for parallelism.
            """
            context = self._root_tool_context_var.get()
            if context is None:
                raise RuntimeError(
                    "llm called outside of a tool request context."
                )
            return await self._root_llm_single(context, prompt)

        async def llm_batch(prompts: list[str]) -> list[str]:
            """
            Call the sub-LLM on multiple prompts in parallel.

            - Input: a list of prompt strings.
            - Output: a list of responses in the same order as the input prompts.
            - Use this inside the REPL to get help on sub-tasks.
            - Prefer this over sequential ``llm()`` calls for better throughput.
            """
            context = self._root_tool_context_var.get()
            if context is None:
                raise RuntimeError(
                    "llm_batch called outside of a tool request context."
                )
            results, _ = await self._root_llm_batch(context, prompts)
            return results

        llm.__name__ = "llm"
        llm_batch.__name__ = "llm_batch"
        return [llm, llm_batch]

    # =========================================================================
    # Single-Shot Sub-LLM Execution
    # =========================================================================

    async def _run_sub_llm(
        self,
        state: State,
        client: Any,
        model: str,
        messages: Messages,
    ) -> SubLLMResult:
        """Run a sub-LLM call: always single-shot, no tool-calling loop.

        Overrides RLMEnv._run_sub_llm to enforce single-shot behavior
        regardless of whether shared tools exist. Each sub-LLM call is
        exactly one API request.
        """
        response = await self._call_sub_llm_api(state, client, model, messages)
        if response is None:
            return self._make_timeout_result([], 0, 0, 0, 0)

        prompt_tokens, completion_tokens = _extract_tokens_from_response(
            response
        )
        content = response.message.content
        final_content = content if isinstance(content, str) else ""

        return SubLLMResult(
            final_content=final_content,
            turns=[
                SubLLMTurn(
                    prompt_messages=_clone_messages(messages),
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

    async def _root_llm_single(
        self,
        context: dict[str, Any],
        prompt: str,
    ) -> str:
        """Run a single sub-LLM call for root REPL usage.

        Routes through ``_run_sub_llm_request`` which handles system prompt
        injection, metric tracking, trajectory steps, and boxed answer
        extraction — identical to how ``_root_llm_batch`` handles each
        prompt.
        """
        client = context.get("client")
        sub_model = context.get("sub_model") or context.get("model")
        state_ref = context.get("state")
        parent_turn = context.get("parent_turn", 0)
        if not client or not sub_model or state_ref is None:
            raise RuntimeError("Sub-LLM context is not available.")

        batch_id = uuid.uuid4().hex[:8]
        request_id = uuid.uuid4().hex[:8]

        messages: Messages = [UserMessage(content=prompt)]
        response_dict = await self._run_sub_llm_request(
            state_ref=state_ref,
            client=client,
            sub_model=sub_model,
            messages=messages,
            batch_id=batch_id,
            request_id=request_id,
            parent_turn=parent_turn,
        )

        content = (
            response_dict.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content


