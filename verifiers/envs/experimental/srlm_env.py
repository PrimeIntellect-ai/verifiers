"""
RLM Environment using the rlm library with verifiers framework model calls.

Drives the RLM REPL loop from the verifiers framework so that root model calls
go through the verifiers Client (providing Response objects with token data for
training). Code execution and sub-LLM calls use the rlm library's environment
infrastructure.

Architecture (matching rlm_env.py's training pattern):
------------------------------------------------------
- Root model calls go through verifiers Client → full Response with tokens
- Code execution uses rlm library's BaseEnv.execute_code()
- Sub-LLM calls from code execution go through rlm's LMHandler socket server
- Root trajectory steps have the main trajectory_id (trained on)
- Sub-LLM calls get separate trajectory_ids (collected, not trained on by default)
- render_completion only uses root model steps
- Rewards are applied at the rollout level (state["reward"])

Required RLM repo changes for full training support:
-----------------------------------------------------
Currently, only root model calls go through the verifiers Client (with token data).
Sub-LLM calls go through rlm's internal LMHandler and lack token-level data.
To train on sub-calls in the future, the rlm repo needs ONE of:

1. Custom BaseLM backend: Accept a BaseLM implementation wrapping verifiers Client,
   so sub-call responses include token_ids and logprobs.
2. Raw API response passthrough: Store raw API responses (with logprobs) in
   RLMChatCompletion so they can be extracted post-hoc.
3. Model call callbacks: Add on_model_call hooks that provide full response objects
   before they're reduced to text.
"""

from __future__ import annotations

import asyncio
import logging
import time
from time import perf_counter
from typing import TYPE_CHECKING, Any

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import (
    Messages,
    Response,
    ResponseMessage,
    State,
    TrajectoryStep,
    Usage,
)
from verifiers.utils.message_utils import (
    concat_messages,
    from_raw_message,
    normalize_messages,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _messages_to_prompt_string(messages: Messages) -> str:
    """Turn normalized messages into a single string for RLM context."""
    parts: list[str] = []
    for msg in messages:
        role = getattr(msg, "role", None) or (
            msg.get("role") if isinstance(msg, dict) else None
        )
        content = getattr(msg, "content", None) or (
            msg.get("content") if isinstance(msg, dict) else None
        )
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = " ".join(text_parts)
        if isinstance(content, str):
            parts.append(f"{role}: {content}" if role else content)
    return "\n\n".join(parts)


def _extract_text_content(content: Any) -> str:
    """Extract plain text from message content (str or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content) if content else ""


def _usage_from_summary(summary: Any) -> Usage | None:
    """Build verifiers Usage from RLM usage_summary if present."""
    if summary is None:
        return None
    inp = getattr(summary, "total_input_tokens", None) or getattr(
        summary, "input_tokens", None
    )
    out = getattr(summary, "total_output_tokens", None) or getattr(
        summary, "output_tokens", None
    )
    if inp is None and out is None:
        return None
    total = (int(inp or 0)) + (int(out or 0))
    return Usage(
        prompt_tokens=int(inp or 0),
        reasoning_tokens=0,
        completion_tokens=int(out or 0),
        total_tokens=total,
    )


def _inject_final_into_env(rlm_env: Any) -> None:
    """Inject a ``FINAL`` function into the RLM REPL environment.

    The rlm library's LocalREPL provides ``FINAL_VAR`` (variable lookup) but
    no ``FINAL`` (direct value).  Models may use either form inside ``repl``
    code blocks *or* in plain response text.  ``find_final_answer`` already
    handles both patterns in text; this function ensures they also work
    inside executed code.

    ``FINAL(value)``  → ``str(value)``, sets the final answer immediately.
    ``FINAL_VAR(name)`` → looks up a variable, same as before.
    """
    env_globals = getattr(rlm_env, "globals", None)
    if env_globals is None:
        return  # not a LocalREPL-like env, skip

    def _final(value: Any) -> str:
        answer = str(value)
        rlm_env._last_final_answer = answer
        return answer

    env_globals["FINAL"] = _final

    # Also patch _restore_scaffold so FINAL survives across code executions.
    # LocalREPL._restore_scaffold resets reserved names after each exec();
    # FINAL is not in RESERVED_TOOL_NAMES so it *could* be overwritten by
    # user code (e.g. ``FINAL = "something"``).  We wrap the original to
    # re-inject it.
    original_restore = getattr(rlm_env, "_restore_scaffold", None)
    if original_restore is not None:

        def _patched_restore() -> None:
            original_restore()
            env_globals["FINAL"] = _final

        rlm_env._restore_scaffold = _patched_restore


def _extract_tokens_from_response(response: Response | Any) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a Response."""
    if not response:
        return 0, 0
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


# =============================================================================
# Metrics
# =============================================================================


def _ensure_srlm_metric_state(state: State) -> None:
    """Initialize all metric keys in state (matching rlm_env.py's metric set)."""
    # Sub-LLM metrics
    state.setdefault("sub_llm_call_count", 0)
    state.setdefault("sub_llm_total_turns", 0)
    state.setdefault("sub_llm_prompt_tokens", 0)
    state.setdefault("sub_llm_completion_tokens", 0)
    state.setdefault("sub_llm_total_tool_calls", 0)
    state.setdefault("sub_llm_batch_count", 0)
    state.setdefault("sub_llm_max_batch_size", 0)
    state.setdefault("sub_llm_mean_batch_size", 0.0)

    # Root model metrics
    state.setdefault("main_rlm_turns", 0)
    state.setdefault("main_rlm_prompt_tokens", 0)
    state.setdefault("main_rlm_completion_tokens", 0)

    # REPL metrics
    state.setdefault("repl_total_time_seconds", 0.0)
    state.setdefault("repl_call_count", 0)
    state.setdefault("repl_mean_time_seconds", 0.0)

    # Code execution timeout metrics
    state.setdefault("repl_timeout_count", 0)


def _update_repl_metrics(state: State, execution_seconds: float) -> None:
    """Update REPL execution timing metrics."""
    _ensure_srlm_metric_state(state)
    state["repl_total_time_seconds"] += execution_seconds
    state["repl_call_count"] += 1
    if state["repl_call_count"] > 0:
        state["repl_mean_time_seconds"] = (
            state["repl_total_time_seconds"] / state["repl_call_count"]
        )


def _update_metrics_from_root_response(state: State, response: Response) -> None:
    """Update root model metrics from a Response."""
    _ensure_srlm_metric_state(state)
    state["main_rlm_turns"] += 1
    prompt_tokens, completion_tokens = _extract_tokens_from_response(response)
    state["main_rlm_prompt_tokens"] += prompt_tokens
    state["main_rlm_completion_tokens"] += completion_tokens


def _update_metrics_from_sub_call(
    state: State, sub_call: Any, parent_turn: int
) -> None:
    """Update sub-LLM metrics from an RLMChatCompletion."""
    _ensure_srlm_metric_state(state)
    state["sub_llm_call_count"] += 1

    usage = getattr(sub_call, "usage_summary", None)
    if usage:
        state["sub_llm_prompt_tokens"] += int(
            getattr(usage, "total_input_tokens", 0) or 0
        )
        state["sub_llm_completion_tokens"] += int(
            getattr(usage, "total_output_tokens", 0) or 0
        )

    # Track sub-call metadata from RLM trajectory if available
    metadata = getattr(sub_call, "metadata", None)
    if metadata and isinstance(metadata, dict):
        iterations = metadata.get("iterations", [])
        state["sub_llm_total_turns"] += len(iterations)


class SRLMMonitorRubric(vf.Rubric):
    """Monitor rubric for SRLMEnv metrics (matching rlm_env.py's RLMMonitorRubric)."""

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
        "repl_timeout_count",
    ]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        for metric_name in self._SIMPLE_METRICS:
            metric_fn = self._make_state_metric(metric_name)
            setattr(self, metric_name, metric_fn)
            self.add_metric(metric_fn)

    def _make_state_metric(self, key: str):
        async def metric(state: State):
            value = state.get(key, 0)
            return 0 if value is None else value

        metric.__name__ = key
        return metric


# =============================================================================
# Environment
# =============================================================================


class SRLMEnv(MultiTurnEnv):
    """
    RLM environment using the rlm library with verifiers Client for model calls.

    Drives the RLM REPL loop from the framework: each root model call goes through
    the verifiers Client (getting Response objects with token data for training).
    Code execution and sub-LLM calls use the rlm library's infrastructure.

    Training trajectory follows rlm_env.py's pattern:
    - Root model steps: trajectory_id = state["trajectory_id"] (trained on)
    - Sub-LLM steps: separate trajectory_ids (collected, not trained on by default)
    """

    def __init__(
        self,
        # Sub-LLM backend for calls from code (root model uses verifiers Client).
        # By default, derives base_url/api_key/model from the root model's
        # verifiers Client so sub-calls go to the same endpoint.
        backend: str = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        sub_model: str | None = None,
        # RLM environment (REPL)
        environment: str = "local",
        environment_kwargs: dict[str, Any] | None = None,
        # Iteration limits
        max_iterations: int = 30,
        max_depth: int = 1,
        max_timeout: float | None = None,
        # Sub-call config
        other_backends: list[str] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        # Training config
        include_sub_llm_in_trajectory: bool = False,
        # Prompt config
        root_prompt_key: str | None = None,
        context_key: str | None = None,
        custom_system_prompt: str | None = None,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        # Execution config
        code_execution_timeout: int | None = 120,
        abort_on_code_timeout: bool = False,
        max_output_length: int = 20000,
        # Misc
        verbose: bool = False,
        **kwargs: Any,
    ):
        super().__init__(max_turns=max_iterations, **kwargs)
        self._backend = backend
        self._backend_kwargs = backend_kwargs
        self._sub_model = sub_model
        self._environment_type = environment
        self._environment_kwargs = environment_kwargs or {}
        self._max_depth = max_depth
        self._max_timeout = max_timeout
        self._other_backends = other_backends
        self._other_backend_kwargs = other_backend_kwargs
        self._include_sub_llm_in_trajectory = include_sub_llm_in_trajectory
        self._root_prompt_key = root_prompt_key
        self._context_key = context_key
        self._custom_system_prompt = custom_system_prompt
        self._custom_tools = custom_tools
        self._custom_sub_tools = custom_sub_tools
        self._code_execution_timeout = code_execution_timeout
        self._abort_on_code_timeout = abort_on_code_timeout
        self._max_output_length = max_output_length
        self._verbose = verbose

        self.add_rubric(SRLMMonitorRubric())

    # =========================================================================
    # Sub-LLM Backend Resolution
    # =========================================================================

    def _derive_backend_kwargs(self, state: State) -> dict[str, Any]:
        """Derive sub-LLM backend_kwargs from the root model's verifiers Client.

        Extracts base_url, api_key, and model name from the underlying client
        so sub-LLM calls from code (llm_query, llm_query_batched) go to the
        same API endpoint as the root model by default.

        The model name is passed through as-is (e.g. 'openai/gpt-5-mini')
        since the Prime Intellect API uses the provider prefix for routing.
        """
        model_name = self._sub_model or state.get("model", "unknown")
        base_url: str | None = None
        api_key: str | None = None

        vf_client = state.get("client")
        if vf_client is not None:
            # Try to extract from the underlying client (AsyncOpenAI, etc.)
            underlying = getattr(vf_client, "client", None)
            if underlying is not None:
                raw_url = getattr(underlying, "base_url", None)
                if raw_url is not None:
                    base_url = str(raw_url).rstrip("/")
                api_key = getattr(underlying, "api_key", None)

        kwargs: dict[str, Any] = {"model_name": model_name}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    # =========================================================================
    # State Setup
    # =========================================================================

    async def setup_state(self, state: State) -> State:
        """Create RLM environment and LMHandler, build RLM system prompt."""
        try:
            from rlm.core.rlm import (
                LMHandler,
                RLM_SYSTEM_PROMPT,
                build_rlm_system_prompt,
                build_user_prompt,
                get_client,
                get_environment,
            )
            from rlm.core.types import QueryMetadata
        except ImportError as e:
            raise ImportError(
                "SRLMEnv requires the rlms package. Install with: pip install rlms"
            ) from e

        # Resolve context: prefer structured context from info[context_key],
        # fall back to stringifying the chat prompt. Preserving the original
        # type (str/list/dict) lets QueryMetadata report accurate metadata
        # to the model (e.g. "list with N chunks" instead of always "str").
        context_payload: str | list | dict
        if self._context_key:
            info = state.get("info", {})
            if isinstance(info, dict) and self._context_key in info:
                context_payload = info[self._context_key]
            else:
                original_prompt = normalize_messages(
                    state["prompt"], field_name="state.prompt"
                )
                context_payload = _messages_to_prompt_string(original_prompt)
        else:
            original_prompt = normalize_messages(
                state["prompt"], field_name="state.prompt"
            )
            context_payload = _messages_to_prompt_string(original_prompt)

        # Get optional root_prompt from info
        root_prompt = None
        if self._root_prompt_key:
            info = state.get("info", {})
            if isinstance(info, dict):
                root_prompt = info.get(self._root_prompt_key)

        # Resolve sub-LLM backend kwargs. When not explicitly provided,
        # derive from the root model's verifiers Client so sub-calls go
        # to the same API endpoint by default.
        backend_kwargs = self._backend_kwargs
        if backend_kwargs is None:
            backend_kwargs = self._derive_backend_kwargs(state)

        # Create sub-call backend client and LMHandler (for sub-LLM calls from code)
        sub_client = get_client(self._backend, backend_kwargs)
        other_backend_client = None
        if self._other_backends and self._other_backend_kwargs:
            other_backend_client = get_client(
                self._other_backends[0], self._other_backend_kwargs[0]
            )
        lm_handler = LMHandler(sub_client, other_backend_client=other_backend_client)

        # Register additional backends for sub-calls
        if self._other_backends and self._other_backend_kwargs:
            for backend, bkwargs in zip(
                self._other_backends, self._other_backend_kwargs, strict=True
            ):
                other_client = get_client(backend, bkwargs)
                lm_handler.register_client(other_client.model_name, other_client)

        lm_handler.start()

        # Build env creation kwargs (stored for recreation after timeout)
        env_kwargs: dict[str, Any] = dict(self._environment_kwargs)
        env_kwargs["lm_handler_address"] = lm_handler.address
        env_kwargs["context_payload"] = context_payload
        env_kwargs["depth"] = 1  # Environment depth is 1 (root RLM is depth 0)
        if self._custom_tools is not None:
            env_kwargs["custom_tools"] = self._custom_tools
        if self._custom_sub_tools is not None:
            env_kwargs["custom_sub_tools"] = self._custom_sub_tools
        rlm_env = get_environment(self._environment_type, env_kwargs)
        _inject_final_into_env(rlm_env)

        # Build RLM system prompt — pass context_payload directly so
        # QueryMetadata sees the actual type (str/list/dict).
        system_prompt_text = self._custom_system_prompt or RLM_SYSTEM_PROMPT
        query_metadata = QueryMetadata(context_payload)
        system_messages = build_rlm_system_prompt(
            system_prompt=system_prompt_text,
            query_metadata=query_metadata,
            custom_tools=self._custom_tools,
        )

        # Build first iteration's user prompt (transient — not part of message_history)
        user_prompt_0 = build_user_prompt(root_prompt, iteration=0)
        # TODO: Pass context_count and history_count to build_user_prompt when
        # persistent environment support is added. Currently defaults to
        # context_count=1, history_count=0.

        # Override state["prompt"] with RLM prompt structure (system + user_prompt_0)
        rlm_initial_prompt = system_messages + [user_prompt_0]
        state["prompt"] = [from_raw_message(m) for m in rlm_initial_prompt]

        # Maintain RLM-style message history (list[dict]) for building prompts.
        # This mirrors message_history in RLM.completion(): it contains only
        # the system prompt + format_iteration outputs. The per-iteration
        # user_prompt is appended transiently in get_prompt_messages, NOT
        # persisted here (matching RLM.completion()'s behavior where
        # current_prompt = message_history + [build_user_prompt(...)]).
        state["rlm_message_history"] = list(system_messages)

        # Store RLM state
        state["rlm_env"] = rlm_env
        state["rlm_lm_handler"] = lm_handler
        state["rlm_env_kwargs"] = env_kwargs  # for recreation after timeout
        state["rlm_iteration"] = 0
        state["rlm_root_prompt"] = root_prompt
        state["rlm_sub_calls"] = []  # All sub-call RLMChatCompletion objects
        state["_rlm_start_time"] = perf_counter()

        # Initialize metrics
        _ensure_srlm_metric_state(state)

        return state

    # =========================================================================
    # Prompt Construction
    # =========================================================================

    async def get_prompt_messages(self, state: State) -> Messages:
        """Build prompt from RLM message history + transient user prompt.

        Mirrors RLM.completion()'s pattern:
            current_prompt = message_history + [build_user_prompt(root_prompt, i)]
        where message_history contains system + past format_iteration outputs,
        and the user_prompt is ephemeral (not persisted in message_history).
        """
        from rlm.core.rlm import build_user_prompt

        if len(state["trajectory"]) == 0:
            return normalize_messages(state["prompt"], field_name="state.prompt")

        # message_history + transient user prompt for this iteration
        # TODO: Pass context_count and history_count when persistent env support
        # is added.
        user_prompt = build_user_prompt(
            state.get("rlm_root_prompt"),
            state["rlm_iteration"],
        )
        prompt_dicts = state["rlm_message_history"] + [user_prompt]
        return [from_raw_message(m) for m in prompt_dicts]

    # =========================================================================
    # Model Response + Code Execution
    # =========================================================================

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ):
        """Add trajectory step, then execute code blocks from the response.

        Code execution happens here (not in env_response) because in the text-based
        RLM pattern, code blocks are embedded in the model's text response rather than
        as tool calls. Executing here ensures code runs right after the model responds,
        so stop conditions (answer_ready) can fire before the next turn.
        """
        # Standard trajectory step (root model, with token data)
        await super().add_model_response(state, prompt_messages, response)

        # Update root model metrics
        _update_metrics_from_root_response(state, response)

        # Track best partial answer for default-answer fallback
        response_text = _extract_text_content(response.message.content)
        if response_text and response_text.strip():
            state["_rlm_best_partial_answer"] = response_text

        # Execute code blocks via RLM environment
        await self._execute_code_blocks(state, response_text)

    async def _execute_code_block_with_timeout(
        self, state: State, rlm_env: Any, code_str: str
    ) -> Any:
        """Execute a single code block with timeout protection.

        Returns the REPLResult on success, or a synthetic error REPLResult on
        timeout. Raises on timeout if abort_on_code_timeout is True.

        On timeout, the old LocalREPL is abandoned (the stuck thread holds its
        lock, making it unusable) and a fresh one is created. REPL state
        (variables, imports) is lost, matching rlm_env.py's sandbox-restart
        behavior.
        """
        from rlm.core.types import REPLResult

        loop = asyncio.get_event_loop()
        exec_start = perf_counter()

        timeout = self._code_execution_timeout
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, rlm_env.execute_code, code_str),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, TimeoutError):
            exec_time = perf_counter() - exec_start
            logger.warning(
                "Code execution timed out after %ss (limit: %ss)",
                f"{exec_time:.1f}",
                self._code_execution_timeout,
            )
            state["repl_timeout_count"] = state.get("repl_timeout_count", 0) + 1
            _update_repl_metrics(state, exec_time)

            if self._abort_on_code_timeout:
                raise vf.ToolCallError(
                    f"Code execution timed out after "
                    f"{self._code_execution_timeout} seconds."
                ) from None

            # The stuck thread still holds the LocalREPL lock and may be
            # mutating its namespace. Abandon the old instance and create
            # a fresh one so subsequent code blocks run safely.
            await self._recreate_rlm_env(state)

            return REPLResult(
                stdout="",
                stderr=(
                    f"Code execution timed out after "
                    f"{self._code_execution_timeout} seconds. "
                    "The REPL environment was restarted and all state "
                    "(variables, imports) has been reset. "
                    "Your code may be too slow — consider a more efficient "
                    "algorithm or breaking the computation into smaller steps."
                ),
                locals={},
                execution_time=exec_time,
            )

        exec_time = perf_counter() - exec_start
        _update_repl_metrics(state, exec_time)
        return result

    async def _recreate_rlm_env(self, state: State) -> None:
        """Replace the current RLM environment with a fresh one.

        Called after a code execution timeout: the old LocalREPL's thread is
        stuck and holds its internal lock, making the instance unusable.
        We create a new environment using the stored creation kwargs.
        """
        from rlm.core.rlm import get_environment

        old_env = state.get("rlm_env")
        # Don't call cleanup — the stuck thread may hold the lock.
        # The old instance will be garbage collected (thread eventually dies
        # or process ends).
        if old_env is not None:
            logger.debug("Abandoning stuck RLM environment after timeout")

        env_kwargs = state["rlm_env_kwargs"]
        # Ensure the LMHandler address is current
        lm_handler = state.get("rlm_lm_handler")
        if lm_handler:
            env_kwargs["lm_handler_address"] = lm_handler.address
        new_env = get_environment(self._environment_type, env_kwargs)
        _inject_final_into_env(new_env)
        state["rlm_env"] = new_env

    async def _execute_code_blocks(self, state: State, response_text: str) -> None:
        """Parse and execute ALL code blocks, then check for final answer.

        Matches RLM.completion()'s _completion_turn: all code blocks are
        executed before checking any of their results for a final answer.
        """
        from rlm.core.rlm import (
            find_code_blocks,
            find_final_answer,
            format_iteration,
        )
        from rlm.core.types import CodeBlock, RLMIteration

        rlm_env = state["rlm_env"]
        code_block_strs = find_code_blocks(response_text)
        code_blocks: list[CodeBlock] = []

        # Execute ALL code blocks (matching RLM._completion_turn which does
        # not break early). Side effects of later blocks still happen even
        # if an earlier block sets FINAL_VAR.
        for code_str in code_block_strs:
            result = await self._execute_code_block_with_timeout(
                state, state["rlm_env"], code_str
            )
            code_blocks.append(CodeBlock(code=code_str, result=result))

            # Collect sub-LLM calls from code execution
            if result.rlm_calls:
                for sub_call in result.rlm_calls:
                    state["rlm_sub_calls"].append(sub_call)
                    _update_metrics_from_sub_call(
                        state, sub_call, parent_turn=state["rlm_iteration"]
                    )
                    if self._include_sub_llm_in_trajectory:
                        await self._add_sub_call_steps(
                            state, sub_call, parent_turn=state["rlm_iteration"]
                        )

        # Check for final answer AFTER all blocks (matching RLM.completion())
        # Prefer FINAL_VAR result from REPL execution (first block wins).
        final_answer = None
        for block in code_blocks:
            if getattr(block.result, "final_answer", None):
                final_answer = block.result.final_answer
                break
        # Fallback: check for final answer in response text
        if final_answer is None:
            final_answer = find_final_answer(response_text, environment=rlm_env)
        if final_answer is not None:
            state["final_answer"] = final_answer

        # Build iteration and format for message history.
        # format_iteration returns [assistant_msg, user_msg_1, ...].
        # All are appended to rlm_message_history (the RLM-style conversation
        # used for prompt building). The per-iteration user_prompt is NOT stored
        # here — it's added transiently in get_prompt_messages, matching
        # RLM.completion()'s pattern.
        iteration = RLMIteration(
            prompt="",  # not used by format_iteration
            response=response_text,
            code_blocks=code_blocks,
        )
        formatted = format_iteration(
            iteration, max_character_length=self._max_output_length
        )
        state["rlm_message_history"].extend(formatted)

        if "final_answer" in state:
            # Signal to MultiTurnEnv that we're done. The final env response
            # includes the execution results so render_completion can include them.
            execution_messages = [from_raw_message(m) for m in formatted[1:]]
            state["final_env_response"] = execution_messages
        else:
            # Advance iteration counter for the next turn's user prompt
            state["rlm_iteration"] += 1

    async def _add_sub_call_steps(
        self,
        state: State,
        sub_call: Any,  # RLMChatCompletion
        parent_turn: int,
    ) -> None:
        """Add sub-LLM call as a trajectory step for collection (not training).

        Sub-calls lack token-level data (prompt_ids, completion_ids, logprobs)
        because they go through rlm's internal LMHandler, not the verifiers Client.
        They are stored with a separate trajectory_id so they are excluded from
        training by default.
        """
        sub_trajectory_id = f"sub_{parent_turn}_{id(sub_call)}"

        usage = _usage_from_summary(getattr(sub_call, "usage_summary", None))
        sub_response = Response(
            id=f"rlm-sub-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=getattr(sub_call, "root_model", "unknown"),
            usage=usage,
            message=ResponseMessage(
                content=getattr(sub_call, "response", ""),
                reasoning_content=None,
                tool_calls=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
            ),
        )

        step = TrajectoryStep(
            prompt=[],
            completion=[],
            response=sub_response,
            tokens=None,  # No token data for sub-calls
            reward=None,
            advantage=None,
            is_truncated=False,
            trajectory_id=sub_trajectory_id,
            extras={
                "is_sub_llm_call": True,
                "parent_turn": parent_turn,
                "sub_call_model": getattr(sub_call, "root_model", "unknown"),
                "execution_time": getattr(sub_call, "execution_time", None),
                "sub_call_metadata": getattr(sub_call, "metadata", None),
            },
        )
        await self.add_trajectory_step(state, step)

    # =========================================================================
    # env_response (not used for code execution, but required by MultiTurnEnv)
    # =========================================================================

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Messages | str:
        """No-op: code execution happens in add_model_response instead.

        MultiTurnEnv.get_prompt_messages calls this, but we override
        get_prompt_messages to use rlm_message_history directly, so this
        is never called in normal flow.
        """
        return []

    # =========================================================================
    # Stop Conditions
    # =========================================================================

    @vf.stop
    async def answer_ready(self, state: State) -> bool:
        """Stop when a final answer has been found (via REPL or response text)."""
        return "final_answer" in state

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        """Count only root model trajectory steps, not sub-LLM steps."""
        if self.max_turns <= 0:
            return False
        main_id = state.get("trajectory_id")
        count = sum(
            1 for s in state.get("trajectory", [])
            if s.get("trajectory_id") == main_id
        )
        return count >= self.max_turns

    @vf.stop
    async def timeout_reached(self, state: State) -> bool:
        """Stop when overall wallclock time exceeds max_timeout.

        Mirrors RLM.completion()'s _check_timeout which fires before each
        iteration. Here it fires between turns (after add_model_response
        completes, including code execution).
        """
        if self._max_timeout is None:
            return False
        start_time = state.get("_rlm_start_time")
        if start_time is None:
            return False
        elapsed = perf_counter() - start_time
        if elapsed > self._max_timeout:
            logger.warning(
                "Rollout timeout: %.1fs elapsed (limit: %.1fs)",
                elapsed,
                self._max_timeout,
            )
            return True
        return False

    # =========================================================================
    # Default Answer Fallback
    # =========================================================================

    async def _ensure_final_answer(self, state: State) -> None:
        """Make one final model call if the loop ended without a final answer.

        Mirrors RLM.completion()'s _default_answer: when max_iterations is
        exhausted, the model is asked once more to provide a final answer
        based on the accumulated context.
        """
        if "final_answer" in state:
            return

        # Build the default-answer prompt: message_history + nudge.
        # This matches RLM.completion()'s _default_answer which appends
        # an assistant message asking for the final answer.
        default_answer_msg = {
            "role": "assistant",
            "content": (
                "Please provide a final answer to the user's question "
                "based on the information provided."
            ),
        }
        prompt_dicts = state["rlm_message_history"] + [default_answer_msg]
        prompt_messages = [from_raw_message(m) for m in prompt_dicts]

        try:
            response = await self.get_model_response(state, prompt_messages)
            # Add as a regular trajectory step (with token data for training)
            await super().add_model_response(state, prompt_messages, response)
            _update_metrics_from_root_response(state, response)

            response_text = _extract_text_content(response.message.content)
            state["final_answer"] = response_text
            logger.debug(
                "Default answer from model: %s...", response_text[:100]
            )
        except Exception:
            logger.debug("Default answer model call failed", exc_info=True)
            # Last resort: use the best partial answer we saw during the rollout
            best = state.get("_rlm_best_partial_answer")
            if best:
                state["final_answer"] = best
                logger.debug(
                    "Using best partial answer as fallback: %s...", best[:100]
                )

    # =========================================================================
    # Render Completion
    # =========================================================================

    async def render_completion(self, state: State):
        """Render completion from root model steps only, ignoring sub-LLM steps.

        Matches rlm_env.py's pattern: only the main trajectory_id steps are
        included in the completion used for reward computation and training.
        """
        # Ensure we have a final answer before rendering
        await self._ensure_final_answer(state)

        if len(state["trajectory"]) == 0:
            state["completion"] = []
            return

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
                [
                    full_conversation,
                    normalize_messages(
                        state["final_env_response"], field_name="final_env_response"
                    ),
                ]
            )
        state["completion"] = full_conversation[len(state["prompt"]):]

    # =========================================================================
    # Model Response Override (for sub-LLM trajectory interleaving)
    # =========================================================================

    async def get_model_response(
        self, state: State, prompt: Messages, **kwargs: Any
    ) -> Response:
        """Ensure get_prompt_ids sees the last main trajectory step, not a sub-LLM step.

        When include_sub_llm_in_trajectory is True, sub-LLM steps may be appended
        to the trajectory after the last root step. Temporarily move them aside
        so token-level prompt building works correctly.
        """
        if not self._include_sub_llm_in_trajectory:
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

        trailing = trajectory[last_main_idx + 1:]
        del trajectory[last_main_idx + 1:]
        try:
            result = await super().get_model_response(state, prompt, **kwargs)
        finally:
            trajectory.extend(trailing)
        return result

    # =========================================================================
    # Cleanup
    # =========================================================================

    @vf.cleanup
    async def cleanup_rlm(self, state: State):
        """Cleanup RLM environment and LM handler."""
        # Ensure final answer is set before cleanup
        await self._ensure_final_answer(state)

        lm_handler = state.pop("rlm_lm_handler", None)
        if lm_handler:
            try:
                lm_handler.stop()
            except Exception:
                logger.debug("Error stopping LMHandler", exc_info=True)
        rlm_env = state.pop("rlm_env", None)
        if rlm_env and hasattr(rlm_env, "cleanup"):
            try:
                rlm_env.cleanup()
            except Exception:
                logger.debug("Error cleaning up RLM environment", exc_info=True)


def load_environment(
    dataset: vf.Dataset | vf.DatasetBuilder | None = None,
    rubric: vf.Rubric | None = None,
    backend: str = "openai",
    backend_kwargs: dict[str, Any] | None = None,
    sub_model: str | None = None,
    environment: str = "local",
    environment_kwargs: dict[str, Any] | None = None,
    max_iterations: int = 30,
    max_depth: int = 1,
    max_timeout: float | None = None,
    other_backends: list[str] | None = None,
    other_backend_kwargs: list[dict[str, Any]] | None = None,
    include_sub_llm_in_trajectory: bool = False,
    root_prompt_key: str | None = None,
    context_key: str | None = None,
    custom_system_prompt: str | None = None,
    custom_tools: dict[str, Any] | None = None,
    custom_sub_tools: dict[str, Any] | None = None,
    code_execution_timeout: int | None = 120,
    abort_on_code_timeout: bool = False,
    max_output_length: int = 20000,
    verbose: bool = False,
    **kwargs: Any,
) -> vf.Environment:
    """Load the RLM training environment using the rlms library.

    This environment drives the RLM REPL loop from the verifiers framework,
    using the verifiers Client for root model calls (enabling training) and
    the rlm library for code execution and sub-LLM calls.

    Args:
        backend: Backend type for sub-LLM calls ("openai", "anthropic", etc.).
        backend_kwargs: Explicit backend configuration. When None (default),
            base_url and api_key are derived from the root model's verifiers
            Client so sub-calls go to the same API endpoint automatically.
        sub_model: Model name for sub-LLM calls. Defaults to the root model.
        environment: RLM REPL environment type ("local", "docker", etc.).
        environment_kwargs: Environment configuration.
        max_iterations: Maximum REPL iterations (root model calls) per rollout.
        max_depth: Maximum recursion depth for sub-calls.
        max_timeout: Maximum wallclock time in seconds for the entire rollout.
            None disables the timeout (default).
        other_backends: Additional backends available for sub-calls.
        other_backend_kwargs: Configuration for additional backends.
        include_sub_llm_in_trajectory: If True, sub-LLM calls are added as
            trajectory steps (with separate trajectory_ids, not trained on).
        root_prompt_key: Key in info dict for an optional root_prompt (small
            prompt the root model sees directly, e.g., the question).
        context_key: Key in info dict for structured context data. When set,
            the context is passed to the RLM environment and QueryMetadata
            in its original form (str/list/dict) instead of stringifying
            the chat prompt. This lets the model see accurate metadata
            like "list with N chunks" instead of always "str".
        custom_system_prompt: Override the default RLM system prompt.
        custom_tools: Custom functions available in the REPL environment.
        custom_sub_tools: Custom tools for sub-agents. If None, inherits
            from custom_tools. Pass {} to disable.
        code_execution_timeout: Timeout in seconds for each code block
            execution. On timeout, the REPL is restarted (state lost) and
            the model receives an error message. None disables the timeout.
        abort_on_code_timeout: If True, raise an error on code execution
            timeout instead of restarting and continuing.
        max_output_length: Maximum character length for code execution
            output shown to the model. Longer output is truncated.
        verbose: Enable verbose logging.
    """
    return SRLMEnv(
        dataset=dataset,
        rubric=rubric,
        backend=backend,
        backend_kwargs=backend_kwargs,
        sub_model=sub_model,
        environment=environment,
        environment_kwargs=environment_kwargs or {},
        max_iterations=max_iterations,
        max_depth=max_depth,
        max_timeout=max_timeout,
        other_backends=other_backends,
        other_backend_kwargs=other_backend_kwargs,
        include_sub_llm_in_trajectory=include_sub_llm_in_trajectory,
        root_prompt_key=root_prompt_key,
        context_key=context_key,
        custom_system_prompt=custom_system_prompt,
        custom_tools=custom_tools,
        custom_sub_tools=custom_sub_tools,
        code_execution_timeout=code_execution_timeout,
        abort_on_code_timeout=abort_on_code_timeout,
        max_output_length=max_output_length,
        verbose=verbose,
        **kwargs,
    )
