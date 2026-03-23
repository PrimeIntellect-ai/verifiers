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
import uuid
from typing import Any, Callable, Literal

import verifiers as vf
from verifiers.envs.experimental.rlm_env import (
    RLMEnv,
    SubLLMResult,
    SubLLMTurn,
    _clone_messages,
    _extract_tokens_from_response,
)
from verifiers.types import (
    Messages,
    State,
    UserMessage,
)

logger = logging.getLogger(__name__)

# Both llm and llm_batch are reserved (cannot be overridden by user tools).
_SRLM_FIXED_REPL_TOOL_NAMES = frozenset({"llm", "llm_batch"})


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


def load_environment(
    dataset: vf.Dataset | vf.DatasetBuilder | None = None,
    rubric: vf.Rubric | None = None,
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
) -> vf.Environment:
    """Load the Simplified RLM training environment.

    This environment provides a sandbox-based Python/Bash REPL where the model
    can execute code, make single-shot sub-LLM calls via ``llm()`` and
    ``llm_batch()``, and interact with filesystem data.

    Sub-LLM calls are single-shot: one API request per call, no tool-calling
    loop.  This is the key difference from :class:`RLMEnv`, which supports
    multi-turn tool-calling loops for sub-LLMs.

    Args:
        dataset: Dataset to use for the environment.
        rubric: Rubric for reward computation.
        tools: List of tools available in the root REPL (shared).
        root_tools: Additional tools available only in the root REPL.
        sub_model: Model for sub-LLM calls (defaults to root model).
        sub_prompt_verbosity: Sub-LLM system prompt verbosity.
        root_prompt_verbosity: Root-LLM system prompt verbosity.
        max_iterations: Maximum REPL iterations per rollout.
        max_output_length: Maximum code execution output length.
        max_sub_llm_parallelism: Max concurrent sub-LLM calls in llm_batch.
        sub_llm_stagger_ms: Per-call stagger delay (ms) in llm_batch.
        sub_llm_stagger_jitter_ms: Random jitter (ms) for stagger.
        context_key: Key in info for legacy context data.
        context_dir_key: Key in info for directory path.
        system_prompt: Custom system prompt override.
        repl_language: REPL language ("bash" or "python").
        interception_host: Interception server hostname.
        interception_port: Interception server port.
        interception_url: Optional base URL for interception.
        pip_install_packages: Space-separated packages for sandbox.
        include_sub_llm_in_trajectory: Include sub-LLM calls as trajectory
            steps for training.
        context_warning_threshold: Fraction of max_seq_len to warn at.
        max_startup_wait_seconds: Max worker startup wait.
        code_execution_timeout: Code execution timeout (seconds).
        abort_on_code_timeout: Abort rollout on code timeout.
        retain_filesystem_after_rollout: Keep filesystem after rollout.
        filesystem_copy_max_bytes: Max bytes for context directory copy.
        sandbox_docker_image: Docker image for sandbox.
        sandbox_start_command: Sandbox start command.
        sandbox_cpu_cores: Sandbox CPU cores.
        sandbox_memory_gb: Sandbox memory (GB).
        sandbox_disk_size_gb: Sandbox disk (GB).
        sandbox_gpu_count: Sandbox GPU count.
        sandbox_timeout_minutes: Sandbox timeout (minutes).
        sandbox_environment_vars: Extra sandbox environment vars.
        sandbox_team_id: Sandbox team ID.
        sandbox_advanced_configs: Sandbox advanced configs.
        sandbox_labels: Sandbox labels.
        sandbox_client_max_workers: Sandbox client pool size.
        sandbox_client_max_connections: Sandbox client max connections.
        sandbox_client_max_keepalive_connections: Sandbox client keepalive.
        **kwargs: Additional arguments passed to StatefulToolEnv.
    """
    return SRLMEnv(
        dataset=dataset,
        rubric=rubric,
        tools=tools,
        root_tools=root_tools,
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
