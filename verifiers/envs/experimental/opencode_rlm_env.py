"""
OpenCode RLM Environment.

Extends OpenCodeEnv with the RLM plugin (https://github.com/snimu/oc),
adding concurrent sub-LLM handling via ``llm-subcall`` / ``llm_batch``.

Sub-agent calls (identified by model name) are handled concurrently with
semaphore-based parallelism control.  Main-agent calls go through the normal
sequential rollout loop.

The RLM plugin sends ``RLM_SUB_MODEL_ID`` (default ``"sub"``) as the model
field for ``llm-subcall`` requests.  Subagent sessions are disabled
(``RLM_DISABLE_SUBAGENT_SESSIONS=true``) so that all sub-LLM traffic goes
through the identifiable ``llm-subcall`` path.
"""

import asyncio
import json
import logging
import time
from typing import Any

import verifiers as vf
from verifiers.envs.experimental.opencode_env import OpenCodeEnv
from verifiers.types import (
    Messages,
    MessageType,
    Response,
    SamplingArgs,
    State,
    Tool,
)
from verifiers.utils.interception_utils import deliver_response, synthesize_stream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monitor rubric
# ---------------------------------------------------------------------------


class OpenCodeRLMMonitorRubric(vf.Rubric):
    """Tracks main-agent and sub-LLM metrics separately."""

    _METRICS = [
        "main_turns",
        "main_prompt_tokens",
        "main_completion_tokens",
        "sub_llm_turns",
        "sub_llm_prompt_tokens",
        "sub_llm_completion_tokens",
    ]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        for name in self._METRICS:
            fn = self._make_metric(name)
            setattr(self, name, fn)
            self.add_metric(fn)

    def _make_metric(self, key: str):
        async def metric(state: State) -> float:
            return float(state.get(key, 0))

        metric.__name__ = key
        return metric


# ---------------------------------------------------------------------------
# Run command template
# ---------------------------------------------------------------------------

# Extends the default OpenCodeEnv template with bun + plugin installation.
RLM_RUN_COMMAND_TEMPLATE = """\
set -eo pipefail

apt-get update && apt-get install -y curl git unzip jq

# Install bun (TypeScript runtime required by the RLM plugin)
curl -fsSL https://bun.sh/install | bash
export PATH="$HOME/.bun/bin:$PATH"

# Install opencode
for install_attempt in 1 2 3; do
    if {install_command}; then
        break
    fi
    if [ "$install_attempt" -eq 3 ]; then
        echo "OpenCode installation failed after 3 attempts" >&2
        exit 1
    fi
    echo "OpenCode install attempt $install_attempt/3 failed, retrying in 5s..." >&2
    sleep 5
done
export PATH="$HOME/.opencode/bin:$PATH"

# Install RLM plugin
git clone --branch {plugin_branch} https://github.com/{plugin_repo}.git {plugin_install_path}
cd {plugin_install_path} && bun install

# Write opencode config
mkdir -p ~/.config/opencode

SCHEMA_DOLLAR='$'

cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG

cd {agent_workdir}
cat {prompt_path} | opencode run 2>&1 | tee {logs_path}
"""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class OpenCodeRLMEnv(OpenCodeEnv):
    """
    OpenCodeEnv with the RLM plugin for recursive sub-LLM calls.

    Intercepts all API calls from the main agent and its sub-agents.
    Requests whose intercepted ``model`` field contains
    *sub_model_identifier* are handled concurrently (with a semaphore);
    all other requests go through the normal sequential rollout loop.

    Args:
        plugin_repo: GitHub ``<org>/<repo>`` for the RLM plugin.
        plugin_install_path: Where the plugin is cloned inside the sandbox.
        sub_model_identifier: Substring used to detect sub-LLM requests in
            the intercepted ``model`` field.
        sub_model: Optional separate model name for sub-LLM inference.
            Defaults to the same model as the main agent.
        max_sub_llm_parallelism: Semaphore limit for concurrent sub-LLM
            model calls.
        include_sub_llm_in_trajectory: Whether to append sub-LLM steps to
            the trajectory (useful for training on sub-LLM calls).
    """

    DEFAULT_PLUGIN_REPO = "snimu/oc"
    DEFAULT_PLUGIN_BRANCH = "main"
    DEFAULT_PLUGIN_INSTALL_PATH = "/tmp/opencode-rlm"
    DEFAULT_SUB_MODEL_IDENTIFIER = "sub"

    def __init__(
        self,
        plugin_repo: str = DEFAULT_PLUGIN_REPO,
        plugin_branch: str = DEFAULT_PLUGIN_BRANCH,
        plugin_install_path: str = DEFAULT_PLUGIN_INSTALL_PATH,
        sub_model_identifier: str = DEFAULT_SUB_MODEL_IDENTIFIER,
        sub_model: str | None = None,
        max_sub_llm_parallelism: int = 10,
        sub_llm_max_turns: int = 10,
        sub_timeout_ms: int = 120_000,
        include_sub_llm_in_trajectory: bool = False,
        **kwargs: Any,
    ):
        self.plugin_repo = plugin_repo
        self.plugin_branch = plugin_branch
        self.plugin_install_path = plugin_install_path
        self.sub_model_identifier = sub_model_identifier
        self.sub_model = sub_model
        self.sub_llm_max_turns = sub_llm_max_turns
        self.sub_timeout_ms = sub_timeout_ms
        self.include_sub_llm_in_trajectory = include_sub_llm_in_trajectory
        self._sub_llm_semaphore = asyncio.Semaphore(max_sub_llm_parallelism)

        kwargs.setdefault("run_command_template", RLM_RUN_COMMAND_TEMPLATE)

        super().__init__(**kwargs)
        self.add_rubric(OpenCodeRLMMonitorRubric())

    # ------------------------------------------------------------------
    # Config & run command
    # ------------------------------------------------------------------

    def build_opencode_config(
        self,
        disabled_tools: list[str] | None = None,
        system_prompt_path: str | None = None,
        disable_compaction: bool = True,
        enable_interleaved: bool = True,
    ) -> str:
        """Extend base config with RLM plugin reference."""
        config_str = super().build_opencode_config(
            disabled_tools=disabled_tools,
            system_prompt_path=system_prompt_path,
            disable_compaction=disable_compaction,
            enable_interleaved=enable_interleaved,
        )
        config = json.loads(config_str)
        config["plugin"] = [f"file://{self.plugin_install_path}"]
        return json.dumps(config, indent=2)

    def build_run_command(
        self,
        run_command_template: str,
        agent_workdir: str,
        disabled_tools: list[str] | None = None,
        system_prompt: str | None = None,
        install_command: str = OpenCodeEnv.DEFAULT_INSTALL_COMMAND,
        disable_compaction: bool = True,
        enable_interleaved: bool = True,
    ) -> str:
        config_json = self.build_opencode_config(
            disabled_tools,
            self.remote_system_prompt_path if system_prompt else None,
            disable_compaction=disable_compaction,
            enable_interleaved=enable_interleaved,
        )

        return run_command_template.format(
            config_json=config_json,
            agent_workdir=agent_workdir,
            prompt_path=self.remote_prompt_path,
            logs_path=self.remote_logs_path,
            install_command=install_command,
            plugin_repo=self.plugin_repo,
            plugin_branch=self.plugin_branch,
            plugin_install_path=self.plugin_install_path,
        )

    # ------------------------------------------------------------------
    # Sandbox env vars
    # ------------------------------------------------------------------

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env = await super().build_env_vars(state)
        # Tell the RLM plugin's llm-subcall to send this model identifier,
        # so verifiers can route sub-LLM requests to the concurrent handler.
        env["RLM_SUB_MODEL_ID"] = self.sub_model_identifier
        # Route llm-subcall through the interception proxy instead of
        # calling the real API directly.
        env["RLM_LLM_SUBCALL_VIA_PROXY"] = "true"
        # Use the OC proxy's custom tool-calling loop for subagent calls
        # instead of OpenCode child sessions (which can't be distinguished
        # from the main agent and would serialize in the rollout loop).
        env["RLM_SUBAGENT_VIA_TOOL_LOOP"] = "true"
        env["RLM_SUB_MAX_TURNS"] = str(getattr(self, "sub_llm_max_turns", 10))
        env["RLM_SUB_TIMEOUT"] = str(getattr(self, "sub_timeout_ms", 120000))
        return env

    # ------------------------------------------------------------------
    # State setup
    # ------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state.setdefault("main_turns", 0)
        state.setdefault("main_prompt_tokens", 0)
        state.setdefault("main_completion_tokens", 0)
        state.setdefault("sub_llm_turns", 0)
        state.setdefault("sub_llm_prompt_tokens", 0)
        state.setdefault("sub_llm_completion_tokens", 0)
        return state

    # ------------------------------------------------------------------
    # Sub-LLM detection
    # ------------------------------------------------------------------

    def _is_sub_llm_request(self, intercept: dict[str, Any]) -> bool:
        model = intercept.get("model") or ""
        return self.sub_model_identifier in model

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------

    async def get_prompt_messages(self, state: State) -> Messages:
        """Drain the request queue.

        Sub-LLM requests are dispatched to concurrent handlers immediately.
        Main-agent requests are returned to the rollout loop as usual.
        """
        request_id_queue: asyncio.Queue[str] = state["request_id_queue"]
        interception_server = self._require_interception_server()

        while True:
            try:
                request_id = await asyncio.wait_for(
                    request_id_queue.get(),
                    timeout=self.poll_interval,
                )
            except asyncio.TimeoutError:
                # Check tunnel liveness
                if self._tunnel is not None and not self._tunnel.is_running:
                    frpc_output = "\n".join(self._tunnel.recent_output)
                    raise vf.TunnelError(
                        f"Tunnel process died during rollout. "
                        f"frpc output:\n{frpc_output}"
                    )
                # Check agent completion / timeout
                if await self.check_agent_completed(state):
                    state["agent_completed"] = True
                    return []
                if time.time() - state["timing"]["start_time"] > self.timeout_seconds:
                    return []
                continue

            intercept = interception_server.intercepts[request_id]

            if self._is_sub_llm_request(intercept):
                # Fire-and-forget: handled concurrently outside the loop
                asyncio.create_task(
                    self._handle_sub_llm_request(state, request_id, intercept)
                )
                continue

            # Main-agent request → return to rollout loop
            state["current_request_id"] = request_id
            return self.normalize_intercepted_messages(intercept["messages"])

    # ------------------------------------------------------------------
    # Concurrent sub-LLM handler
    # ------------------------------------------------------------------

    async def _handle_sub_llm_request(
        self,
        state: State,
        request_id: str,
        intercept: dict[str, Any],
    ) -> None:
        """Handle a single sub-LLM request outside the rollout loop."""
        async with self._sub_llm_semaphore:
            model = self.sub_model or state.get("model")
            prompt = self.normalize_intercepted_messages(intercept["messages"])

            tool_defs: list[Tool] | None = None
            intercept_tools = intercept.get("tools")
            if intercept_tools:
                tool_defs = self.normalize_intercepted_tools(intercept_tools) or None

            response: Response | None = None
            error: BaseException | None = None
            try:
                # Call the model directly via Environment.get_model_response,
                # bypassing CliAgentEnv's intercept-delivery logic (we handle
                # delivery ourselves in the finally block).
                response = await vf.Environment.get_model_response(
                    self,
                    state=state,
                    prompt=prompt,
                    model=model,
                    tool_defs=tool_defs,
                )
            except BaseException as e:
                error = e
                logger.warning("Sub-LLM request %s failed: %s", request_id, e)
            finally:
                if intercept.get("stream"):
                    await synthesize_stream(intercept, response, error)
                else:
                    deliver_response(intercept, response, error)
                # Clean up intercept entry
                self._require_interception_server().intercepts.pop(request_id, None)

            if response is not None:
                self._update_sub_metrics(state, response)
                if self.include_sub_llm_in_trajectory:
                    state["trajectory"].append(
                        {
                            "prompt_messages": prompt,
                            "response": response,
                            "extras": {
                                "is_sub_llm_call": True,
                                "agent_role": intercept.get("model", ""),
                            },
                        }
                    )

    # ------------------------------------------------------------------
    # Main-agent model response (metrics only)
    # ------------------------------------------------------------------

    async def get_model_response(
        self,
        state: State,
        prompt: Messages | str,
        client: Any | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
        message_type: MessageType | None = None,
    ) -> Response:
        """Forward to parent and update main-agent metrics."""
        response = await super().get_model_response(
            state=state,
            prompt=prompt,
            client=client,
            model=model,
            tool_defs=tool_defs,
            sampling_args=sampling_args,
            message_type=message_type,
        )
        # Only count non-empty turns (skip the synthetic agent-completed step)
        if prompt:
            self._update_main_metrics(state, response)
        return response

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_token_counts(response: Response) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if not usage:
            return 0, 0
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _update_main_metrics(self, state: State, response: Response) -> None:
        prompt_tokens, completion_tokens = self._extract_token_counts(response)
        state["main_turns"] = state.get("main_turns", 0) + 1
        state["main_prompt_tokens"] = state.get("main_prompt_tokens", 0) + prompt_tokens
        state["main_completion_tokens"] = (
            state.get("main_completion_tokens", 0) + completion_tokens
        )

    def _update_sub_metrics(self, state: State, response: Response) -> None:
        prompt_tokens, completion_tokens = self._extract_token_counts(response)
        state["sub_llm_turns"] = state.get("sub_llm_turns", 0) + 1
        state["sub_llm_prompt_tokens"] = (
            state.get("sub_llm_prompt_tokens", 0) + prompt_tokens
        )
        state["sub_llm_completion_tokens"] = (
            state.get("sub_llm_completion_tokens", 0) + completion_tokens
        )
