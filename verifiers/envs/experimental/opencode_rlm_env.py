"""
OpenCode RLM Environment.

Extends OpenCodeEnv with the RLM plugin (https://github.com/snimu/oc),
adding concurrent sub-LLM handling via ``subagent`` / ``llm-subcall``.

Sub-agent calls are identified by the ``X-RLM-Role: sub`` HTTP header
(set by the OC plugin) and handled concurrently with semaphore-based
parallelism control.  Main-agent calls go through the normal sequential
rollout loop.
"""

import asyncio
import json
from typing import Any

import verifiers as vf
from verifiers.envs.experimental.opencode_env import OpenCodeEnv
from verifiers.types import (
    Messages,
    Response,
    State,
    Tool,
)
from verifiers.utils.interception_utils import deliver_response, synthesize_stream


class OpenCodeRLMMonitorRubric(vf.Rubric):
    """Tracks main-agent and sub-LLM metrics separately."""

    _STATE_METRICS = [
        "sub_llm_turns",
        "sub_llm_prompt_tokens",
        "sub_llm_completion_tokens",
    ]

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_metric(self.main_turns)
        self.add_metric(self.main_prompt_tokens)
        self.add_metric(self.main_completion_tokens)
        for name in self._STATE_METRICS:
            fn = self._make_state_metric(name)
            setattr(self, name, fn)
            self.add_metric(fn)

    @staticmethod
    def _is_main_step(step: dict) -> bool:
        return not (step.get("extras") or {}).get("is_sub_llm_call")

    @staticmethod
    async def main_turns(state: State) -> float:
        return float(
            sum(
                1
                for s in state.get("trajectory", [])
                if OpenCodeRLMMonitorRubric._is_main_step(s)
            )
        )

    @staticmethod
    async def main_prompt_tokens(state: State) -> float:
        total = 0
        for step in state.get("trajectory", []):
            if not OpenCodeRLMMonitorRubric._is_main_step(step):
                continue
            resp = step.get("response")
            usage = getattr(resp, "usage", None) if resp else None
            if usage:
                total += int(getattr(usage, "prompt_tokens", 0) or 0)
        return float(total)

    @staticmethod
    async def main_completion_tokens(state: State) -> float:
        total = 0
        for step in state.get("trajectory", []):
            if not OpenCodeRLMMonitorRubric._is_main_step(step):
                continue
            resp = step.get("response")
            usage = getattr(resp, "usage", None) if resp else None
            if usage:
                total += int(getattr(usage, "completion_tokens", 0) or 0)
        return float(total)

    @staticmethod
    def _make_state_metric(key: str):
        async def metric(state: State) -> float:
            return float(state.get(key, 0))

        metric.__name__ = key
        return metric


# Extends the default OpenCodeEnv template with bun + plugin installation.
RLM_RUN_COMMAND_TEMPLATE = """\
set -e

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
set +e
opencode run < {prompt_path} > {logs_path} 2>&1
_oc_exit=$?
set -e
cat {logs_path}
exit $_oc_exit
"""


class OpenCodeRLMEnv(OpenCodeEnv):
    """
    OpenCodeEnv with the RLM plugin for recursive sub-LLM calls.

    Intercepts all API calls from the main agent and its sub-agents.
    Sub-LLM requests are identified by the ``X-RLM-Role: sub`` HTTP
    header (set by the OC plugin) and handled concurrently with a
    semaphore.  All other requests go through the normal sequential
    rollout loop.

    Args:
        plugin_repo: GitHub ``<org>/<repo>`` for the RLM plugin.
        plugin_install_path: Where the plugin is cloned inside the sandbox.
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

    def __init__(
        self,
        plugin_repo: str = DEFAULT_PLUGIN_REPO,
        plugin_branch: str = DEFAULT_PLUGIN_BRANCH,
        plugin_install_path: str = DEFAULT_PLUGIN_INSTALL_PATH,
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
        self.sub_model = sub_model
        self.sub_llm_max_turns = sub_llm_max_turns
        self.sub_timeout_ms = sub_timeout_ms
        self.include_sub_llm_in_trajectory = include_sub_llm_in_trajectory
        self._sub_llm_semaphore = asyncio.Semaphore(max_sub_llm_parallelism)

        kwargs.setdefault("run_command_template", RLM_RUN_COMMAND_TEMPLATE)

        super().__init__(**kwargs)
        self.add_rubric(OpenCodeRLMMonitorRubric())

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

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env = await super().build_env_vars(state)
        # Use the OC proxy's custom tool-calling loop for subagent calls
        # instead of OpenCode child sessions (which can't be distinguished
        # from the main agent and would serialize in the rollout loop).
        env["RLM_SUBAGENT_VIA_TOOL_LOOP"] = "true"
        env["RLM_SUB_MAX_TURNS"] = str(self.sub_llm_max_turns)
        env["RLM_SUB_TIMEOUT"] = str(self.sub_timeout_ms)
        return env

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state.setdefault("sub_llm_turns", 0)
        state.setdefault("sub_llm_prompt_tokens", 0)
        state.setdefault("sub_llm_completion_tokens", 0)
        state.setdefault("_sub_llm_tasks", set())
        return state

    @staticmethod
    def _is_sub_llm_request(intercept: dict[str, Any]) -> bool:
        return intercept.get("headers", {}).get("x-rlm-role") == "sub"

    async def get_prompt_messages(self, state: State) -> Messages:
        """Extends parent to route sub-LLM requests concurrently.

        Uses _poll_next_request from CliAgentEnv for the polling loop.
        Sub-LLM requests (identified by X-RLM-Role header) are dispatched
        to concurrent handlers. Main-agent requests are returned normally.
        """
        interception_server = self._require_interception_server()

        while True:
            request_id = await self._poll_next_request(state)
            if request_id is None:
                # Agent completed or timed out — drain pending sub-LLM tasks
                tasks: set = state.get("_sub_llm_tasks", set())
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                return []

            intercept = interception_server.intercepts[request_id]

            if self._is_sub_llm_request(intercept):
                task = asyncio.create_task(
                    self._handle_sub_llm_request(state, request_id, intercept)
                )
                state["_sub_llm_tasks"].add(task)
                task.add_done_callback(state["_sub_llm_tasks"].discard)
                continue

            state["current_request_id"] = request_id
            return self.normalize_intercepted_messages(intercept["messages"])

    @vf.cleanup(priority=1)
    async def cancel_sub_llm_tasks(self, state: State) -> None:
        """Cancel any in-flight sub-LLM tasks during rollout cleanup."""
        tasks: set = state.get("_sub_llm_tasks", set())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
                # delivery ourselves below).
                response = await vf.Environment.get_model_response(
                    self,
                    state=state,
                    prompt=prompt,
                    model=model,
                    tool_defs=tool_defs,
                )
            except BaseException as e:
                error = e
                if isinstance(e, Exception):
                    self.logger.warning("Sub-LLM request %s failed: %s", request_id, e)
                else:
                    # CancelledError / KeyboardInterrupt — deliver error to
                    # unblock the HTTP future, then re-raise.
                    self.logger.debug("Sub-LLM request %s cancelled", request_id)
            finally:
                if intercept.get("stream"):
                    await synthesize_stream(intercept, response, error)
                else:
                    deliver_response(intercept, response, error)
                # Clean up intercept entry
                self._require_interception_server().intercepts.pop(request_id, None)

            # Re-raise non-Exception BaseExceptions (CancelledError, etc.)
            # after delivering the response so the future doesn't hang.
            if error is not None and not isinstance(error, Exception):
                raise error

            if response is not None:
                self._update_sub_metrics(state, response)
                if self.include_sub_llm_in_trajectory:
                    completion = [response.message] if response.message else []
                    prompt_tokens, completion_tokens = self._extract_token_counts(
                        response
                    )
                    state["trajectory"].append(
                        {
                            "prompt": prompt,
                            "completion": completion,
                            "response": response,
                            "tokens": {
                                "prompt": prompt_tokens,
                                "completion": completion_tokens,
                                "reasoning": 0,
                            },
                            "reward": None,
                            "advantage": None,
                            "is_truncated": False,
                            "trajectory_id": state.get("trajectory_id", ""),
                            "extras": {
                                "is_sub_llm_call": True,
                                "agent_role": intercept.get("model", ""),
                            },
                        }
                    )

    @staticmethod
    def _extract_token_counts(response: Response) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if not usage:
            return 0, 0
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
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
