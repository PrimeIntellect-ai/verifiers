from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State

from .utils import _map_nemo_result_to_state, _reward_from_nemo
from verifiers.envs.integrations.nemo_gym.utils import _build_dataset  # noqa: F401


class NemoGymAgentEnv(vf.Environment):
    """NeMo Gym integration via the NeMo Gym agent server (RunHelper).

    Delegates the entire multi-turn rollout loop to the agent server: it calls
    vLLM directly, manages tool interactions with resource servers, and returns
    the completed rollout with per-turn token IDs, log probs, and reward.
    verifiers does not drive the LLM here.

    Parameters
    ----------
    gym_configs:
        Paths to NeMo Gym resource-server YAML config files passed as
        ``config_paths`` to RunHelper.
    dataset:
        HuggingFace Dataset produced by ``_build_dataset``, with
        ``info["dataset_row_json"]`` carrying the serialised NeMo Gym row.
    rubric:
        Optional custom rubric. Defaults to reading ``state["nemo_reward"]``.
    vllm_server_host / vllm_server_port:
        vLLM policy server address. Should be the same instance verifiers uses.
    head_server_host / head_server_port:
        Address the NeMo Gym head server (RunHelper) will listen on.
    head_server_client_host:
        Host RolloutCollectionHelper uses to reach the head server.
    """

    def __init__(
        self,
        *,
        gym_configs: list[str],
        dataset: Dataset,
        rubric: vf.Rubric | None = None,
        vllm_server_host: str = "127.0.0.1",
        vllm_server_port: int = 8000,
        head_server_host: str = "0.0.0.0",
        head_server_port: int = 11000,
        head_server_client_host: str = "127.0.0.1",
        system_prompt: str | None = None,
        **kwargs: Any,
    ):
        self.gym_configs = gym_configs
        self.vllm_server_host = vllm_server_host
        self.vllm_server_port = vllm_server_port
        self.head_server_host = head_server_host
        self.head_server_port = head_server_port
        self.head_server_client_host = head_server_client_host

        # Lazily initialised on first rollout.
        self._run_helper: Any | None = None
        self._rch: Any | None = None
        self._head_server_config: Any | None = None
        self._server_lock: asyncio.Lock | None = None

        super().__init__(
            dataset=dataset,
            rubric=rubric or vf.Rubric(funcs=[_reward_from_nemo], weights=[1.0]),
            system_prompt=system_prompt,
            message_type="chat",
            **kwargs,
        )

    def _start_run_helper(self, model: str) -> None:
        """Synchronous RunHelper startup — called from an executor."""
        try:
            from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
            from nemo_gym.rollout_collection import RolloutCollectionHelper
            from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
            from omegaconf import DictConfig
        except ImportError as exc:
            raise ImportError(
                "NemoGymAgentEnv requires nemo-gym. Install with: pip install nemo-gym"
            ) from exc

        initial_global_config = {
            HEAD_SERVER_KEY_NAME: {
                "host": self.head_server_host,
                "port": self.head_server_port,
            },
            "config_paths": self.gym_configs,
            "policy_base_url": f"http://{self.vllm_server_host}:{self.vllm_server_port}/v1",
            "policy_api_key": "EMPTY",
            "policy_model_name": model,
            "global_aiohttp_connector_limit_per_host": 16_384,
            "global_aiohttp_connector_limit": 65_536,
        }

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            initial_global_config["hf_token"] = hf_token

        rh = RunHelper()
        rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                initial_global_config_dict=DictConfig(initial_global_config),
                skip_load_from_cli=True,
            )
        )

        self._run_helper = rh
        self._head_server_config = BaseServerConfig(
            host=self.head_server_client_host,
            port=self.head_server_port,
        )
        self._rch = RolloutCollectionHelper()

    async def _ensure_server(self, model: str) -> tuple[Any, Any]:
        if self._rch is not None:
            return self._rch, self._head_server_config

        if self._server_lock is None:
            self._server_lock = asyncio.Lock()

        async with self._server_lock:
            if self._rch is not None:
                return self._rch, self._head_server_config
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._start_run_helper, model)

        return self._rch, self._head_server_config

    @vf.teardown
    async def teardown_agent_server(self) -> None:
        if self._run_helper is not None:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._run_helper.shutdown)
            except Exception:
                pass
            self._run_helper = None
            self._rch = None
            self._head_server_config = None

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        start_time: float = state["timing"]["start_time"]

        try:
            rch, head_server_config = await self._ensure_server(model)

            dataset_row: dict[str, Any] = json.loads(state["info"]["dataset_row_json"])
            dataset_row["_rowidx"] = 0

            responses_create_params: dict[str, Any] = dataset_row.setdefault(
                "responses_create_params", {}
            )
            if sampling_args:
                for key in ("temperature", "top_p", "max_tokens"):
                    if key in sampling_args:
                        responses_create_params[key] = sampling_args[key]

            nemo_result: Any = None
            for task in rch.run_examples(
                examples=[dataset_row],
                head_server_config=head_server_config,
            ):
                _row, nemo_result = await task

        except Exception as exc:
            state["error"] = vf.InfraError(
                f"NemoGymAgentEnv rollout failed: {type(exc).__name__}: {exc}"
            )
            state["completion"] = []
            state["is_completed"] = True
            state["stop_condition"] = "has_error"
            _fill_timing(state, start_time)
            return state

        _map_nemo_result_to_state(state, nemo_result, model)

        state["stop_condition"] = "has_error" if state.get("error") else "completed"
        state["is_completed"] = True
        _fill_timing(state, start_time)
        return state


def _fill_timing(state: State, start_time: float) -> None:
    elapsed_ms = (time.time() - start_time) * 1000.0
    state["timing"]["generation_ms"] = elapsed_ms
    state["timing"]["total_ms"] = elapsed_ms
