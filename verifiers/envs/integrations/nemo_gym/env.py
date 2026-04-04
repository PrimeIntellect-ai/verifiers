from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import RolloutInput, SamplingArgs, State

from .utils import (
    _map_nemo_gym_result_to_state,
    _resolve_agent_name,
    _reward_from_nemo_gym,
)


class NemoGymEnv(vf.Environment):
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
        policy_base_url: str | None = None,
        policy_api_key: str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ):
        self.gym_configs = gym_configs
        self.vllm_server_host = vllm_server_host
        self.vllm_server_port = vllm_server_port
        self.head_server_host = head_server_host
        self.head_server_port = head_server_port
        self.head_server_client_host = head_server_client_host
        self.policy_base_url = policy_base_url
        self.policy_api_key = policy_api_key

        self._run_helper: Any | None = None
        self._rch: Any | None = None
        self._head_server_config: Any | None = None
        self._server_lock = asyncio.Lock()
        self._agent_name: str | None = None

        self._bg_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._bg_loop.run_forever, daemon=True, name="nemo-gym-loop"
        )
        self._bg_thread.start()

        super().__init__(
            dataset=dataset,
            rubric=rubric or vf.Rubric(funcs=[_reward_from_nemo_gym], weights=[1.0]),
            system_prompt=system_prompt,
            message_type="chat",
            **kwargs,
        )

    def _start_run_helper(self, model: str) -> None:
        try:
            from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
            from nemo_gym.rollout_collection import RolloutCollectionHelper
            from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
            from omegaconf import DictConfig
        except ImportError as exc:
            raise ImportError(
                "NemoGymEnv currently requires nemo-gym installed as an editable local clone (this should be resolved on 0.3):\n"
                "  git clone https://github.com/NVIDIA-NeMo/Gym /path/to/Gym\n"
                "  pip install -e /path/to/Gym"
            ) from exc

        import importlib.util
        from pathlib import Path

        responses_spec = importlib.util.find_spec("responses_api_models")
        if responses_spec and responses_spec.submodule_search_locations:
            responses_root = Path(next(iter(responses_spec.submodule_search_locations)))
            policy_model_config = str(
                responses_root
                / "vllm_model"
                / "configs"
                / "vllm_model_for_training.yaml"
            )
        else:
            raise RuntimeError(
                "Could not locate responses_api_models. "
                "nemo-gym must be installed as an editable local clone: pip install -e /path/to/Gym (this should be resolved on 0.3)."
            )

        config = {
            HEAD_SERVER_KEY_NAME: {
                "host": self.head_server_host,
                "port": self.head_server_port,
            },
            "config_paths": [
                policy_model_config,
                *self.gym_configs,
            ],
            "policy_base_url": self.policy_base_url
            or f"http://{self.vllm_server_host}:{self.vllm_server_port}/v1",
            "policy_api_key": self.policy_api_key
            or os.environ.get("POLICY_API_KEY", "EMPTY"),
            "policy_model_name": model,
            "global_aiohttp_connector_limit_per_host": 16_384,
            "global_aiohttp_connector_limit": 65_536,
            "skip_venv_if_present": True,
        }

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if hf_token:
            config["hf_token"] = hf_token

        rh = RunHelper()
        rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                initial_global_config_dict=DictConfig(config),
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
        async with self._server_lock:
            if self._rch is not None:
                return self._rch, self._head_server_config
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._start_run_helper, model)
        return self._rch, self._head_server_config

    @vf.teardown
    async def teardown_server(self) -> None:
        if self._run_helper is not None:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._run_helper.shutdown)
            except Exception:
                pass
            self._run_helper = None
            self._rch = None
            self._head_server_config = None
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)

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
            if "agent_ref" not in dataset_row:
                if self._agent_name is None:
                    self._agent_name = _resolve_agent_name(self.gym_configs[0])
                dataset_row["agent_ref"] = {"name": self._agent_name}

            rcp: dict[str, Any] = dataset_row.setdefault("responses_create_params", {})
            if sampling_args:
                for key in ("temperature", "top_p"):
                    if sampling_args.get(key) is not None:
                        rcp[key] = sampling_args[key]

            # reuse the zmq loop, kinda ugly
            async def _run() -> Any:
                for task in rch.run_examples(
                    examples=[dataset_row],
                    head_server_config=head_server_config,
                ):
                    _row, result = await task
                    return result
                return None

            future = asyncio.run_coroutine_threadsafe(_run(), self._bg_loop)
            nemo_gym_result = await asyncio.get_running_loop().run_in_executor(
                None, future.result
            )

        except Exception as exc:
            state["error"] = vf.InfraError(
                f"NemoGymEnv rollout failed: {type(exc).__name__}: {exc}"
            )
            state["completion"] = []
            state["is_completed"] = True
            state["stop_condition"] = "has_error"
            _fill_timing(state, start_time)
            return state

        _map_nemo_gym_result_to_state(state, nemo_gym_result, model)
        state["stop_condition"] = "has_error" if state.get("error") else "completed"
        state["is_completed"] = True
        _fill_timing(state, start_time)
        return state


def _fill_timing(state: State, start_time: float) -> None:
    elapsed_ms = (time.time() - start_time) * 1000.0
    state["timing"]["generation_ms"] = elapsed_ms
    state["timing"]["total_ms"] = elapsed_ms
