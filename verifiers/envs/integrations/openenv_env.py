from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import requests
import tenacity as tc
from datasets import Dataset

import verifiers as vf
from verifiers.types import ChatMessage, ChatMessages
from verifiers.utils.tool_utils import is_valid_tool_content_parts

try:
    from openenv.core.generic_client import GenericEnvClient
except ImportError as e:
    raise ImportError(
        "OpenEnvEnv requires openenv-core. Install with: uv add 'verifiers[openenv]'"
    ) from e

try:
    from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest, SandboxClient
    from prime_sandboxes.core import APIClient
except ImportError as e:
    raise ImportError(
        "OpenEnvEnv requires prime-sandboxes. Install with: uv add prime-sandboxes"
    ) from e

logger = logging.getLogger(__name__)


@dataclass
class _OpenEnvServer:
    sandbox_id: str
    exposure_id: str
    base_url: str
    port: int
    contract: str


class OpenEnvEpisodicSumRubric(vf.Rubric):
    def __init__(self, weight: float = 1.0, **kwargs: Any):
        async def sum_step_rewards(state: vf.State) -> float:
            return float(
                sum(
                    float(step.get("reward", 0.0) or 0.0)
                    for step in state.get("trajectory", [])
                )
            )

        super().__init__(funcs=[sum_step_rewards], weights=[weight], **kwargs)


class OpenEnvEnv(vf.MultiTurnEnv):
    """
    Drop-in OpenEnv integration for Verifiers.

    - Always runs inside Prime Sandboxes.
    - Uses prebuilt container images at runtime (from `.build.json`).
    - Uses seeds as the generic dataset mechanism.
    - Supports both gym (step/reset) and MCP tool environments.
    - Expects a prompt renderer that maps observations to chat messages.
    """

    _DATASET_RESET_MAX_RETRIES = 5
    _DATASET_RESET_BASE_BACKOFF_SECONDS = 0.25
    _DATASET_RESET_MAX_BACKOFF_SECONDS = 3.0

    def __init__(
        self,
        openenv_project: str | Path,
        num_train_examples: int = 100,
        num_eval_examples: int = 50,
        seed: int = 0,
        prompt_renderer: Callable[..., ChatMessages] | None = None,
        max_turns: int = -1,
        rubric: vf.Rubric | None = None,
        startup_timeout_seconds: int = 30,
        startup_poll_interval_seconds: float = 1.0,
        health_request_timeout_seconds: float = 2.0,
        schema_request_timeout_seconds: float = 5.0,
        wait_for_creation_max_attempts: int = 20,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        **kwargs: Any,
    ):
        self.openenv_project = str(openenv_project)
        self.num_train_examples = num_train_examples
        self.num_eval_examples = num_eval_examples
        self.seed = seed
        if prompt_renderer is None:
            raise ValueError(
                "OpenEnvEnv requires `prompt_renderer`. "
                "Define a renderer in your environment module that converts observations "
                "to non-empty chat messages."
            )
        self.prompt_renderer = prompt_renderer
        self.startup_timeout_seconds = startup_timeout_seconds
        self.startup_poll_interval_seconds = startup_poll_interval_seconds
        self.health_request_timeout_seconds = health_request_timeout_seconds
        self.schema_request_timeout_seconds = schema_request_timeout_seconds
        self.wait_for_creation_max_attempts = wait_for_creation_max_attempts

        self._active_servers: dict[str, _OpenEnvServer] = {}
        self._contract: str | None = None  # "gym" or "mcp"
        self._action_schema: dict[str, Any] | None = None
        self._mcp_tools: list[Any] | None = None
        self._mcp_request_id = 0

        self._with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(cast(Any, logger), logging.WARNING),
            reraise=True,
        ).wraps

        dataset, eval_dataset = self._build_seed_datasets()

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric or OpenEnvEpisodicSumRubric(),
            max_turns=max_turns,
            message_type="chat",
            **kwargs,
        )

    async def start_server(
        self,
        address: str | None = None,
        extra_env_kwargs: dict[str, Any] | None = None,
        log_level: str | None = None,
        log_file: str | None = None,
        startup_timeout: float = 120.0,
    ) -> None:
        await super().start_server(
            address=address,
            extra_env_kwargs=extra_env_kwargs or {},
            log_level=log_level,
            log_file=log_file,
            startup_timeout=startup_timeout,
        )

    def _build_seed_datasets(self) -> tuple[Dataset, Dataset | None]:
        total = self.num_train_examples + self.num_eval_examples
        rows = self._build_seed_rows(total)
        train_rows = rows[: self.num_train_examples]
        eval_rows = rows[self.num_train_examples :]

        dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
        self._validate_dataset_prompts(dataset, split_name="train")
        if eval_dataset is not None:
            self._validate_dataset_prompts(eval_dataset, split_name="eval")
        return dataset, eval_dataset

    def _build_seed_rows(self, total: int) -> list[dict[str, Any]]:
        if total <= 0:
            return []
        project_path = self._resolve_project_path()
        image, port, start_command, contract = self._resolve_runtime_config(
            project_path
        )
        server = self._launch_image_server_sync(image, port, start_command, contract)
        rows: list[dict[str, Any]] = []
        mcp_action_schema: dict[str, Any] | None = None
        try:
            if contract == "mcp":
                mcp_action_schema = self._fetch_action_schema_sync(server.base_url)
            seeds = [self.seed + i for i in range(total)]
            observations = self._fetch_reset_observations_sync(server.base_url, seeds)
            for seed, obs in zip(seeds, observations, strict=False):
                prompt = self._render_observation_messages(
                    obs,
                    context="reset",
                    action_schema=mcp_action_schema if contract == "mcp" else None,
                    contract=contract,
                    seed=seed,
                )
                rows.append(
                    {
                        "prompt": prompt,
                        "info": {"seed": seed},
                    }
                )
            return rows
        finally:
            self._cleanup_server_sync(server)

    def _fetch_reset_observations_sync(
        self, base_url: str, seeds: list[int]
    ) -> list[Any]:
        if not seeds:
            return []
        client = GenericEnvClient(base_url=base_url)
        observations: list[Any] = []
        try:
            self._connect_generic_client_sync(client)
            for seed in seeds:
                observation = self._reset_with_retry_sync(client, seed)
                observations.append(observation)
        finally:
            try:
                client.close()
            except Exception:
                pass
        return observations

    def _connect_generic_client_sync(self, client: GenericEnvClient) -> None:
        try:
            client.connect()
        except Exception as e:
            raise RuntimeError(
                "OpenEnv dataset bootstrap failed to establish a reset session."
            ) from e

    def _reset_with_retry_sync(self, client: GenericEnvClient, seed: int) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self._DATASET_RESET_MAX_RETRIES + 1):
            try:
                result = client.reset(seed=int(seed))
                observation = getattr(result, "observation", None)
                if observation is None:
                    raise RuntimeError(
                        "OpenEnv reset result is missing required `observation`."
                    )
                return observation
            except Exception as e:
                last_error = e
                if attempt >= self._DATASET_RESET_MAX_RETRIES:
                    break
                backoff = min(
                    self._DATASET_RESET_BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                    self._DATASET_RESET_MAX_BACKOFF_SECONDS,
                )
                time.sleep(backoff)
                # Reconnect the session before retrying this seed.
                try:
                    client.close()
                except Exception:
                    pass
                self._connect_generic_client_sync(client)
        assert last_error is not None
        raise RuntimeError(
            f"OpenEnv reset failed during dataset build for seed={seed} "
            f"after {self._DATASET_RESET_MAX_RETRIES} attempts."
        ) from last_error

    def _fetch_action_schema_sync(self, base_url: str) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{base_url}/schema",
                timeout=self.schema_request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            raise RuntimeError(
                "OpenEnv schema fetch failed while building dataset."
            ) from e
        if not isinstance(payload, dict):
            raise RuntimeError("OpenEnv /schema response must be a JSON object.")
        action_schema = payload.get("action", {})
        if not isinstance(action_schema, dict):
            raise RuntimeError(
                "OpenEnv /schema response missing object `action` schema."
            )
        return action_schema

    def _validate_dataset_prompts(self, dataset: Dataset, split_name: str) -> None:
        if "prompt" not in dataset.column_names:
            raise RuntimeError(
                f"OpenEnv {split_name} dataset is invalid: missing required `prompt` column."
            )
        prompts = dataset["prompt"]
        for idx, prompt in enumerate(prompts):
            if not self._looks_like_messages(prompt):
                raise RuntimeError(
                    f"OpenEnv {split_name} dataset is invalid at row {idx}: "
                    "`prompt` must be a non-empty chat messages list."
                )
            messages = cast(list[dict[str, Any]], prompt)
            if not messages:
                raise RuntimeError(
                    f"OpenEnv {split_name} dataset is invalid at row {idx}: "
                    "`prompt` cannot be empty."
                )
            for msg_idx, msg in enumerate(messages):
                content = msg.get("content")
                if content is None:
                    raise RuntimeError(
                        f"OpenEnv {split_name} dataset is invalid at row {idx}, "
                        f"message {msg_idx}: `content` cannot be null."
                    )

    async def setup_state(self, state: vf.State) -> vf.State:
        try:
            server = await self._create_server()
            state["openenv_server"] = server
            state["openenv_contract"] = server.contract
            action_schema = await self._fetch_action_schema(server.base_url)
            self._assert_contract_matches_schema(server.contract, action_schema)
            state["openenv_action_schema"] = action_schema
            if self._contract is None:
                self._contract = server.contract
            if self._action_schema is None:
                self._action_schema = action_schema

            seed = 0
            info = state.get("info")
            if isinstance(info, dict):
                seed = int(info.get("seed", 0))

            if server.contract == "mcp":
                mcp_client = GenericEnvClient(base_url=server.base_url)
                await self._invoke(cast(Any, mcp_client).connect)
                state["openenv_mcp_client"] = mcp_client
                if self._mcp_tools is None:
                    self._mcp_tools = await self._mcp_list_tools(mcp_client)
                state["oai_tools"] = self._convert_mcp_tools(self._mcp_tools)
                result = await self._invoke(cast(Any, mcp_client).reset, seed=seed)
                state["openenv_done"] = bool(result.done)
                state["prompt"] = self._require_prompt_messages(state)
                return state

            client = GenericEnvClient(base_url=server.base_url)
            await self._invoke(cast(Any, client).connect)
            state["openenv_client"] = client
            result = await self._invoke(cast(Any, client).reset, seed=seed)
            state["openenv_done"] = bool(result.done)
            state["prompt"] = self._require_prompt_messages(state)
            return state
        except Exception:
            await self._cleanup_openenv_state(state)
            raise

    def _make_user_message(self, content: str) -> ChatMessage:
        return cast(ChatMessage, {"role": "user", "content": content})

    def _make_tool_message(self, content: Any, tool_call_id: str) -> ChatMessage:
        return cast(
            ChatMessage,
            {"role": "tool", "content": content, "tool_call_id": tool_call_id},
        )

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        contract = state.get("openenv_contract") or self._contract
        if contract == "mcp":
            return await self._mcp_env_response(messages, state)
        return await self._gym_env_response(messages, state)

    async def _gym_env_response(
        self, messages: vf.Messages, state: vf.State
    ) -> vf.Messages:
        assert isinstance(messages, list)
        last_msg = messages[-1]
        if last_msg.get("role") != "assistant":
            return [self._make_user_message("Expected assistant response.")]

        raw_text = str(last_msg.get("content", "")).strip()
        action_schema = state.get("openenv_action_schema") or self._action_schema or {}
        action = self._parse_action(raw_text, action_schema)

        client: Any = state["openenv_client"]
        result = await self._invoke(client.step, action)

        if state["trajectory"]:
            state["trajectory"][-1]["reward"] = result.reward

        state["openenv_done"] = bool(result.done)
        obs_messages = self._render_observation_messages(
            result.observation,
            context="step",
            action_schema=action_schema if isinstance(action_schema, dict) else None,
            contract="gym",
        )
        return obs_messages

    async def _mcp_env_response(
        self, messages: vf.Messages, state: vf.State
    ) -> vf.Messages:
        assert isinstance(messages, list)
        last_msg = messages[-1]
        tool_calls = (
            last_msg.get("tool_calls", []) if isinstance(last_msg, dict) else []
        )
        if not tool_calls:
            return cast(ChatMessages, [])

        mcp_client: Any = state["openenv_mcp_client"]
        tool_messages: ChatMessages = []
        total_reward = 0.0
        done = False
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "")
            fn_payload = tool_call.get("function", {})
            tool_name = str(fn_payload.get("name", "")).strip()
            try:
                tool_args = json.loads(fn_payload.get("arguments", "{}"))
                if not isinstance(tool_args, dict):
                    raise ValueError("tool arguments must be an object")
                if not tool_name:
                    raise ValueError("tool name cannot be empty")
                result = await self._mcp_step_tool(
                    mcp_client, tool_name=tool_name, arguments=tool_args
                )
                if isinstance(result.reward, (int, float)):
                    total_reward += float(result.reward)
                done = done or bool(result.done)
                content = self._format_tool_content(
                    self._extract_mcp_tool_content(result.observation)
                )
            except Exception as e:
                content = f"Error: {e}"

            tool_messages.append(self._make_tool_message(content, tool_call_id))
        if state["trajectory"]:
            state["trajectory"][-1]["reward"] = total_reward
        state["openenv_done"] = done
        return tool_messages

    def _format_tool_content(self, result: Any) -> Any:
        if is_valid_tool_content_parts(result):
            return result
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=True)

    @vf.stop
    async def openenv_done(self, state: vf.State) -> bool:
        contract = state.get("openenv_contract") or self._contract
        return bool(state.get("openenv_done")) and contract == "gym"

    @vf.stop
    async def mcp_no_tool_calls(self, state: vf.State) -> bool:
        contract = state.get("openenv_contract") or self._contract
        if contract != "mcp":
            return False
        if state.get("openenv_done"):
            return True
        if not state["trajectory"]:
            return False
        last_msg = state["trajectory"][-1]["completion"][-1]
        return last_msg.get("role") == "assistant" and not last_msg.get("tool_calls")

    async def _cleanup_openenv_state(self, state: vf.State) -> None:
        client = state.pop("openenv_client", None)
        if client is not None:
            try:
                await self._invoke(cast(Any, client).close)
            except Exception:
                pass

        mcp_client = state.pop("openenv_mcp_client", None)
        if mcp_client is not None:
            try:
                await self._invoke(cast(Any, mcp_client).close)
            except Exception:
                pass

        server = state.pop("openenv_server", None)
        if server is not None:
            try:
                await self._cleanup_server(server)
            except Exception:
                pass

    @vf.cleanup
    async def cleanup_openenv(self, state: vf.State) -> None:
        await self._cleanup_openenv_state(state)

    async def _cleanup_server(self, server: _OpenEnvServer) -> None:
        async with AsyncSandboxClient() as sandboxes:
            try:
                await self._with_retry(sandboxes.unexpose)(
                    server.sandbox_id, server.exposure_id
                )
            except Exception:
                pass
            try:
                await self._with_retry(sandboxes.delete)(server.sandbox_id)
            except Exception:
                pass
        self._active_servers.pop(server.sandbox_id, None)

    async def _try_get_logs(
        self, sandboxes: AsyncSandboxClient, sandbox_id: str
    ) -> str | None:
        try:
            logs = await sandboxes.get_logs(sandbox_id)
        except Exception:
            return None
        return self._trim_logs(logs)

    def _trim_logs(self, logs: Any) -> str | None:
        if not logs:
            return None
        logs_str = str(logs)
        if len(logs_str) > 4000:
            return logs_str[-4000:]
        return logs_str

    def _format_sandbox_error(
        self,
        sandbox_id: str,
        context: str,
        err: Exception,
        image: str | None = None,
        logs: str | None = None,
    ) -> vf.SandboxError:
        parts = [f"OpenEnv sandbox {sandbox_id} failed during {context}."]
        status = getattr(err, "status", None) or getattr(err, "sandbox_status", None)
        if status:
            parts.append(f"Status={status}.")
        if image:
            parts.append(f"Image={image}.")
        parts.append(
            "If this uses a custom Dockerfile, ensure the image is built and available in Prime."
        )
        if logs:
            parts.append(f"Logs (tail):\n{logs}")
        return vf.SandboxError(" ".join(parts))

    @vf.teardown
    async def teardown_server(self) -> None:
        if not self._active_servers:
            return
        servers = list(self._active_servers.values())
        for server in servers:
            try:
                await self._cleanup_server(server)
            except Exception:
                pass

    async def _create_server(self) -> _OpenEnvServer:
        project_path = self._resolve_project_path()
        image, port, start_command, contract = self._resolve_runtime_config(
            project_path
        )
        server = await self._launch_image_server(image, port, start_command, contract)
        self._active_servers[server.sandbox_id] = server
        return server

    def _resolve_project_path(self) -> Path:
        path = Path(self.openenv_project).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path
        raise ValueError(
            "OpenEnvEnv requires a local OpenEnv project directory. "
            f"Got: {self.openenv_project}"
        )

    def _resolve_runtime_config(self, project_path: Path) -> tuple[str, int, str, str]:
        manifest = self._read_build_manifest(project_path)
        image = manifest.get("image")
        port = manifest.get("port", 8000)
        start_command = manifest.get("start_command")
        contract = manifest.get("contract")
        if not isinstance(image, str) or not image.strip():
            raise RuntimeError(
                "Invalid .build.json: `image` must be a non-empty string. "
                "Run: vf-build <env-id> (optionally with -p <environments-path>)."
            )
        try:
            port_num = int(port)
        except Exception as e:
            raise RuntimeError("Invalid .build.json: `port` must be an integer.") from e
        if not isinstance(start_command, str) or not start_command.strip():
            raise RuntimeError(
                "Invalid .build.json: `start_command` must be a non-empty string. "
                "Run: vf-build <env-id> (optionally with -p <environments-path>)."
            )
        if not isinstance(contract, str) or contract not in {"gym", "mcp"}:
            raise RuntimeError(
                "Invalid .build.json: `contract` must be either 'gym' or 'mcp'. "
                "Run: vf-build <env-id> (optionally with -p <environments-path>)."
            )
        return image.strip(), port_num, start_command.strip(), contract

    def _read_build_manifest(self, project_path: Path) -> dict[str, Any]:
        manifest_path = project_path / ".build.json"
        if not manifest_path.exists():
            raise RuntimeError(
                "OpenEnv project is missing .build.json. "
                "Run: vf-build <env-id> (optionally with -p <environments-path>) to build and register the image."
            )
        try:
            data = json.loads(manifest_path.read_text())
        except Exception as e:
            raise RuntimeError(
                "Failed to parse OpenEnv build manifest at "
                f"{manifest_path}. Re-run vf-build <env-id>."
            ) from e
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Invalid OpenEnv build manifest at {manifest_path}: expected a JSON object."
            )
        return data

    async def _launch_image_server(
        self, image: str, port: int, start_command: str, contract: str
    ) -> _OpenEnvServer:
        async with AsyncSandboxClient() as sandboxes:
            req = self._build_sandbox_request(image, start_command=start_command)
            try:
                sandbox = await self._with_retry(sandboxes.create)(req)
            except Exception as e:
                raise vf.SandboxError(
                    f"Failed to create OpenEnv sandbox for image {image}."
                ) from e
            exposure = None
            try:
                await sandboxes.wait_for_creation(
                    sandbox.id,
                    max_attempts=self.wait_for_creation_max_attempts,
                )
                exposure = await sandboxes.expose(
                    sandbox.id,
                    port=port,
                    name="openenv-env",
                    protocol="TCP",
                )
                base_url = self._exposure_to_base_url(exposure)
                server = _OpenEnvServer(
                    sandbox_id=sandbox.id,
                    exposure_id=exposure.exposure_id,
                    base_url=base_url,
                    port=port,
                    contract=contract,
                )
                await self._wait_for_ready(server.base_url)
                return server
            except Exception as e:
                logs = await self._try_get_logs(sandboxes, sandbox.id)
                local_health = await self._probe_local_health(
                    sandboxes, sandbox.id, port
                )
                if local_health:
                    logs = (logs + "\n" if logs else "") + local_health
                if exposure is not None:
                    try:
                        await sandboxes.unexpose(sandbox.id, exposure.exposure_id)
                    except Exception:
                        pass
                try:
                    await sandboxes.delete(sandbox.id)
                except Exception:
                    pass
                raise self._format_sandbox_error(
                    sandbox.id, "startup", e, image=image, logs=logs
                ) from e

    def _launch_image_server_sync(
        self, image: str, port: int, start_command: str, contract: str
    ) -> _OpenEnvServer:
        sandboxes = SandboxClient(APIClient())
        req = self._build_sandbox_request(image, start_command=start_command)
        try:
            sandbox = sandboxes.create(req)
        except Exception as e:
            raise vf.SandboxError(
                f"Failed to create OpenEnv sandbox for image {image}."
            ) from e

        exposure: Any | None = None
        try:
            sandboxes.wait_for_creation(
                sandbox.id,
                max_attempts=self.wait_for_creation_max_attempts,
            )
            exposure = sandboxes.expose(
                sandbox.id,
                port=port,
                name="openenv-env",
                protocol="TCP",
            )
            base_url = self._exposure_to_base_url(exposure)
            server = _OpenEnvServer(
                sandbox_id=sandbox.id,
                exposure_id=exposure.exposure_id,
                base_url=base_url,
                port=port,
                contract=contract,
            )
            self._wait_for_ready_sync(server.base_url)
            return server
        except Exception as e:
            logs = self._try_get_logs_sync(sandboxes, sandbox.id)
            local_health = self._probe_local_health_sync(sandboxes, sandbox.id, port)
            if local_health:
                logs = (logs + "\n" if logs else "") + local_health
            if exposure is not None:
                try:
                    sandboxes.unexpose(sandbox.id, exposure.exposure_id)
                except Exception:
                    pass
            try:
                sandboxes.delete(sandbox.id)
            except Exception:
                pass
            raise self._format_sandbox_error(
                sandbox.id, "startup", e, image=image, logs=logs
            ) from e

    async def _probe_local_health(
        self, sandboxes: AsyncSandboxClient, sandbox_id: str, port: int
    ) -> str | None:
        """Debug helper to distinguish app-start failures vs exposure/network failures."""
        cmd = f'sh -lc "curl -sS -m 2 http://localhost:{int(port)}/health 2>&1 || true"'
        try:
            result = await sandboxes.execute_command(
                sandbox_id,
                cmd,
                timeout=5,
            )
        except Exception as e:
            return f"Local /health probe failed to execute: {type(e).__name__}: {e}"
        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        if stdout:
            return f"Local /health probe stdout: {stdout}"
        if stderr:
            return f"Local /health probe stderr: {stderr}"
        return "Local /health probe returned no output."

    def _probe_local_health_sync(
        self, sandboxes: SandboxClient, sandbox_id: str, port: int
    ) -> str | None:
        cmd = f'sh -lc "curl -sS -m 2 http://localhost:{int(port)}/health 2>&1 || true"'
        try:
            result = sandboxes.execute_command(
                sandbox_id,
                cmd,
                timeout=5,
            )
        except Exception as e:
            return f"Local /health probe failed to execute: {type(e).__name__}: {e}"
        stdout = (getattr(result, "stdout", "") or "").strip()
        stderr = (getattr(result, "stderr", "") or "").strip()
        if stdout:
            return f"Local /health probe stdout: {stdout}"
        if stderr:
            return f"Local /health probe stderr: {stderr}"
        return "Local /health probe returned no output."

    def _exposure_to_base_url(self, exposure: Any) -> str:
        endpoint = getattr(exposure, "external_endpoint", None)
        if isinstance(endpoint, str) and endpoint.strip():
            return f"http://{endpoint.strip()}"

        raw_url = str(getattr(exposure, "url", "") or "").strip()
        if raw_url.startswith("tcp://"):
            host_port = raw_url[len("tcp://") :].rstrip("/")
            if host_port:
                return f"http://{host_port}"
        if raw_url.startswith("http://") or raw_url.startswith("https://"):
            return raw_url.rstrip("/")

        raise RuntimeError(
            "OpenEnv sandbox exposure did not provide a usable endpoint URL."
        )

    async def _invoke(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))

    def _next_mcp_request_id(self) -> int:
        self._mcp_request_id += 1
        return self._mcp_request_id

    async def _mcp_rpc(
        self, client: GenericEnvClient, method: str, params: dict[str, Any]
    ) -> Any:
        request = {
            "type": "mcp",
            "data": {
                "jsonrpc": "2.0",
                "id": self._next_mcp_request_id(),
                "method": method,
                "params": params,
            },
        }
        response = await self._invoke(cast(Any, client)._send_and_receive, request)
        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid MCP response type: {type(response).__name__}")
        data = response.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("Invalid MCP response: missing JSON-RPC data object.")
        if data.get("jsonrpc") != "2.0":
            raise RuntimeError("Invalid MCP response: jsonrpc must be '2.0'.")
        if "error" in data:
            err = data["error"]
            if isinstance(err, dict):
                raise RuntimeError(
                    f"MCP RPC error ({err.get('code', 'unknown')}): {err.get('message', err)}"
                )
            raise RuntimeError(f"MCP RPC error: {err}")
        if "result" not in data:
            raise RuntimeError("Invalid MCP response: missing result.")
        return data["result"]

    async def _mcp_list_tools(self, client: GenericEnvClient) -> list[dict[str, Any]]:
        result = await self._mcp_rpc(client, method="tools/list", params={})
        if not isinstance(result, dict):
            raise RuntimeError("Invalid MCP tools/list result: expected object.")
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise RuntimeError("Invalid MCP tools/list result: missing tools list.")
        parsed_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                parsed_tools.append(tool)
        if not parsed_tools:
            raise RuntimeError("MCP tools/list returned no usable tools.")
        return parsed_tools

    async def _mcp_step_tool(
        self, client: GenericEnvClient, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        return await self._invoke(
            client.step,
            {"type": "call_tool", "tool_name": tool_name, "arguments": arguments},
        )

    def _extract_mcp_tool_content(self, observation: Any) -> Any:
        if hasattr(observation, "model_dump"):
            try:
                observation = observation.model_dump()
            except Exception:
                pass
        if not isinstance(observation, dict):
            return observation
        if observation.get("error") is not None:
            return {"error": observation.get("error")}
        return self._unwrap_mcp_result(observation.get("result"))

    def _unwrap_mcp_result(self, value: Any) -> Any:
        if hasattr(value, "data"):
            return cast(Any, value).data
        if isinstance(value, dict) and "data" in value:
            return value["data"]
        return value

    def _build_sandbox_request(
        self, image: str, start_command: str
    ) -> CreateSandboxRequest:
        params: dict[str, Any] = {
            "name": "openenv-env",
            "docker_image": image,
            "start_command": start_command,
            "cpu_cores": 2,
            "memory_gb": 4,
            "disk_size_gb": 10,
            "timeout_minutes": 60,
            "environment_vars": {"ENABLE_WEB_INTERFACE": "false"},
        }
        return CreateSandboxRequest(**cast(Any, params))

    async def _wait_for_ready(
        self, base_url: str, timeout_s: int | None = None
    ) -> None:
        timeout = timeout_s if timeout_s is not None else self.startup_timeout_seconds
        loop = asyncio.get_running_loop()
        start = loop.time()
        last_health_error = "no attempts"
        while (loop.time() - start) < timeout:
            ok, detail = await asyncio.to_thread(self._check_health, base_url)
            if ok:
                return
            last_health_error = detail
            await asyncio.sleep(self.startup_poll_interval_seconds)
        raise RuntimeError(
            "OpenEnv server not ready. "
            f"Health check timeout={timeout}s, url={base_url}, "
            f"last error: {last_health_error}"
        )

    def _wait_for_ready_sync(self, base_url: str, timeout_s: int | None = None) -> None:
        timeout = timeout_s if timeout_s is not None else self.startup_timeout_seconds
        start = time.monotonic()
        last_health_error = "no attempts"
        while (time.monotonic() - start) < timeout:
            ok, detail = self._check_health(base_url)
            if ok:
                return
            last_health_error = detail
            time.sleep(self.startup_poll_interval_seconds)
        raise RuntimeError(
            "OpenEnv server not ready. "
            f"Health check timeout={timeout}s, url={base_url}, "
            f"last error: {last_health_error}"
        )

    def _try_get_logs_sync(
        self, sandboxes: SandboxClient, sandbox_id: str
    ) -> str | None:
        try:
            logs = sandboxes.get_logs(sandbox_id)
        except Exception:
            return None
        return self._trim_logs(logs)

    def _check_health(self, base_url: str) -> tuple[bool, str]:
        try:
            resp = requests.get(
                f"{base_url}/health",
                timeout=self.health_request_timeout_seconds,
            )
            if resp.status_code == 200:
                return True, "ok"
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _cleanup_server_sync(self, server: _OpenEnvServer) -> None:
        sandboxes = SandboxClient(APIClient())
        try:
            sandboxes.unexpose(server.sandbox_id, server.exposure_id)
        except Exception:
            pass
        try:
            sandboxes.delete(server.sandbox_id)
        except Exception:
            pass
        self._active_servers.pop(server.sandbox_id, None)

    async def _fetch_action_schema(self, base_url: str) -> dict[str, Any]:
        if self._action_schema is not None:
            return self._action_schema
        schema = await self._fetch_schema(base_url)
        action_schema = schema.get("action", {}) if isinstance(schema, dict) else {}
        self._action_schema = action_schema
        return action_schema

    def _assert_contract_matches_schema(
        self, contract: str, action_schema: dict[str, Any]
    ) -> None:
        looks_mcp = self._looks_like_mcp_schema(action_schema)
        if contract == "mcp" and not looks_mcp:
            raise RuntimeError(
                "OpenEnv contract mismatch: manifest contract is 'mcp' but action schema "
                "does not match MCP tool action shape."
            )
        if contract == "gym" and looks_mcp:
            raise RuntimeError(
                "OpenEnv contract mismatch: manifest contract is 'gym' but action schema "
                "looks like MCP."
            )

    async def _fetch_schema(self, base_url: str) -> dict[str, Any]:
        def _get() -> dict[str, Any]:
            resp = requests.get(
                f"{base_url}/schema",
                timeout=self.schema_request_timeout_seconds,
            )
            resp.raise_for_status()
            return resp.json()

        async def _run_once() -> dict[str, Any]:
            return await asyncio.to_thread(_get)

        return await self._with_retry(_run_once)()

    def _looks_like_mcp_schema(self, schema: dict[str, Any]) -> bool:
        if not isinstance(schema, dict):
            return False
        props = schema.get("properties", {})
        if isinstance(props, dict) and "tool_name" in props and "arguments" in props:
            return True
        return self._schema_contains_values(schema, {"list_tools", "call_tool"})

    def _schema_contains_values(self, obj: Any, values: set[str]) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "enum" and isinstance(v, list):
                    if any(item in values for item in v):
                        return True
                if self._schema_contains_values(v, values):
                    return True
        elif isinstance(obj, list):
            return any(self._schema_contains_values(v, values) for v in obj)
        return False

    def _parse_action(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        cleaned = self._strip_code_fence(text)
        try:
            action = json.loads(cleaned)
            if isinstance(action, dict):
                return action
        except Exception:
            pass

        single_field = self._single_string_field(schema)
        if single_field:
            return {single_field: cleaned}
        raise ValueError(
            "Failed to parse action JSON. Provide a JSON object matching the action schema."
        )

    def _strip_code_fence(self, text: str) -> str:
        if text.startswith("```") and text.endswith("```"):
            return "\n".join(text.split("\n")[1:-1]).strip()
        return text

    def _single_string_field(self, schema: dict[str, Any]) -> str | None:
        if not isinstance(schema, dict):
            return None
        props = schema.get("properties")
        if not isinstance(props, dict):
            return None
        required = schema.get("required")
        if isinstance(required, list):
            required_str = [name for name in required if isinstance(name, str)]
            if len(required_str) == 1:
                required_name = required_str[0]
                required_spec = props.get(required_name)
                if (
                    isinstance(required_spec, dict)
                    and required_spec.get("type") == "string"
                ):
                    return required_name
        if len(props) == 1:
            field_name, spec = next(iter(props.items()))
            if isinstance(spec, dict) and spec.get("type") == "string":
                return field_name
        return None

    def _normalize_observation(self, obs: Any) -> Any:
        if hasattr(obs, "model_dump"):
            try:
                return obs.model_dump()
            except Exception:
                return obs
        return obs

    def _render_observation_messages(
        self,
        obs: Any,
        *,
        context: str,
        action_schema: dict[str, Any] | None = None,
        contract: str | None = None,
        seed: int | None = None,
    ) -> ChatMessages:
        normalized_obs = self._normalize_observation(obs)
        renderer_kwargs = {
            "context": context,
            "action_schema": action_schema,
            "contract": contract,
            "seed": seed,
        }
        renderer = self.prompt_renderer
        renderer_any = cast(Any, renderer)
        try:
            sig = inspect.signature(renderer)
        except (TypeError, ValueError):
            sig = None

        if sig is None:
            rendered = renderer_any(normalized_obs, **renderer_kwargs)
        else:
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            valid_kwargs: dict[str, Any] = {}
            for key, value in renderer_kwargs.items():
                if has_var_kwargs or key in sig.parameters:
                    valid_kwargs[key] = value
            rendered = renderer_any(normalized_obs, **valid_kwargs)

        if not self._looks_like_messages(rendered):
            raise RuntimeError(
                f"OpenEnv prompt_renderer returned invalid output for {context}: "
                "expected a non-empty chat messages list."
            )
        messages = list(cast(list[dict[str, Any]], rendered))
        if not messages:
            raise RuntimeError(
                f"OpenEnv prompt_renderer returned an empty messages list for {context}."
            )
        for idx, msg in enumerate(messages):
            if msg.get("content") is None:
                raise RuntimeError(
                    "OpenEnv prompt_renderer returned a message with null content "
                    f"for {context} at index {idx}."
                )
        return cast(ChatMessages, messages)

    def _looks_like_messages(self, value: Any) -> bool:
        if not isinstance(value, list):
            return False
        for item in value:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
        return True

    def _require_prompt_messages(self, state: vf.State) -> ChatMessages:
        current_prompt = state.get("prompt")
        if self._looks_like_messages(current_prompt) and cast(
            list[Any], current_prompt
        ):
            return cast(ChatMessages, list(cast(list[Any], current_prompt)))
        raise RuntimeError(
            "OpenEnv dataset must include a non-empty `prompt`. "
            "No prompt fallback is supported."
        )

    def _convert_mcp_tools(self, tools: Iterable[Any]) -> list[dict[str, Any]]:
        oai_tools = []
        for tool in tools:
            tool_dict: dict[str, Any] | None = None
            if hasattr(tool, "model_dump"):
                try:
                    tool_dict = tool.model_dump()
                except Exception:
                    tool_dict = None
            if tool_dict is None:
                if isinstance(tool, dict):
                    tool_dict = tool
                else:
                    tool_dict = {
                        "name": getattr(tool, "name", ""),
                        "description": getattr(tool, "description", ""),
                        "input_schema": getattr(tool, "input_schema", None),
                    }
            oai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_dict.get("name", ""),
                        "description": tool_dict.get("description", ""),
                        "parameters": tool_dict.get("input_schema")
                        or tool_dict.get("inputSchema")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        return oai_tools
