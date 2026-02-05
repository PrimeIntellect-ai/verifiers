from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, cast, overload

import requests
import tenacity as tc
from datasets import Dataset

import verifiers as vf
from verifiers.types import ChatMessage, ChatMessages
from verifiers.utils.tool_utils import is_valid_tool_content_parts

try:
    from openenv.core.generic_client import GenericEnvClient
    from openenv.core.mcp_client import MCPToolClient
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
except ImportError as e:
    raise ImportError(
        "OpenEnvEnv requires openenv-core. Install with: uv add 'verifiers[openenv]'"
    ) from e

try:
    from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
except ImportError as e:
    raise ImportError(
        "OpenEnvEnv requires prime-sandboxes. Install with: uv add prime-sandboxes"
    ) from e

yaml: Any | None
try:
    import yaml as _yaml  # type: ignore
except ImportError:
    yaml = None
else:
    yaml = _yaml

logger = logging.getLogger(__name__)


@dataclass
class _OpenEnvServer:
    sandbox_id: str
    exposure_id: str
    base_url: str
    port: int
    background_job: Any | None = None
    needs_manual_start: bool = False
    temp_dir: Path | None = None


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
    - Uses seeds as the generic dataset mechanism.
    - Supports both simulation (step/reset) and MCP tool environments.
    """

    def __init__(
        self,
        openenv_project: str | Path,
        num_train_examples: int = 1000,
        num_eval_examples: int = 100,
        seed: int = 0,
        max_turns: int = -1,
        rubric: vf.Rubric | None = None,
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

        self._active_servers: dict[str, _OpenEnvServer] = {}
        self._mode: str | None = None  # "sim" or "mcp"
        self._action_schema: dict[str, Any] | None = None
        self._mcp_tools: list[Any] | None = None

        dataset, eval_dataset = self._build_seed_datasets()

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric or OpenEnvEpisodicSumRubric(),
            max_turns=max_turns,
            message_type="chat",
            **kwargs,
        )
        self._with_retry = tc.AsyncRetrying(
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(
                initial=base_delay,
                exp_base=backoff_factor,
                max=max_backoff_seconds,
                jitter=jitter,
            ),
            before_sleep=tc.before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        ).wraps

    def _build_seed_datasets(self) -> tuple[Dataset, Dataset | None]:
        def make_rows(start_idx: int, count: int) -> list[dict[str, Any]]:
            rows = []
            for i in range(count):
                seed = self.seed + start_idx + i
                rows.append(
                    {
                        "question": f"OpenEnv episode seed={seed}",
                        "info": {"seed": seed},
                    }
                )
            return rows

        train_rows = make_rows(0, self.num_train_examples)
        eval_rows = make_rows(self.num_train_examples, self.num_eval_examples)

        dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
        return dataset, eval_dataset

    async def setup_state(self, state: vf.State) -> vf.State:
        try:
            server = await self._create_server()
            state["openenv_server"] = server
            mode, action_schema = await self._ensure_mode_and_schema(server.base_url)
            state["openenv_mode"] = mode
            state["openenv_action_schema"] = action_schema
            if self._mode is None:
                self._mode = mode
            if self._action_schema is None:
                self._action_schema = action_schema

            seed = 0
            info = state.get("info")
            if isinstance(info, dict):
                seed = int(info.get("seed", 0))

            if self._mode == "mcp":
                mcp_client = MCPToolClient(base_url=server.base_url)
                openenv_mcp = cast(Any, mcp_client)
                await openenv_mcp.connect()
                state["openenv_mcp_client"] = mcp_client
                if self._mcp_tools is None:
                    self._mcp_tools = await openenv_mcp.list_tools()
                state["oai_tools"] = self._convert_mcp_tools(self._mcp_tools)
                result = await openenv_mcp.reset(seed=seed)
                state["openenv_done"] = bool(result.done)
                obs_messages = self._obs_to_messages(result.observation)
                state["prompt"] = self._maybe_prepend_system(obs_messages)
                return state

            client = GenericEnvClient(base_url=server.base_url)
            openenv_client = cast(Any, client)
            await openenv_client.connect()
            state["openenv_client"] = client
            result = await openenv_client.reset(seed=seed)
            state["openenv_done"] = bool(result.done)
            obs_messages = self._obs_to_messages(result.observation)
            state["prompt"] = self._maybe_prepend_system(obs_messages)
            return state
        except Exception:
            await self._cleanup_openenv_state(state)
            raise

    def _make_user_message(self, content: str) -> ChatMessage:
        return cast(ChatMessage, {"role": "user", "content": content})

    def _make_system_message(self, content: str) -> ChatMessage:
        return cast(ChatMessage, {"role": "system", "content": content})

    def _make_tool_message(self, content: Any, tool_call_id: str) -> ChatMessage:
        return cast(
            ChatMessage,
            {"role": "tool", "content": content, "tool_call_id": tool_call_id},
        )

    def _maybe_prepend_system(self, messages: ChatMessages) -> ChatMessages:
        if self.system_prompt:
            if messages and messages[0].get("role") == "system":
                return messages
            return [self._make_system_message(self.system_prompt)] + messages
        return messages

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        mode = state.get("openenv_mode") or self._mode
        if mode == "mcp":
            return await self._mcp_env_response(messages, state)
        return await self._sim_env_response(messages, state)

    async def _sim_env_response(
        self, messages: vf.Messages, state: vf.State
    ) -> vf.Messages:
        assert isinstance(messages, list)
        last_msg = messages[-1]
        if last_msg.get("role") != "assistant":
            return [self._make_user_message("Expected assistant response.")]

        raw_text = str(last_msg.get("content", "")).strip()
        action_schema = state.get("openenv_action_schema") or self._action_schema or {}
        action = self._parse_action(raw_text, action_schema)

        client: GenericEnvClient = state["openenv_client"]
        result = await client.step(action)

        if state["trajectory"]:
            state["trajectory"][-1]["reward"] = result.reward

        state["openenv_done"] = bool(result.done)
        obs_messages = self._obs_to_messages(result.observation)
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

        mcp_client: MCPToolClient = state["openenv_mcp_client"]
        tool_messages: ChatMessages = []
        total_reward = 0.0
        done = False
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id", "")
            tool_name = tool_call.get("function", {}).get("name", "")
            try:
                tool_args = json.loads(
                    tool_call.get("function", {}).get("arguments", "{}")
                )
                if not isinstance(tool_args, dict):
                    raise ValueError("tool arguments must be an object")
                step_result = await mcp_client.step(
                    CallToolAction(tool_name=tool_name, arguments=tool_args)
                )
                obs = step_result.observation
                if isinstance(obs, CallToolObservation) and obs.error is not None:
                    content = f"Error: {obs.error.message}"
                else:
                    result_payload = obs
                    if isinstance(obs, CallToolObservation):
                        result_payload = obs.result
                    elif hasattr(obs, "result"):
                        result_payload = getattr(obs, "result")
                    if hasattr(result_payload, "model_dump"):
                        result_payload = result_payload.model_dump()
                    content = self._format_tool_content(result_payload)
                if step_result.reward is not None:
                    total_reward += float(step_result.reward)
                done = done or bool(step_result.done)
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
        mode = state.get("openenv_mode") or self._mode
        return bool(state.get("openenv_done")) and mode == "sim"

    @vf.stop
    async def mcp_no_tool_calls(self, state: vf.State) -> bool:
        mode = state.get("openenv_mode") or self._mode
        if mode != "mcp":
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
            await cast(Any, client).close()

        mcp_client = state.pop("openenv_mcp_client", None)
        if mcp_client is not None:
            await cast(Any, mcp_client).close()

        server = state.pop("openenv_server", None)
        if server is not None:
            await self._cleanup_server(server)

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
        if server.temp_dir is not None:
            shutil.rmtree(server.temp_dir, ignore_errors=True)

    async def _try_get_logs(
        self, sandboxes: AsyncSandboxClient, sandbox_id: str
    ) -> str | None:
        try:
            logs = await sandboxes.get_logs(sandbox_id)
        except Exception:
            return None
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
        project_path, source_type, temp_dir = self._resolve_project_path()
        try:
            if source_type == "hf":
                repo_id = cast(str, project_path)
                image = f"registry.hf.space/{repo_id.replace('/', '-')}:latest"
                server = await self._launch_image_server(image, 8000)
            else:
                assert isinstance(project_path, Path)
                dockerfile = self._find_dockerfile(project_path)
                if dockerfile is not None:
                    image = self._read_image_marker(project_path)
                    if image is None:
                        raise RuntimeError(
                            "OpenEnv project contains a Dockerfile but no .openenv_image marker. "
                            "Run: vf-openenv-build --path <openenv_project> to build and register the image."
                        )
                    port = self._read_project_port(project_path)
                    server = await self._launch_image_server(image, port)
                else:
                    port = self._read_project_port(project_path)
                    server = await self._launch_source_server(project_path, port)
        except Exception:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        server.temp_dir = temp_dir
        self._active_servers[server.sandbox_id] = server
        return server

    @overload
    def _resolve_project_path(self) -> tuple[Path, Literal["local"], Path | None]: ...

    @overload
    def _resolve_project_path(self) -> tuple[str, Literal["hf"], None]: ...

    def _resolve_project_path(self) -> tuple[str | Path, str, Path | None]:
        path = Path(self.openenv_project)
        if path.exists():
            return path.resolve(), "local", None
        if "://" in self.openenv_project or self.openenv_project.endswith(".git"):
            tmpdir = Path(tempfile.mkdtemp(prefix="openenv_git_"))
            cmd = ["git", "clone", self.openenv_project, str(tmpdir)]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except FileNotFoundError as e:
                raise RuntimeError("git is required to clone OpenEnv projects.") from e
            except subprocess.CalledProcessError as e:
                details = (e.stderr or e.stdout or "").strip()
                suffix = f" ({details})" if details else ""
                raise RuntimeError(
                    f"Failed to clone OpenEnv project: {self.openenv_project}{suffix}"
                ) from e
            return tmpdir.resolve(), "local", tmpdir
        if "/" in self.openenv_project:
            return self.openenv_project, "hf", None
        raise ValueError(f"Unsupported openenv_project: {self.openenv_project}")

    def _find_dockerfile(self, project_path: Path) -> Path | None:
        dockerfile = project_path / "server" / "Dockerfile"
        if dockerfile.exists():
            return dockerfile
        fallback = project_path / "Dockerfile"
        if fallback.exists():
            return fallback
        return None

    def _read_image_marker(self, project_path: Path) -> str | None:
        marker = project_path / ".openenv_image"
        if not marker.exists():
            return None
        return marker.read_text().strip()

    def _read_project_port(self, project_path: Path) -> int:
        openenv_yaml = project_path / "openenv.yaml"
        if not openenv_yaml.exists() or yaml is None:
            return 8000
        try:
            data = yaml.safe_load(openenv_yaml.read_text())
        except Exception:
            return 8000
        if isinstance(data, dict) and "port" in data:
            try:
                return int(data["port"])
            except Exception:
                return 8000
        return 8000

    async def _launch_image_server(self, image: str, port: int) -> _OpenEnvServer:
        async with AsyncSandboxClient() as sandboxes:
            req = self._build_sandbox_request(image, start_command=None)
            try:
                sandbox = await self._with_retry(sandboxes.create)(req)
            except Exception as e:
                raise vf.SandboxError(
                    f"Failed to create OpenEnv sandbox for image {image}."
                ) from e
            exposure = None
            try:
                await self._with_retry(sandboxes.wait_for_creation)(sandbox.id)
                exposure = await self._with_retry(sandboxes.expose)(
                    sandbox.id, port=port, name="openenv-env"
                )
                server = _OpenEnvServer(
                    sandbox_id=sandbox.id,
                    exposure_id=exposure.exposure_id,
                    base_url=exposure.url.rstrip("/"),
                    port=port,
                    needs_manual_start=False,
                )
                await self._wait_for_ready(server.base_url)
                return server
            except Exception as e:
                logs = await self._try_get_logs(sandboxes, sandbox.id)
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

    async def _launch_source_server(
        self, project_path: Path, port: int
    ) -> _OpenEnvServer:
        async with AsyncSandboxClient() as sandboxes:
            req = self._build_sandbox_request(
                "python:3.11-slim", start_command="tail -f /dev/null"
            )
            try:
                sandbox = await self._with_retry(sandboxes.create)(req)
            except Exception as e:
                raise vf.SandboxError("Failed to create OpenEnv sandbox.") from e
            exposure = None
            try:
                await self._with_retry(sandboxes.wait_for_creation)(sandbox.id)

                await self._upload_project(sandboxes, sandbox.id, project_path)
                await self._with_retry(sandboxes.execute_command)(
                    sandbox.id,
                    "python -m pip install -e /workspace/openenv_project",
                    working_dir="/workspace/openenv_project",
                )
                await self._with_retry(sandboxes.start_background_job)(
                    sandbox.id,
                    "bash -lc 'cd /workspace/openenv_project && server'",
                )

                exposure = await self._with_retry(sandboxes.expose)(
                    sandbox.id, port=port, name="openenv-env"
                )
                server = _OpenEnvServer(
                    sandbox_id=sandbox.id,
                    exposure_id=exposure.exposure_id,
                    base_url=exposure.url.rstrip("/"),
                    port=port,
                    needs_manual_start=True,
                )
                await self._wait_for_ready(server.base_url)
                return server
            except Exception as e:
                logs = await self._try_get_logs(sandboxes, sandbox.id)
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
                    sandbox.id, "startup", e, logs=logs
                ) from e

    def _build_sandbox_request(
        self, image: str, start_command: str | None
    ) -> CreateSandboxRequest:
        params: dict[str, Any] = {
            "name": "openenv-env",
            "docker_image": image,
            "cpu_cores": 2,
            "memory_gb": 4,
            "disk_size_gb": 10,
            "timeout_minutes": 60,
            "environment_vars": {"ENABLE_WEB_INTERFACE": "false"},
        }
        if start_command is not None:
            params["start_command"] = start_command
        try:
            return CreateSandboxRequest(**cast(Any, params))
        except TypeError:
            if "start_command" not in params:
                params["start_command"] = "tail -f /dev/null"
                return CreateSandboxRequest(**cast(Any, params))
            raise

    async def _upload_project(
        self, sandboxes: AsyncSandboxClient, sandbox_id: str, project_path: Path
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(project_path, arcname="openenv_project")
            remote_tar = "/tmp/openenv_project.tar.gz"
            await self._with_retry(sandboxes.upload_file)(
                sandbox_id, remote_tar, str(tar_path)
            )
            await self._with_retry(sandboxes.execute_command)(
                sandbox_id,
                "mkdir -p /workspace && tar -xzf /tmp/openenv_project.tar.gz -C /workspace",
                working_dir="/",
            )
        finally:
            tar_path.unlink(missing_ok=True)

    async def _wait_for_ready(self, base_url: str, timeout_s: int = 120) -> None:
        def _check() -> bool:
            try:
                resp = requests.get(f"{base_url}/health", timeout=2)
                return resp.status_code == 200
            except Exception:
                return False

        loop = asyncio.get_running_loop()
        start = loop.time()
        while (loop.time() - start) < timeout_s:
            ok = await asyncio.to_thread(_check)
            if ok:
                return
            await asyncio.sleep(1)
        raise RuntimeError(f"OpenEnv server not ready after {timeout_s}s: {base_url}")

    async def _ensure_mode_and_schema(
        self, base_url: str
    ) -> tuple[str, dict[str, Any]]:
        if self._mode is not None and self._action_schema is not None:
            return self._mode, self._action_schema

        schema = await self._fetch_schema(base_url)
        action_schema = schema.get("action", {}) if isinstance(schema, dict) else {}
        mode = "mcp" if self._looks_like_mcp_schema(action_schema) else "sim"
        self._mode = mode
        self._action_schema = action_schema
        return mode, action_schema

    async def _fetch_schema(self, base_url: str) -> dict[str, Any]:
        def _get() -> dict[str, Any]:
            resp = requests.get(f"{base_url}/schema", timeout=5)
            resp.raise_for_status()
            return resp.json()

        return await asyncio.to_thread(_get)

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
            return {single_field: text}
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
        if len(props) != 1:
            return None
        field_name, spec = next(iter(props.items()))
        if isinstance(spec, dict) and spec.get("type") == "string":
            return field_name
        return None

    def _obs_to_messages(self, obs: Any) -> ChatMessages:
        if hasattr(obs, "model_dump"):
            try:
                obs = obs.model_dump()
            except Exception:
                pass
        if isinstance(obs, dict):
            if "messages" in obs and self._looks_like_messages(obs["messages"]):
                return cast(ChatMessages, list(obs["messages"]))
            for key in ("prompt", "content", "message"):
                if key in obs and isinstance(obs[key], str):
                    return [self._make_user_message(obs[key])]
            metadata = obs.get("metadata")
            if isinstance(metadata, dict):
                for key in ("prompt", "content", "message"):
                    if key in metadata and isinstance(metadata[key], str):
                        return [self._make_user_message(metadata[key])]
            return [self._make_user_message(json.dumps(obs, ensure_ascii=True))]
        return [self._make_user_message(str(obs))]

    def _looks_like_messages(self, value: Any) -> bool:
        if not isinstance(value, list):
            return False
        for item in value:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
        return True

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
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        return oai_tools
