from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import aiohttp
from datasets import Dataset

import verifiers as vf
from verifiers.types import AssistantMessage, Messages, State, ToolMessage
from verifiers.utils.message_utils import concat_messages, normalize_messages

try:
    from prime_sandboxes import AsyncSandboxClient, CreateSandboxRequest
except ImportError as e:
    raise ImportError(
        "NemoGymEnv requires prime-sandboxes. Install with: uv add prime-sandboxes"
    ) from e

_ALLOWED_DATASET_SPLITS = {"example", "train", "validation"}
_SERVER_LOG_PATH = "/tmp/nemo_gym_resource_server.log"
_DEFAULT_NEMO_PACKAGE = "nemo-gym @ https://test-files.pythonhosted.org/packages/c0/58/451a826009a0b206c932e1ebde3dcff2a8b31152c77133fdde7e5f7ccd90/nemo_gym-0.2.9892rc0-py3-none-any.whl"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return _json_dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _sanitize_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, raw_child in value.items():
            if raw_child is None:
                continue
            child = _sanitize_json_schema(raw_child)
            if child is None:
                continue
            sanitized[key] = child

        properties = sanitized.get("properties")
        if isinstance(properties, dict):
            sanitized["properties"] = {
                name: schema
                for name, schema in properties.items()
                if isinstance(schema, (dict, bool))
            }
            required = sanitized.get("required")
            if isinstance(required, list):
                allowed = set(sanitized["properties"].keys())
                sanitized["required"] = [
                    name
                    for name in required
                    if isinstance(name, str) and name in allowed
                ]

        return sanitized

    if isinstance(value, list):
        return [
            child
            for item in value
            if (child := _sanitize_json_schema(item)) is not None
        ]

    return value


def _normalize_parameters_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "object", "properties": {}}
    return _sanitize_json_schema(value)


def _nemo_tools_to_tool_defs(raw_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []

    tool_defs: list[dict[str, Any]] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue

        # OpenAI Chat Completions-style tool schema.
        if raw_tool.get("type") == "function" and isinstance(
            raw_tool.get("function"), dict
        ):
            fn = cast(dict[str, Any], raw_tool["function"])
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            tool_def: dict[str, Any] = {
                "name": name,
                "description": _stringify(fn.get("description", "")),
                "parameters": _normalize_parameters_schema(fn.get("parameters")),
            }
            strict = fn.get("strict", raw_tool.get("strict"))
            if isinstance(strict, bool):
                tool_def["strict"] = strict
            tool_defs.append(tool_def)
            continue

        # OpenAI Responses API function tool schema.
        tool_type = raw_tool.get("type")
        if tool_type not in (None, "function"):
            continue

        name = raw_tool.get("name")
        if not isinstance(name, str) or not name:
            continue

        tool_def = {
            "name": name,
            "description": _stringify(raw_tool.get("description", "")),
            "parameters": _normalize_parameters_schema(raw_tool.get("parameters")),
        }
        strict = raw_tool.get("strict")
        if isinstance(strict, bool):
            tool_def["strict"] = strict
        tool_defs.append(tool_def)

    return tool_defs


def _resolve_resources_servers_root() -> Path:
    resources_spec = importlib.util.find_spec("resources_servers")
    if resources_spec and resources_spec.submodule_search_locations:
        root = Path(next(iter(resources_spec.submodule_search_locations))).resolve()
        if root.exists():
            return root

    nemo_spec = importlib.util.find_spec("nemo_gym")
    if nemo_spec and nemo_spec.origin:
        nemo_root = Path(nemo_spec.origin).resolve().parent
        sibling = nemo_root.parent / "resources_servers"
        if sibling.exists():
            return sibling

    raise RuntimeError(
        "Unable to locate NeMo Gym resources_servers package. "
        "Install `nemo-gym` or pass `dataset_path` explicitly."
    )


def _resolve_dataset_path(
    resource_server: str,
    dataset_split: str,
    dataset_path: str | None,
) -> Path:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {path}")
        return path

    resources_root = _resolve_resources_servers_root()
    path = resources_root / resource_server / "data" / f"{dataset_split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            "Could not find dataset file for server "
            f"'{resource_server}' split '{dataset_split}': {path}"
        )
    return path


def _load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Row {line_no} in {path} is not an object")
            if "responses_create_params" not in row:
                raise ValueError(
                    f"Row {line_no} in {path} is missing required key "
                    "'responses_create_params'"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset file {path} contains no rows")
    return rows


def _build_dataset(
    resource_server: str,
    dataset_split: str,
    dataset_path: str | None = None,
    dataset_limit: int | None = None,
) -> tuple[Dataset, Path]:
    resolved_path = _resolve_dataset_path(resource_server, dataset_split, dataset_path)
    rows = _load_rows_from_jsonl(resolved_path)

    if dataset_limit is not None:
        if dataset_limit <= 0:
            raise ValueError("dataset_limit must be > 0 when provided")
        rows = rows[:dataset_limit]

    dataset_rows: list[dict[str, Any]] = []
    for row in rows:
        responses_create_params = row.get("responses_create_params")
        if not isinstance(responses_create_params, dict):
            raise ValueError("responses_create_params must be an object")

        raw_input = responses_create_params.get("input", [])
        if isinstance(raw_input, str):
            prompt = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            prompt = raw_input
        else:
            prompt = [{"role": "user", "content": _stringify(raw_input)}]
        dataset_rows.append(
            {
                "prompt": prompt,
                "answer": _stringify(row.get("answer", "")),
                "task": resource_server,
                "info": {
                    "dataset_row_json": _json_dumps(row),
                    "resource_server": resource_server,
                },
            }
        )

    return Dataset.from_list(dataset_rows), resolved_path


def _completion_to_nemo_response(
    completion: Messages,
    model_name: str,
    trajectory_id: str,
    responses_create_params: dict[str, Any],
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    message_idx = 0

    for msg in completion:
        if isinstance(msg, AssistantMessage):
            text = msg.content or ""
            if isinstance(text, list):
                text = "\n".join(getattr(p, "text", str(p)) for p in text)
            if text:
                output.append(
                    {
                        "id": f"msg_{message_idx}",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []}
                        ],
                    }
                )
                message_idx += 1

            for tc in msg.tool_calls or []:
                output.append(
                    {
                        "id": tc.id,
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )
                message_idx += 1

        elif isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, list):
                content = "\n".join(getattr(p, "text", str(p)) for p in content)
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": content or "",
                }
            )

    return {
        "id": f"verifiers-{trajectory_id}",
        "created_at": int(time.time()),
        "model": model_name,
        "object": "response",
        "output": output,
        "parallel_tool_calls": bool(
            responses_create_params.get("parallel_tool_calls", False)
        ),
        "tool_choice": responses_create_params.get("tool_choice", "none"),
        "tools": responses_create_params.get("tools", []),
    }


def _reward_from_verify(state: State, **kwargs: Any) -> float:
    verify_response = state.get("verify_response")
    if not isinstance(verify_response, dict):
        return 0.0
    return float(verify_response.get("reward", 0.0) or 0.0)


@dataclass
class _NemoGymServer:
    sandbox_id: str
    exposure_id: str
    base_url: str
    openapi_paths: set[str]


_DEFAULT_SANDBOX_OPTIONS = {
    "docker_image": "python:3.12",
    "cpu_cores": 2,
    "memory_gb": 4,
    "disk_size_gb": 10,
    "timeout_minutes": 60,
    "port": 8000,
    "server_start_timeout_s": 120,
    "http_timeout_s": 60,
}


class NemoGymEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        *,
        resource_server: str,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_turns: int = 16,
        config_overrides: dict[str, Any] | None = None,
        extra_pip_packages: list[str] | None = None,
        nemo_package: str = _DEFAULT_NEMO_PACKAGE,
        nemo_package_version: str | None = None,
        seed_session_on_start: bool = True,
        system_prompt: str | None = None,
        sandbox_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.resource_server = resource_server
        self.nemo_package = nemo_package
        self.nemo_package_version = nemo_package_version
        self.extra_pip_packages = list(extra_pip_packages or [])
        self.config_overrides = dict(config_overrides or {})
        self.seed_session_on_start = seed_session_on_start

        opts = {**_DEFAULT_SANDBOX_OPTIONS, **(sandbox_options or {})}
        self.docker_image: str = opts["docker_image"]
        self.sandbox_cpu_cores: int = opts["cpu_cores"]
        self.sandbox_memory_gb: int = opts["memory_gb"]
        self.sandbox_disk_size_gb: int = opts["disk_size_gb"]
        self.sandbox_timeout_minutes: int = opts["timeout_minutes"]
        self.sandbox_port: int = opts["port"]
        self.server_start_timeout_s: int = opts["server_start_timeout_s"]
        self.http_timeout_s: int = opts["http_timeout_s"]

        self._http_session: aiohttp.ClientSession | None = None
        self._server: _NemoGymServer | None = None
        self._sandbox_lock: asyncio.Lock | None = None

        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=max_turns,
            system_prompt=system_prompt,
            message_type="chat",
            **kwargs,
        )

    def _nemo_package_spec(self) -> str:
        if self.nemo_package_version:
            return f"{self.nemo_package}=={self.nemo_package_version}"
        return self.nemo_package

    def _install_command(self) -> str:
        package_specs = [self._nemo_package_spec(), "httpx", *self.extra_pip_packages]
        quoted_specs = " ".join(shlex.quote(spec) for spec in package_specs)

        env_chunks: list[str] = []
        for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
            value = os.getenv(key)
            if value:
                env_chunks.append(f"{key}={shlex.quote(value)}")

        env_prefix = " ".join(env_chunks)
        command = f"python -m pip install --no-cache-dir {quoted_specs}"
        return f"{env_prefix} {command}".strip()

    def _server_launcher_script(self) -> str:
        serialized_overrides = _json_dumps(self.config_overrides)
        return f"""
import importlib
import inspect
import json
import sys

import uvicorn
from omegaconf import OmegaConf

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.server_utils import BaseServerConfig, ServerClient

RESOURCE_SERVER = {self.resource_server!r}
PORT = {self.sandbox_port}
SERVER_CONFIG_OVERRIDES = json.loads({serialized_overrides!r})

# Add the server's package directory to sys.path and PYTHONPATH so relative
# imports (e.g. "from lcb_integration import ...") resolve correctly.
# PYTHONPATH is needed so Ray workers also inherit the path.
import os
_rs_pkg = importlib.import_module("resources_servers")
for _search_path in getattr(_rs_pkg, "__path__", []):
    _server_dir = _search_path + "/" + RESOURCE_SERVER
    if _server_dir not in sys.path:
        sys.path.insert(0, _server_dir)
    _pypath = os.environ.get("PYTHONPATH", "")
    if _server_dir not in _pypath:
        os.environ["PYTHONPATH"] = _server_dir + (":" + _pypath if _pypath else "")

module = importlib.import_module(f"resources_servers.{{RESOURCE_SERVER}}.app")
server_cls = None
for obj in module.__dict__.values():
    if (
        inspect.isclass(obj)
        and issubclass(obj, SimpleResourcesServer)
        and obj is not SimpleResourcesServer
        and obj.__module__ == module.__name__
    ):
        server_cls = obj
        break

if server_cls is None:
    raise RuntimeError(
        f"Could not locate SimpleResourcesServer subclass in {{module.__name__}}"
    )

config_cls = server_cls.model_fields["config"].annotation
config_payload = {{
    "name": RESOURCE_SERVER,
    "entrypoint": "app.py",
    "host": "0.0.0.0",
    "port": PORT,
    "domain": "other",
}}
config_payload.update(SERVER_CONFIG_OVERRIDES)

config = config_cls(**config_payload)

server_client = ServerClient(
    head_server_config=BaseServerConfig(host="127.0.0.1", port=11000),
    global_config_dict=OmegaConf.create({{}}),
)
server = server_cls(config=config, server_client=server_client)
app = server.setup_webserver()
server.setup_exception_middleware(app)

uvicorn.run(
    app,
    host="0.0.0.0",
    port=PORT,
    timeout_graceful_shutdown=0.5,
    log_level="info",
)
""".strip()

    def _start_server_command(self) -> str:
        launcher_path = "/tmp/nemo_gym_server_launcher.py"
        launcher_script = self._server_launcher_script()
        return (
            f"cat > {launcher_path} <<'PY'\n"
            f"{launcher_script}\n"
            "PY\n"
            f"nohup python {launcher_path} > {_SERVER_LOG_PATH} 2>&1 &"
        )

    def _build_sandbox_request(self) -> CreateSandboxRequest:
        params: dict[str, Any] = {
            "name": f"nemo-gym-{self.resource_server}",
            "docker_image": self.docker_image,
            "start_command": "tail -f /dev/null",
            "cpu_cores": self.sandbox_cpu_cores,
            "memory_gb": self.sandbox_memory_gb,
            "disk_size_gb": self.sandbox_disk_size_gb,
            "timeout_minutes": self.sandbox_timeout_minutes,
            "environment_vars": {"ENABLE_WEB_INTERFACE": "false"},
        }
        return CreateSandboxRequest(**cast(Any, params))

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

        raise RuntimeError("NeMo Gym sandbox exposure did not provide a usable URL.")

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            timeout = aiohttp.ClientTimeout(total=float(self.http_timeout_s))
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        return self._http_session

    async def _ensure_server(self) -> _NemoGymServer:
        if self._server is not None:
            return self._server
        if self._sandbox_lock is None:
            self._sandbox_lock = asyncio.Lock()
        async with self._sandbox_lock:
            if self._server is not None:
                return self._server
            self._server = await self._create_sandbox()
            return self._server

    async def _request(
        self,
        *,
        base_url: str,
        method: str,
        endpoint: str,
        payload: Any | None = None,
        cookie: str | None = None,
    ) -> tuple[int, Any, dict[str, str]]:
        session = await self._ensure_http_session()
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {"cookie": cookie} if cookie else None
        request_kwargs: dict[str, Any] = {}
        if payload is not None:
            request_kwargs["json"] = payload

        async with session.request(
            method,
            url,
            headers=headers,
            **request_kwargs,
        ) as response:
            text = await response.text()
            if not text:
                body: Any = {}
            else:
                try:
                    body = json.loads(text)
                except json.JSONDecodeError:
                    body = text
            return int(response.status), body, dict(response.headers)

    async def _wait_for_server_ready(self, base_url: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        start = loop.time()
        last_error = "no attempts"

        while (loop.time() - start) < float(self.server_start_timeout_s):
            try:
                status, body, _ = await self._request(
                    base_url=base_url,
                    method="GET",
                    endpoint="/openapi.json",
                )
                if status == 200 and isinstance(body, dict):
                    return body
                last_error = f"HTTP {status}: {_stringify(body)[:400]}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            await asyncio.sleep(2)

        raise vf.SandboxError(
            "NeMo Gym server failed to become ready within "
            f"{self.server_start_timeout_s}s at {base_url}. Last error: {last_error}"
        )

    async def _server_log_tail(
        self,
        sandboxes: AsyncSandboxClient,
        sandbox_id: str,
        lines: int = 120,
    ) -> str:
        try:
            result = await sandboxes.execute_command(
                sandbox_id,
                f"tail -n {lines} {_SERVER_LOG_PATH} 2>/dev/null || true",
                timeout=10,
            )
            return (result.stdout or "").strip()
        except Exception:
            return ""

    async def _create_sandbox(self) -> _NemoGymServer:
        await self._ensure_http_session()

        async with AsyncSandboxClient() as sandboxes:
            sandbox: Any | None = None
            exposure: Any | None = None
            try:
                sandbox = await sandboxes.create(self._build_sandbox_request())
                print(
                    f"[NemoGymEnv] Created sandbox {sandbox.id} for '{self.resource_server}'"
                )
                await sandboxes.wait_for_creation(sandbox.id)

                install_result = await sandboxes.execute_command(
                    sandbox.id,
                    self._install_command(),
                    timeout=900,
                )
                if install_result.exit_code != 0:
                    stderr = (install_result.stderr or "").strip()
                    stdout = (install_result.stdout or "").strip()
                    raise vf.SandboxError(
                        "Failed to install NeMo Gym in sandbox. "
                        f"stdout: {stdout[-500:]} stderr: {stderr[-500:]}"
                    )

                start_result = await sandboxes.execute_command(
                    sandbox.id,
                    self._start_server_command(),
                    timeout=30,
                )
                if start_result.exit_code != 0:
                    stderr = (start_result.stderr or "").strip()
                    raise vf.SandboxError(
                        f"Failed to launch NeMo Gym resource server: {stderr[-500:]}"
                    )

                exposure = await sandboxes.expose(
                    sandbox.id,
                    port=self.sandbox_port,
                    name="nemo-gym",
                    protocol="TCP",
                )
                base_url = self._exposure_to_base_url(exposure)

                openapi = await self._wait_for_server_ready(base_url)
                openapi_paths = set((openapi.get("paths") or {}).keys())

                return _NemoGymServer(
                    sandbox_id=sandbox.id,
                    exposure_id=str(getattr(exposure, "exposure_id", "")),
                    base_url=base_url,
                    openapi_paths=openapi_paths,
                )
            except Exception as exc:
                if sandbox is not None:
                    logs = await self._server_log_tail(sandboxes, sandbox.id)
                    if exposure is not None:
                        try:
                            await sandboxes.unexpose(sandbox.id, exposure.exposure_id)
                        except Exception:
                            pass
                    try:
                        await sandboxes.delete(sandbox.id)
                    except Exception:
                        pass
                else:
                    logs = ""

                if isinstance(exc, vf.SandboxError):
                    detail = str(exc)
                else:
                    detail = f"{type(exc).__name__}: {exc}"
                if logs:
                    detail = f"{detail}\nServer log tail:\n{logs}"
                sandbox_id = sandbox.id if sandbox is not None else "N/A"
                raise vf.SandboxError(
                    f"Failed at sandbox startup for NeMo Gym resource server "
                    f"'{self.resource_server}' (sandbox={sandbox_id}): {detail}"
                ) from exc

    def _get_server(self) -> _NemoGymServer:
        if self._server is None:
            raise RuntimeError("No server available — was setup_state() called?")
        return self._server

    async def _seed_session(self, base_url: str, payload: dict[str, Any]) -> str | None:
        status, body, headers = await self._request(
            base_url=base_url,
            method="POST",
            endpoint="/seed_session",
            payload=payload,
        )
        if status >= 400:
            raise vf.SandboxError(
                f"seed_session failed with status {status}: {_stringify(body)[:400]}"
            )

        set_cookie = headers.get("set-cookie") or headers.get("Set-Cookie")
        if isinstance(set_cookie, str):
            cookie = set_cookie.split(";", 1)[0].strip()
            if cookie:
                return cookie
        return None

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        server = await self._ensure_server()

        # dataset_row_json is a JSON string we set in _build_dataset — deserialize
        # it here rather than using info dict fields directly, because HF Arrow
        # serialization corrupts heterogeneous nested schemas.
        dataset_row = json.loads(state["info"]["dataset_row_json"])
        responses_create_params = dataset_row["responses_create_params"]

        tool_defs_raw = _nemo_tools_to_tool_defs(
            responses_create_params.get("tools", [])
        )
        state["tool_defs"] = self._normalize_tool_defs(tool_defs_raw) or []
        state["nemo_dataset_row"] = dataset_row
        state["verify_response"] = None

        seed_payload = {
            k: v for k, v in dataset_row.items() if k != "responses_create_params"
        }
        cookie: str | None = None
        if self.seed_session_on_start and "/seed_session" in server.openapi_paths:
            cookie = await self._seed_session(server.base_url, seed_payload)
        state["nemo_cookie"] = cookie
        return state

    async def env_response(
        self,
        messages: Messages,
        state: State,
        **kwargs: Any,
    ) -> Messages:
        if not messages:
            return []

        last_message = messages[-1]
        if (
            not isinstance(last_message, AssistantMessage)
            or not last_message.tool_calls
        ):
            return []

        server = self._get_server()
        cookie = state.get("nemo_cookie")

        tool_messages: Messages = []
        for tool_call in last_message.tool_calls:
            call_id = tool_call.id
            tool_name = tool_call.name
            endpoint = f"/{tool_name}"

            try:
                parsed_args = json.loads(tool_call.arguments)
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        tool_call_id=call_id,
                        content=_json_dumps(
                            {
                                "error": "Invalid JSON tool arguments",
                                "detail": f"{type(exc).__name__}: {exc}",
                                "arguments": tool_call.arguments,
                            }
                        ),
                    )
                )
                continue

            try:
                status, body, _ = await self._request(
                    base_url=server.base_url,
                    method="POST",
                    endpoint=endpoint,
                    payload=parsed_args,
                    cookie=cookie,
                )
            except Exception as exc:
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        tool_call_id=call_id,
                        content=_json_dumps(
                            {
                                "error": "Tool request failed",
                                "endpoint": endpoint,
                                "detail": f"{type(exc).__name__}: {exc}",
                            }
                        ),
                    )
                )
                continue

            if status >= 400:
                content = _json_dumps(
                    {
                        "error": "Tool endpoint returned non-success status",
                        "endpoint": endpoint,
                        "status_code": status,
                        "body": body,
                    }
                )
            elif isinstance(body, str):
                content = body
            else:
                content = _json_dumps(body)

            tool_messages.append(
                ToolMessage(role="tool", tool_call_id=call_id, content=content)
            )

        return tool_messages

    @vf.stop
    async def no_tool_calls(self, state: State, **kwargs: Any) -> bool:
        trajectory = state.get("trajectory")
        if not trajectory:
            return False
        last_message = trajectory[-1]["completion"][-1]
        return (
            isinstance(last_message, AssistantMessage) and not last_message.tool_calls
        )

    @vf.cleanup
    async def cleanup_nemo(self, state: State) -> None:
        await self._verify(state)

    def _completion_for_verify(self, state: State) -> Messages:
        completion = state.get("completion")
        if isinstance(completion, list):
            return normalize_messages(completion, field_name="state.completion")

        trajectory = state.get("trajectory", [])
        if not isinstance(trajectory, list) or not trajectory:
            return []

        last_step = trajectory[-1]
        last_prompt = normalize_messages(
            last_step["prompt"], field_name="trajectory.prompt"
        )
        last_completion = normalize_messages(
            last_step["completion"],
            field_name="trajectory.completion",
        )
        full_conversation = concat_messages([last_prompt, last_completion])

        final_env_response = state.get("final_env_response")
        if final_env_response is not None:
            final_messages = normalize_messages(
                final_env_response, field_name="final_env_response"
            )
            full_conversation = concat_messages([full_conversation, final_messages])

        prompt_messages = normalize_messages(state["prompt"], field_name="state.prompt")
        return full_conversation[len(prompt_messages) :]

    async def _verify(self, state: State) -> None:
        try:
            dataset_row = state["nemo_dataset_row"]
            responses_create_params = dataset_row["responses_create_params"]

            completion = self._completion_for_verify(state)
            nemo_response = _completion_to_nemo_response(
                completion=completion,
                model_name=str(state.get("model", "")),
                trajectory_id=str(state.get("trajectory_id", "unknown")),
                responses_create_params=responses_create_params,
            )

            verify_payload = {
                "responses_create_params": responses_create_params,
                "response": nemo_response,
                **{
                    k: v
                    for k, v in dataset_row.items()
                    if k != "responses_create_params"
                },
            }

            server = self._get_server()
            cookie = state.get("nemo_cookie")
            status, body, _ = await self._request(
                base_url=server.base_url,
                method="POST",
                endpoint="/verify",
                payload=verify_payload,
                cookie=cookie,
            )

            if status >= 400:
                state["verify_response"] = {
                    "reward": 0.0,
                    "error": f"Verify endpoint returned status {status}: {_stringify(body)[:400]}",
                }
            else:
                state["verify_response"] = body

            if (
                "/close" in server.openapi_paths
                and dataset_row.get("env_id") is not None
            ):
                try:
                    await self._request(
                        base_url=server.base_url,
                        method="POST",
                        endpoint="/close",
                        payload={"env_id": dataset_row["env_id"]},
                        cookie=cookie,
                    )
                except Exception:
                    pass
        except Exception as exc:
            state["verify_response"] = {
                "reward": 0.0,
                "error": f"Verification failed: {type(exc).__name__}: {exc}",
            }

    async def _destroy_sandbox(self, server: _NemoGymServer) -> None:
        if not server.sandbox_id:
            return
        async with AsyncSandboxClient() as sandboxes:
            if server.exposure_id:
                try:
                    await sandboxes.unexpose(server.sandbox_id, server.exposure_id)
                except Exception:
                    pass
            try:
                await sandboxes.delete(server.sandbox_id)
            except Exception:
                pass

    @vf.teardown
    async def teardown_server(self) -> None:
        if self._server is not None:
            await self._destroy_sandbox(self._server)
            self._server = None
        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None
