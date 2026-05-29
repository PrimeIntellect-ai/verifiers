import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

import requests
import tenacity as tc
from openenv.core.containers.runtime.providers import ContainerProvider
from verifiers.v1.types import ConfigData

if TYPE_CHECKING:
    from prime_sandboxes import SandboxClient as PrimeSandboxClient
    from prime_sandboxes.models import ExposedPort, Sandbox

    from tasksets.openenv import OpenEnvRuntimeConfig


class PrimeSandboxOpenEnvProvider(ContainerProvider):
    def __init__(self, spec: "OpenEnvRuntimeConfig"):
        self.spec = spec
        self.sandbox_id: str | None = None
        self.exposure_id: str | None = None
        self._base_url: str | None = None
        self._client: "PrimeSandboxClient | None" = None

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("OpenEnv sandbox has not started.")
        return self._base_url

    def start_container(
        self,
        image: str,
        port: int | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: object,
    ) -> str:
        from prime_sandboxes import APIClient, CreateSandboxRequest, SandboxClient

        api_client = APIClient()
        client = SandboxClient(api_client)
        self._client = client
        start_command = kwargs.get("start_command")
        if not isinstance(start_command, str) or not start_command:
            start_command = self.spec.start_command
        container_port = port or self.spec.port
        request = CreateSandboxRequest(
            name="openenv-env",
            docker_image=image,
            start_command=start_command,
            cpu_cores=2,
            memory_gb=4,
            disk_size_gb=10,
            timeout_minutes=60,
            environment_vars={"ENABLE_WEB_INTERFACE": "false", **dict(env_vars or {})},
        )
        sandbox_id: str | None = None
        exposure_id: str | None = None
        try:
            sandbox = self._retry(client.create, request)
            sandbox_value = cast("Sandbox", sandbox)
            sandbox_id = sandbox_value.id
            self.sandbox_id = sandbox_id
            client.wait_for_creation(
                sandbox_id,
                max_attempts=self.spec.wait_for_creation_max_attempts,
            )
            exposure = client.expose(
                sandbox_id,
                port=container_port,
                name="openenv-env",
                protocol="TCP",
            )
            exposure_value = cast("ExposedPort", exposure)
            exposure_id = exposure_value.exposure_id
            self.exposure_id = exposure_id
            endpoint = exposure_value.external_endpoint
            if isinstance(endpoint, str) and endpoint.strip():
                self._base_url = f"http://{endpoint.strip()}"
            else:
                raw_url = str(exposure_value.url or "").strip()
                if raw_url.startswith("tcp://"):
                    host_port = raw_url[len("tcp://") :].rstrip("/")
                    if host_port:
                        self._base_url = f"http://{host_port}"
                elif raw_url.startswith(("http://", "https://")):
                    self._base_url = raw_url.rstrip("/")
            if self._base_url is None:
                raise RuntimeError(
                    "OpenEnv sandbox exposure did not provide a usable URL."
                )
            return self._base_url
        except Exception as exc:
            details = (
                self._failure_details(client, sandbox_id, container_port)
                if sandbox_id is not None
                else None
            )
            if sandbox_id is not None:
                if exposure_id is not None:
                    try:
                        client.unexpose(sandbox_id, exposure_id)
                    except Exception:
                        pass
                try:
                    client.delete(sandbox_id)
                except Exception:
                    pass
            message = f"OpenEnv sandbox failed during startup for image {image}."
            if details:
                message = f"{message}\n{details}"
            raise RuntimeError(message) from exc

    def stop_container(self) -> None:
        client = self._client
        if client is None:
            return
        if self.sandbox_id is not None and self.exposure_id is not None:
            try:
                client.unexpose(self.sandbox_id, self.exposure_id)
            except Exception:
                pass
        if self.sandbox_id is not None:
            try:
                client.delete(self.sandbox_id)
            except Exception:
                pass
        self.sandbox_id = None
        self.exposure_id = None
        self._base_url = None
        self._client = None

    def wait_for_ready(self, base_url: str, timeout_s: float = 30.0) -> None:
        del timeout_s
        start_time = time.monotonic()
        last_error = "no attempts"
        while time.monotonic() - start_time < self.spec.startup_timeout_seconds:
            try:
                response = requests.get(
                    f"{base_url}/health",
                    timeout=self.spec.health_request_timeout_seconds,
                )
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(self.spec.startup_poll_interval_seconds)
        raise RuntimeError(
            "OpenEnv server not ready. "
            f"Health timeout={self.spec.startup_timeout_seconds}s, "
            f"url={base_url}, last error: {last_error}"
        )

    def fetch_schema(self) -> ConfigData:
        def request_schema() -> ConfigData:
            response = requests.get(
                f"{self.base_url}/schema",
                timeout=self.spec.schema_request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, Mapping):
                raise TypeError("OpenEnv /schema must return a JSON object.")
            return {str(key): value for key, value in data.items()}

        return cast(ConfigData, self._retry(request_schema))

    def _retry(self, fn: Callable[..., object], *args: object) -> object:
        retrying = tc.Retrying(
            stop=tc.stop_after_attempt(self.spec.max_retries),
            wait=tc.wait_exponential_jitter(
                initial=self.spec.base_delay,
                exp_base=self.spec.backoff_factor,
                max=self.spec.max_backoff_seconds,
                jitter=self.spec.jitter,
            ),
            reraise=True,
        )
        return retrying(fn, *args)

    def _failure_details(
        self, client: "PrimeSandboxClient", sandbox_id: str, port: int | None
    ) -> str | None:
        details: list[str] = []
        try:
            logs = str(client.get_logs(sandbox_id) or "")
        except Exception:
            logs = ""
        if logs:
            details.append(f"Logs tail:\n{logs[-4000:]}")
        if port is not None:
            try:
                result = client.execute_command(
                    sandbox_id,
                    f'sh -lc "curl -sS -m 2 http://localhost:{int(port)}/health 2>&1 || true"',
                    timeout=5,
                )
            except Exception as exc:
                details.append(
                    f"Local /health probe failed: {type(exc).__name__}: {exc}"
                )
            else:
                stdout = str(result.stdout or "").strip()
                stderr = str(result.stderr or "").strip()
                if stdout:
                    details.append(f"Local /health probe stdout: {stdout}")
                if stderr:
                    details.append(f"Local /health probe stderr: {stderr}")
                if not stdout and not stderr:
                    details.append("Local /health probe returned no output.")
        return "\n".join(details) or None
