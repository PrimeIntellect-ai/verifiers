import shlex

import httpx

import verifiers as vf
from verifiers.envs.sandbox_env import SandboxEnv

DEFAULT_API_URL = "https://api.primeintellect.ai"


class RemoteEnv(SandboxEnv):
    def __init__(
        self,
        environment: str,
        upload_path: str = "/app",
        docker_image: str = "python:3.11-slim",
        api_base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        Remote environment that downloads files from the Prime Environments Hub.

        Args:
            environment: Environment identifier in format "owner/name" or "owner/name@version"
            upload_path: Path inside sandbox where files are extracted (default: /app)
            docker_image: Docker image for sandbox (default: python:3.11-slim)
            api_base_url: Base URL for Prime API (default: https://api.primeintellect.ai)
            api_key: API key for authentication (optional, needed for private environments)
            **kwargs: Additional arguments passed to SandboxEnv
        """
        self.environment = environment
        self.upload_path = upload_path
        self.api_base_url = (api_base_url or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key
        self._package_url: str | None = None

        if "@" in environment:
            env_id, self.version = environment.rsplit("@", 1)
        else:
            env_id = environment
            self.version = "latest"

        parts = env_id.split("/")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid environment format: {environment}. Expected: owner/name or owner/name@version"
            )
        self.owner, self.name = parts

        super().__init__(
            docker_image=docker_image,
            start_command="tail -f /dev/null",
            **kwargs,
        )

    async def _fetch_package_url(self) -> str:
        """Fetch the package URL from the environments hub."""
        if self._package_url:
            return self._package_url

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_base_url}/environmentshub/{self.owner}/{self.name}/@{self.version}",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        details = data.get("data", data)
        package_url = details.get("package_url")

        if not package_url:
            raise ValueError(f"No package URL found for environment {self.environment}")

        self._package_url = package_url
        return package_url

    async def _download_and_extract(self, sandbox_id: str) -> None:
        """Download tarball from hub and extract to sandbox."""
        package_url = await self._fetch_package_url()

        download_script = f"""
import urllib.request
import tarfile
import os

os.makedirs("{self.upload_path}", exist_ok=True)
urllib.request.urlretrieve("{package_url}", "/tmp/env.tar.gz")
with tarfile.open("/tmp/env.tar.gz", "r:gz") as tar:
    tar.extractall("{self.upload_path}")
os.remove("/tmp/env.tar.gz")
print("Download and extraction complete")
"""

        result = await self.sandbox_client.execute_command(
            sandbox_id,
            f"python3 -c {shlex.quote(download_script)}",
            timeout=120,
        )

        if result.exit_code != 0:
            raise RuntimeError(f"Failed to download environment: {result.stderr}")

    async def _run_setup(self, sandbox_id: str) -> None:
        """Run setup.sh from the sandbox directory."""
        sandbox_dir = f"{self.upload_path}/sandbox"

        await self.sandbox_client.execute_command(
            sandbox_id,
            f"chmod +x {sandbox_dir}/setup.sh",
            timeout=10,
        )

        await self.sandbox_client.start_background_job(
            sandbox_id,
            f"{sandbox_dir}/setup.sh",
            working_dir=sandbox_dir,
        )

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        sandbox_id = state["sandbox_id"]

        await self._wait_for_sandbox_ready(state["sandbox_state"], sandbox_id)
        await self._download_and_extract(sandbox_id)
        await self._run_setup(sandbox_id)

        return state
