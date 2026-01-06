from pathlib import Path

import verifiers as vf
from verifiers.envs.sandbox_env import SandboxEnv


class RemoteEnv(SandboxEnv):
    def __init__(
        self,
        sandbox_path: Path | str,
        upload_path: str = "/app",
        docker_image: str = "python:3.11-slim",
        **kwargs,
    ):
        self.sandbox_path = Path(sandbox_path)
        self.upload_path = upload_path

        super().__init__(
            docker_image=docker_image,
            start_command="tail -f /dev/null",
            **kwargs,
        )

    async def _upload_sandbox_files(self, sandbox_id: str) -> None:
        if not self.sandbox_path.exists():
            raise FileNotFoundError(f"Sandbox path not found: {self.sandbox_path}")

        for file_path in self.sandbox_path.rglob("*"):
            if file_path.is_file():
                if any(
                    part in file_path.parts
                    for part in ["node_modules", "__pycache__", ".git", "dist", "build"]
                ):
                    continue

                relative_path = file_path.relative_to(self.sandbox_path)
                remote_path = f"{self.upload_path}/{relative_path}"

                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                await self.sandbox_client.upload_bytes(
                    sandbox_id,
                    remote_path,
                    file_bytes,
                    file_path.name,
                )

    async def _run_setup(self, sandbox_id: str) -> None:
        await self.sandbox_client.execute_command(
            sandbox_id,
            f"chmod +x {self.upload_path}/setup.sh",
            timeout=10,
        )

        await self.sandbox_client.start_background_job(
            sandbox_id,
            f"{self.upload_path}/setup.sh",
            working_dir=self.upload_path,
        )

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        sandbox_id = state["sandbox_id"]

        await self._wait_for_sandbox_ready(state["sandbox_state"], sandbox_id)
        await self._upload_sandbox_files(sandbox_id)
        await self._run_setup(sandbox_id)

        return state
