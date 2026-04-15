from __future__ import annotations

import importlib
import importlib.resources as resources
import json
import shlex
import tarfile
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import verifiers as vf
from verifiers.envs.experimental.composable.composable_env import ComposableEnv
from verifiers.types import State

_RLM_METRIC_KEYS = [
    "turns",
    "stop_reason",
    "prompt_tokens",
    "completion_tokens",
    "tool_result_tokens",
    "prompt_tokens_per_turn",
    "completion_tokens_per_turn",
    "summarize_count",
    "summarize_rejected_count",
    "summarize_total_turns_dropped",
    "turns_between_summarizes",
    "sub_rlm_prompt_tokens",
    "sub_rlm_completion_tokens",
    "sub_rlm_count",
    "max_turns",
    "max_tokens",
]


class RlmComposableEnv(ComposableEnv):
    """ComposableEnv with RLM-specific skill upload and metrics extraction."""

    def __init__(
        self,
        *args: Any,
        install_env: dict[str, str] | None = None,
        uploaded_skills_dir: str = "/task/rlm-skills",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.install_env = dict(install_env) if install_env else None
        self.uploaded_skills_dir = uploaded_skills_dir

    async def post_sandbox_setup(self, state: State) -> None:
        """Task setup → upload task assets → install RLM agent."""
        sandbox_id = state["sandbox_id"]

        state["sandbox_client"] = self.sandbox_client
        spec = self._get_spec(state)
        if spec:
            state["test_timeout"] = spec.timeout_minutes * 60
        elif self.harness.sandbox_spec:
            state["test_timeout"] = self.harness.sandbox_spec.timeout_minutes * 60
        else:
            state["test_timeout"] = 900

        await self.taskset.setup(state)

        dirs = {self.harness.instruction_path.rsplit("/", 1)[0]}
        if self.harness.system_prompt:
            dirs.add(self.harness.system_prompt_path.rsplit("/", 1)[0])
        mkdir_args = " ".join(shlex.quote(path) for path in sorted(dirs))
        await self.sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {mkdir_args}", timeout=10
        )

        info = state.get("info") or {}
        instruction = self.taskset.get_instruction(info)
        if instruction.strip():
            await self.upload_content(
                sandbox_id, instruction, self.harness.instruction_path
            )

        if self.harness.system_prompt:
            await self.upload_content(
                sandbox_id, self.harness.system_prompt, self.harness.system_prompt_path
            )

        skills_dir = self._discover_taskset_skills_dir()
        if skills_dir is not None:
            await self._upload_skills_dir(sandbox_id, skills_dir)

        if self.harness.install_script:
            self.logger.debug(f"Installing agent in sandbox {sandbox_id}")
            execute_kwargs: dict[str, Any] = {"timeout": 300}
            if self.install_env:
                execute_kwargs["env"] = self.install_env
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                self.harness.install_script,
                **execute_kwargs,
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"Agent install failed (exit={result.exit_code}): {output[:500]}"
                )

    async def post_rollout(self, state: State) -> None:
        """Collect agent logs and RLM session metrics after the rollout."""
        sandbox_id = state.get("sandbox_id")
        if sandbox_id and self.harness.log_path and "agent_logs" not in state:
            try:
                log_path = shlex.quote(self.harness.log_path)
                result = await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"cat {log_path} 2>/dev/null || echo '<no logs>'",
                    working_dir=None,
                )
                state["agent_logs"] = (result.stdout or "").strip()
            except Exception as e:
                self.logger.warning(f"Failed to collect agent logs: {e}")

        if sandbox_id:
            info = state.get("info") or {}
            workdir = self.taskset.get_workdir(info)
            try:
                quoted_workdir = shlex.quote(workdir)
                result = await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"f=$(ls {quoted_workdir}/.rlm/sessions/*/meta.json 2>/dev/null | head -1) "
                    '&& cat "$f" || echo "{}"',
                    working_dir=None,
                )
                meta = json.loads((result.stdout or "{}").strip())
                metrics = meta.get("metrics", {})
                for key in _RLM_METRIC_KEYS:
                    if key in metrics:
                        state[f"rlm_{key}"] = metrics[key]
            except Exception as e:
                self.logger.warning(f"Failed to read rlm session metrics: {e}")

        await super().post_rollout(state)

    def _discover_taskset_skills_dir(self) -> Path | None:
        module = importlib.import_module(self.taskset.__class__.__module__)

        package_name = self._module_package_name(module)
        if package_name:
            try:
                candidate = resources.files(package_name) / "skills"
                if candidate.is_dir():
                    with resources.as_file(candidate) as local_path:
                        local_dir = Path(local_path)
                        if local_dir.is_dir() and any(local_dir.iterdir()):
                            return local_dir
            except Exception:
                pass

        module_file = getattr(module, "__file__", None)
        if module_file:
            candidate = Path(module_file).resolve().parent / "skills"
            if candidate.is_dir() and any(candidate.iterdir()):
                return candidate
        return None

    @staticmethod
    def _module_package_name(module: ModuleType) -> str | None:
        if hasattr(module, "__path__"):
            return module.__name__
        package_name = getattr(module, "__package__", None)
        return package_name or None

    async def _upload_skills_dir(self, sandbox_id: str, skills_dir: Path) -> None:
        remote_tar = "/tmp/rlm-skills.tar.gz"
        tmp_path = self._build_skills_archive(skills_dir)
        try:
            await self.upload_file(sandbox_id, remote_tar, str(tmp_path))
            dest_parent = shlex.quote(str(Path(self.uploaded_skills_dir).parent))
            quoted_remote_tar = shlex.quote(remote_tar)
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p {dest_parent} && "
                f"tar -xzf {quoted_remote_tar} -C / && "
                f"rm -f {quoted_remote_tar}",
                timeout=60,
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"RLM skills extract failed (exit={result.exit_code}): {output[:500]}"
                )
        finally:
            tmp_path.unlink(missing_ok=True)

    def _build_skills_archive(self, skills_dir: Path) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(skills_dir, arcname=self.uploaded_skills_dir.lstrip("/"))
        return tar_path
