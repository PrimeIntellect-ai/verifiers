from __future__ import annotations

import importlib
import importlib.resources as resources
import json
import shlex
import tarfile
import tempfile
from importlib.abc import Traversable
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

    async def post_rollout(self, state: State) -> None:
        """Collect RLM session metrics after the rollout."""
        await super().post_rollout(state)

        sandbox_id = state.get("sandbox_id")
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

    async def _after_harness_inputs_uploaded(self, state: State) -> None:
        sandbox_id = state["sandbox_id"]
        skills_dir = self._discover_taskset_skills_dir()
        if skills_dir is not None:
            await self._upload_skills_dir(sandbox_id, skills_dir)

    def _get_install_execute_kwargs(self) -> dict[str, Any]:
        execute_kwargs = super()._get_install_execute_kwargs()
        if self.install_env:
            execute_kwargs["env"] = self.install_env
        return execute_kwargs

    def _discover_taskset_skills_dir(self) -> Traversable | Path | None:
        module = importlib.import_module(self.taskset.__class__.__module__)

        package_name = self._module_package_name(module)
        if package_name:
            try:
                candidate = resources.files(package_name) / "skills"
                if candidate.is_dir() and any(candidate.iterdir()):
                    return candidate
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

    async def _upload_skills_dir(
        self, sandbox_id: str, skills_dir: Traversable | Path
    ) -> None:
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

    def _build_skills_archive(self, skills_dir: Traversable | Path) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)
        with tarfile.open(tar_path, "w:gz") as tar:
            self._add_skills_dir_to_archive(tar, skills_dir)
        return tar_path

    def _add_skills_dir_to_archive(
        self,
        tar: tarfile.TarFile,
        skills_dir: Traversable | Path,
    ) -> None:
        if isinstance(skills_dir, Path):
            tar.add(skills_dir, arcname=self.uploaded_skills_dir.lstrip("/"))
            return

        with resources.as_file(skills_dir) as local_path:
            tar.add(local_path, arcname=self.uploaded_skills_dir.lstrip("/"))
