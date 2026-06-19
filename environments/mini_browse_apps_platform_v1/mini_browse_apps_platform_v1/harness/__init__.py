"""Browser-app harness: stages a privately-distributed browser agent into the sandbox.

The browser agent is proprietary and is NOT vendored in this repo. It is fetched at run time
from a private, auth-gated GitHub repo (pinned to a commit), cached locally, then tarred and
staged into the sandbox, where `program.py` (a uv script) imports and runs it. For local
development, point `agent_path` at a checkout instead of fetching.
"""

from __future__ import annotations

import io
import os
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Literal

import httpx
from verifiers.v1.clients import RolloutContext
from verifiers.v1.errors import HarnessError
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import DockerConfig, ProgramResult, Runtime, RuntimeConfig
from verifiers.v1.trace import Trace

from .contract import (
    METRICS_PATH,
    MiniBrowseTaskPayload,
    PROGRESS_PATH,
    RESULT_PATH,
    TASK_PAYLOAD_PATH,
    TRANSCRIPT_PATH,
    WORKSPACE_ROOT,
)
from .diagnostics import read_jsonl_tail

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

AGENT_RUNTIME = "/opt/browse-agent-runtime"
AGENT_TARBALL = "/tmp/vf-browse-agent-runtime.tgz"

CoordinateMode = Literal["relative_1000", "absolute", "auto"]


class MiniBrowseHarnessConfig(HarnessConfig):
    """Reusable browser harness; fetches its proprietary agent from a private repo."""

    id: str = "mini-browse-apps-platform-v1"
    runtime: RuntimeConfig = DockerConfig(image="python:3.12-slim-bookworm")
    # --- proprietary agent source (not vendored; fetched at run time) ---
    agent_repo: str = "PrimeIntellect-ai/plex-mini-browse"
    """Private GitHub repo (owner/name) the agent is fetched from."""
    agent_ref: str = ""
    """Pinned commit sha to fetch (required unless `agent_path` is set)."""
    agent_package: str = "mini_browse"
    """Importable package dir within the repo (and on the sandbox PYTHONPATH) to stage."""
    agent_token_env: str = "MINI_BROWSE_GITHUB_TOKEN"
    """Env var holding a GitHub token with read access to `agent_repo`."""
    agent_path: str | None = None
    """Local dir containing `<agent_package>/` — when set, skips the GitHub fetch (dev)."""
    agent_cache_dir: str | None = None
    """Where fetched agent revisions are cached (default: ~/.cache/verifiers/browse-agent)."""
    # --- agent behavior ---
    max_steps: int = 75
    coordinate_mode: CoordinateMode = "relative_1000"
    keep_last_images: int = 3
    image_compaction_at_tokens: int = 45_000
    include_builtin_tools: bool = False
    browser_start_min_interval_seconds: float = 0.0
    browser_start_jitter_seconds: float = 0.0
    browser_start_max_in_flight: int = 0
    record_frames: bool = False
    task_payload_path: str = TASK_PAYLOAD_PATH
    result_path: str = RESULT_PATH
    transcript_path: str = TRANSCRIPT_PATH
    metrics_path: str = METRICS_PATH
    progress_path: str = PROGRESS_PATH
    workspace_root: str = WORKSPACE_ROOT


class MiniBrowseHarness(Harness[MiniBrowseHarnessConfig]):
    """Stages the privately-fetched browser agent and executes its agent loop."""

    SUPPORTS_TASK_TOOLS = False
    SUPPORTS_MESSAGE_PROMPT = False

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if mcp_urls:
            names = ", ".join(sorted(mcp_urls))
            raise ValueError(
                f"Browser harness does not expose v1 MCP task tools: {names}"
            )
        if trace.task.system_prompt:
            raise ValueError(
                "Browser harness owns the system prompt; put task-specific instructions "
                "in task.prompt or the task payload."
            )
        if not isinstance(trace.task.prompt, str):
            raise ValueError("Browser harness requires a string task prompt")

        await self._stage_agent(runtime)
        env = {
            **self.config.env,
            "OPENAI_BASE_URL": endpoint,
            "OPENAI_API_KEY": secret,
            "OPENAI_MODEL": ctx.model,
            "PYTHONPATH": self._pythonpath(),
            "MINI_BROWSE_COORDINATE_MODE": self.config.coordinate_mode,
            "MINI_BROWSE_KEEP_LAST_IMAGES": str(self.config.keep_last_images),
            "MINI_BROWSE_IMAGE_COMPACTION_AT_TOKENS": str(
                self.config.image_compaction_at_tokens
            ),
            "MINI_BROWSE_INCLUDE_BUILTIN_TOOLS": (
                "1" if self.config.include_builtin_tools else "0"
            ),
            "MINI_BROWSE_BROWSER_START_MIN_INTERVAL_SECONDS": str(
                self.config.browser_start_min_interval_seconds
            ),
            "MINI_BROWSE_BROWSER_START_JITTER_SECONDS": str(
                self.config.browser_start_jitter_seconds
            ),
            "MINI_BROWSE_BROWSER_START_MAX_IN_FLIGHT": str(
                self.config.browser_start_max_in_flight
            ),
            "MINI_BROWSE_PROGRESS_PATH": self.config.progress_path,
        }
        if self.config.record_frames:
            env["MINI_BROWSE_RECORD_FRAMES_DIR"] = "/logs/mini_browse/frames"

        args = [
            "--task",
            self.config.task_payload_path,
            "--result",
            self.config.result_path,
            "--transcript",
            self.config.transcript_path,
            "--metrics",
            self.config.metrics_path,
            "--progress",
            self.config.progress_path,
            "--max-steps",
            str(self.config.max_steps),
            "--workspace-root",
            self.config.workspace_root,
        ]
        return await runtime.run_uv_script(PROGRAM_SOURCE, args=args, env=env)

    async def _stage_agent(self, runtime: Runtime) -> None:
        await runtime.write(AGENT_TARBALL, self._agent_tarball())
        command = (
            f"rm -rf {shlex.quote(AGENT_RUNTIME)} && "
            f"mkdir -p {shlex.quote(AGENT_RUNTIME)} && "
            f"tar -xzf {shlex.quote(AGENT_TARBALL)} -C {shlex.quote(AGENT_RUNTIME)}"
        )
        result = await runtime.run(["sh", "-c", command], {})
        if result.exit_code != 0:
            raise HarnessError(
                f"agent staging failed: {result.stderr.strip()[-500:]}"
            )

    def _agent_tarball(self) -> bytes:
        package = self._ensure_agent() / self.config.agent_package
        if not package.is_dir():
            raise HarnessError(
                f"agent package {self.config.agent_package!r} not found under {package.parent}"
            )
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(package, arcname=self.config.agent_package)
        return buffer.getvalue()

    def _ensure_agent(self) -> Path:
        """Return a dir that contains `<agent_package>/` — a local checkout or the fetch cache."""
        if self.config.agent_path:
            return Path(self.config.agent_path).expanduser()
        if not self.config.agent_ref:
            raise HarnessError(
                "set --harness.agent-ref to a pinned commit sha "
                "(or --harness.agent-path to a local checkout for development)"
            )
        cache_root = (
            Path(self.config.agent_cache_dir).expanduser()
            if self.config.agent_cache_dir
            else Path.home() / ".cache" / "verifiers" / "browse-agent"
        )
        dest = cache_root / self.config.agent_ref
        if not (dest / self.config.agent_package).exists():
            self._download_agent(dest)
        return dest

    def _download_agent(self, dest: Path) -> None:
        token = os.environ.get(self.config.agent_token_env)
        if not token:
            raise HarnessError(
                f"missing ${self.config.agent_token_env} to fetch the private agent repo "
                f"{self.config.agent_repo!r}"
            )
        url = f"https://api.github.com/repos/{self.config.agent_repo}/tarball/{self.config.agent_ref}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        with tempfile.TemporaryDirectory(prefix="browse-agent-") as tmp:
            archive = Path(tmp) / "agent.tar.gz"
            # httpx drops the Authorization header on the cross-host redirect to codeload.
            with httpx.stream(
                "GET", url, headers=headers, follow_redirects=True, timeout=120
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise HarnessError(
                        f"fetching {self.config.agent_repo}@{self.config.agent_ref} failed: "
                        f"HTTP {resp.status_code}"
                    )
                with open(archive, "wb") as handle:
                    for chunk in resp.iter_bytes():
                        handle.write(chunk)
            extract = Path(tmp) / "extract"
            extract.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(extract, filter="data")
            matches = sorted(extract.glob(f"*/{self.config.agent_package}"))
            if not matches:
                raise HarnessError(
                    f"{self.config.agent_package!r} not found in "
                    f"{self.config.agent_repo}@{self.config.agent_ref}"
                )
            dest.mkdir(parents=True, exist_ok=True)
            staging = dest / (self.config.agent_package + ".tmp")
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(matches[0], staging)
            os.replace(staging, dest / self.config.agent_package)

    def _pythonpath(self) -> str:
        existing = self.config.env.get("PYTHONPATH", "")
        entries = [AGENT_RUNTIME]
        if existing:
            entries.append(existing)
        return ":".join(entries)


def load_harness(config: MiniBrowseHarnessConfig) -> MiniBrowseHarness:
    return MiniBrowseHarness(config)


__all__ = [
    "MiniBrowseHarness",
    "MiniBrowseHarnessConfig",
    "MiniBrowseTaskPayload",
    "read_jsonl_tail",
]
