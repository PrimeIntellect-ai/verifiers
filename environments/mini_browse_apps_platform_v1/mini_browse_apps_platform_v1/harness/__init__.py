"""Browser-app harness: stages a privately-distributed browser agent into the sandbox.

The browser agent is proprietary and is NOT vendored in this repo. It is fetched at run time
from a private, auth-gated GitHub repo (pinned to a commit), cached locally, then tarred and
staged into the sandbox, where `program.py` (a uv script) imports and runs it. For local
development, point `agent.path` at a checkout instead of fetching.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field
from pydantic_config import BaseConfig
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
AGENT_CACHE_DIR = Path.home() / ".cache" / "verifiers" / "browse-agent"
MODEL_CLIENT_PATH = "/tmp/vf-browse-model-client.json"

CoordinateMode = Literal["relative_1000", "absolute", "auto"]


class AgentConfig(BaseConfig):
    """The proprietary browser agent — fetched at run time from a private repo, not vendored."""

    repo: str = "PrimeIntellect-ai/plex-mini-browse"
    """Private GitHub repo (owner/name) the agent is fetched from."""
    ref: str = ""
    """Pinned commit sha to fetch (required unless `path` is set)."""
    package: str = "mini_browse"
    """Importable package dir within the repo (and on the sandbox PYTHONPATH) to stage."""
    token_env: str = "MINI_BROWSE_GITHUB_TOKEN"
    """Env var holding a GitHub token with read access to `repo`."""
    path: str | None = None
    """Local dir containing `<package>/` — when set, skips the GitHub fetch (development)."""
    cache_dir: str | None = None
    """Where fetched revisions are cached (default: ~/.cache/verifiers/browse-agent)."""


class MiniBrowseHarnessConfig(HarnessConfig):
    """Reusable browser harness; fetches its proprietary agent from a private repo."""

    id: str = "mini-browse-apps-platform-v1"
    runtime: RuntimeConfig = DockerConfig(image="python:3.12-slim-bookworm")
    agent: AgentConfig = Field(default_factory=AgentConfig)
    max_steps: int = 75
    coordinate_mode: CoordinateMode = "relative_1000"
    keep_last_images: int = 3
    image_compaction_at_tokens: int = 45_000
    include_builtin_tools: bool = False
    browser_start_min_interval_seconds: float = 0.0
    browser_start_jitter_seconds: float = 0.0
    browser_start_max_in_flight: int = 0
    record_frames: bool = False


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
        await runtime.write(
            MODEL_CLIENT_PATH,
            json.dumps(
                {"base_url": endpoint, "api_key": secret, "model": ctx.model}
            ).encode("utf-8"),
        )
        env = {
            **self.config.env,
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
            "MINI_BROWSE_PROGRESS_PATH": PROGRESS_PATH,
        }
        if self.config.record_frames:
            env["MINI_BROWSE_RECORD_FRAMES_DIR"] = "/logs/mini_browse/frames"

        args = [
            "--task",
            TASK_PAYLOAD_PATH,
            "--model-client",
            MODEL_CLIENT_PATH,
            "--result",
            RESULT_PATH,
            "--transcript",
            TRANSCRIPT_PATH,
            "--metrics",
            METRICS_PATH,
            "--progress",
            PROGRESS_PATH,
            "--max-steps",
            str(self.config.max_steps),
            "--workspace-root",
            WORKSPACE_ROOT,
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
        package = self._ensure_agent() / self.config.agent.package
        if not package.is_dir():
            raise HarnessError(
                f"agent package {self.config.agent.package!r} not found under {package.parent}"
            )
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(package, arcname=self.config.agent.package)
        return buffer.getvalue()

    def _ensure_agent(self) -> Path:
        """Return a dir that contains `<package>/` — a local checkout or the fetch cache."""
        agent = self.config.agent
        if agent.path:
            return Path(agent.path).expanduser()
        if not agent.ref:
            raise HarnessError(
                "set --harness.agent.ref to a pinned commit sha "
                "(or --harness.agent.path to a local checkout for development)"
            )
        cache_root = (
            Path(agent.cache_dir).expanduser() if agent.cache_dir else AGENT_CACHE_DIR
        )
        dest = cache_root / agent.ref
        if not (dest / agent.package).exists():
            self._download_agent(dest)
        return dest

    def _download_agent(self, dest: Path) -> None:
        agent = self.config.agent
        token = os.environ.get(agent.token_env)
        if not token:
            raise HarnessError(
                f"missing ${agent.token_env} to fetch the private agent repo {agent.repo!r}"
            )
        url = f"https://api.github.com/repos/{agent.repo}/tarball/{agent.ref}"
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
                        f"fetching {agent.repo}@{agent.ref} failed: HTTP {resp.status_code}"
                    )
                with open(archive, "wb") as handle:
                    for chunk in resp.iter_bytes():
                        handle.write(chunk)
            extract = Path(tmp) / "extract"
            extract.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(extract, filter="data")
            matches = sorted(extract.glob(f"*/{agent.package}"))
            if not matches:
                raise HarnessError(
                    f"{agent.package!r} not found in {agent.repo}@{agent.ref}"
                )
            dest.mkdir(parents=True, exist_ok=True)
            staging = dest / (agent.package + ".tmp")
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(matches[0], staging)
            os.replace(staging, dest / agent.package)

    def _pythonpath(self) -> str:
        existing = self.config.env.get("PYTHONPATH", "")
        entries = [AGENT_RUNTIME]
        if existing:
            entries.append(existing)
        return ":".join(entries)


def load_harness(config: MiniBrowseHarnessConfig) -> MiniBrowseHarness:
    return MiniBrowseHarness(config)


__all__ = [
    "AgentConfig",
    "MiniBrowseHarness",
    "MiniBrowseHarnessConfig",
    "MiniBrowseTaskPayload",
    "read_jsonl_tail",
]
