import json
import shlex
from pathlib import Path
from typing import Literal

from pydantic import PositiveInt, model_validator

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.harnesses.utils.remote_runtime import (
    REMOTE_TOOL_CLIENT_SOURCE,
    prepare_remote_bridge,
)
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import _ENSURE_UV
from verifiers.v1.runtimes.deployment import ExecutionConnection
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
STIRRUP_DIR = "/var/tmp/vf-stirrup"


class StirrupHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    version: PinnedVersion = "c092cee80ce09cd1dfb93aef053296a9ecd9dbb6"
    """Stirrup Git revision to install, pinned for reproducibility."""
    context_window_tokens: PositiveInt = 128_000
    """Model context capacity used by Stirrup's native history summarization."""

    @model_validator(mode="after")
    def reject_disabled_finish(self) -> "StirrupHarnessConfig":
        if "finish" in (self.disabled_tools or []):
            raise ValueError("Stirrup requires its finish tool to terminate the run")
        return self


class StirrupHarness(ACPHarness[StirrupHarnessConfig]):
    HARNESS_TARGET = "stirrup"
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True
    SUPPORTS_TOOL_INTERCEPTION = True

    def _source(self) -> str:
        return PROGRAM_SOURCE.replace("{version}", self.config.version).replace(
            "# REMOTE_TOOL_CLIENT", REMOTE_TOOL_CLIENT_SOURCE
        )

    async def _prepare_runner(
        self, runtime: Runtime, env: dict[str, str], *, label: str
    ) -> list[str]:
        source = self._source()
        if getattr(runtime, "type", self.config.id) not in ("docker", "prime"):
            return await runtime.prepare_uv_script(source, env, activate=False)
        ver_q = shlex.quote(self.config.version)
        dir_q = shlex.quote(STIRRUP_DIR)
        script_path = f"{STIRRUP_DIR}/program-{self.config.version}.py"
        install = (
            f"set -e; {_ENSURE_UV}; "
            f"UV_PYTHON_INSTALL_DIR={dir_q}/python uv venv --allow-existing --python 3.13 {dir_q}/venv -q; "
            f"uv pip install --python {dir_q}/venv/bin/python -q "
            f"'stirrup[mcp] @ https://github.com/ArtificialAnalysis/Stirrup/archive/'{ver_q}'.tar.gz' "
            "'mcp==1.30.0' 'agent-client-protocol==1.0.0'; "
            f"printf '%s' {ver_q} > {dir_q}/.version"
        )
        cache = getattr(getattr(runtime, "backend", None), "cache", None)
        if cache is not None:
            await ensure_installed(
                runtime,
                directory=STIRRUP_DIR,
                ready=(
                    f"[ -x {dir_q}/venv/bin/python ] && "
                    f'[ "$(cat {dir_q}/.version 2>/dev/null)" = {ver_q} ]'
                ),
                install=install,
                env=env,
                label=label,
            )
        script_key = f"stirrup-script:{script_path}"
        root_rt = runtime.with_user(None) if hasattr(runtime, "with_user") else runtime
        if cache is None or script_key not in cache.installed:
            await root_rt.write(script_path, source.encode())
            await root_rt.run(["chmod", "644", script_path], {})
            if cache is not None:
                cache.installed.add(script_key)
        return [f"{STIRRUP_DIR}/venv/bin/python", "-P", "-u", script_path]

    async def setup_execution(
        self, runtime: Runtime, connection: ExecutionConnection
    ) -> None:
        await self._prepare_runner(connection.runtime, {}, label="Stirrup worker")

    async def setup(self, runtime: Runtime) -> None:
        await self._prepare_runner(runtime, self.config.resolved_env, label="Stirrup")
        await super().setup(runtime)

    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ACPConfig:
        system_prompt, prompt = self.resolve_text_prompt(data)
        state_dir = f"/tmp/vf-stirrup-{trace.id}"
        skills_dir = f"{state_dir}/skills"
        system_prompt_args = []
        if system_prompt:
            # Kept until cleanup, not deleted on read, so a restarted bridge finds it.
            path = f"{state_dir}/system-prompt.txt"
            await runtime.write(path, system_prompt.encode())
            system_prompt_args.append(f"--system-prompt-file={path}")
        program = await self._prepare_runner(
            runtime, self.config.resolved_env, label="Stirrup"
        )
        if runtime._execution_bridge is None:
            raise HarnessError("Stirrup requires an execution bridge")
        bridge = runtime._execution_bridge
        cwd = bridge.runtime.info.workdir
        created = await runtime.run(["mkdir", "-p", "--", cwd], {})
        if created.exit_code:
            raise HarnessError(
                "could not prepare the Stirrup supervisor working directory"
            )
        await self.install_skills(runtime, skills_dir, mirror_to=bridge.runtime)
        worker_program = await self._prepare_runner(
            bridge.runtime, {}, label="Stirrup worker"
        )
        worker = f"stirrup-{trace.id}"
        bridge.register(
            worker,
            [
                *worker_program,
                "--execution-worker",
                f"--skills-dir={skills_dir}",
                "--disabled-tools=" + json.dumps(self.config.disabled_tools or []),
            ],
            {},
        )
        root = endpoint.removesuffix("/v1").rstrip("/")
        command = await prepare_remote_bridge(
            runtime, f"{root}/execution", secret, worker
        )
        remote_args = [
            "--execution-command=" + json.dumps(command),
            f"--tool-interception-url={root}/tool",
        ]
        return ACPConfig(
            env=self.config.resolved_env,
            cwd=cwd,
            command=[
                *program,
                f"--base-url={endpoint}",
                f"--api-key={secret}",
                f"--model={ctx.model}",
                f"--context-window-tokens={self.config.context_window_tokens}",
                f"--max-tokens={ctx.sampling.max_tokens or 8192}",
                f"--skills-dir={skills_dir}",
                f"--tool-timeout={self.config.tool_timeout}",
                "--disabled-tools=" + json.dumps(self.config.disabled_tools or []),
                *remote_args,
                *system_prompt_args,
            ],
            prompt=prompt,
            # The bridge owns system-prompt placement; the ACP runner must not
            # also fold it into the first user message.
        )
