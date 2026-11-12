import json
from pathlib import Path
from typing import Literal

from pydantic import PositiveInt, model_validator

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig, PinnedVersion
from verifiers.v1.harnesses.utils.install import remove_dir
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


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
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_SKILLS = True

    def _source(self) -> str:
        return PROGRAM_SOURCE.replace("{version}", self.config.version)

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(self._source(), self.config.resolved_env)
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
    ) -> ACPConfig:
        system_prompt, prompt = self.resolve_text_prompt(data)
        state_dir = runtime.temp_path(
            f"vf-stirrup-{trace.id}", env=self.config.resolved_env
        )
        skills_dir = f"{state_dir}/skills"
        await self.install_skills(runtime, skills_dir)
        system_prompt_args = []
        if system_prompt:
            # Kept until cleanup, not deleted on read, so a restarted bridge finds it.
            path = f"{state_dir}/system-prompt.txt"
            await runtime.write(path, system_prompt.encode())
            system_prompt_args.append(f"--system-prompt-file={path}")
        program = await runtime.prepare_uv_script(
            self._source(), self.config.resolved_env, activate=False
        )
        return ACPConfig(
            env=self.config.resolved_env,
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
                *system_prompt_args,
            ],
            prompt=prompt,
            # The bridge owns system-prompt placement; the ACP runner must not
            # also fold it into the first user message.
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        await remove_dir(
            runtime,
            runtime.temp_path(f"vf-stirrup-{trace.id}", env=self.config.resolved_env),
            "Stirrup skills",
        )
