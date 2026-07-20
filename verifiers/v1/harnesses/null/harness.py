import json
from pathlib import Path

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class NullHarnessConfig(HarnessConfig):
    pass


class NullHarness(Harness[NullHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            "--payload-stdin",
        ]
        payload = {
            "system_prompt": system_prompt,
            "prompt": prompt if isinstance(prompt, str) else None,
            "initial_messages": (
                [message_to_wire(message) for message in prompt]
                if prompt is not None and not isinstance(prompt, str)
                else []
            ),
            "mcp_config": {
                "mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}
            },
        }
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        return await runtime.run_program(
            [*program, *args], env, stdin=json.dumps(payload).encode("utf-8")
        )
