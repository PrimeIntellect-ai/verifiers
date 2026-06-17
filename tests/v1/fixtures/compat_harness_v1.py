"""Harness fixture for v1-as-v0 compatibility tests."""

from __future__ import annotations

import aiohttp

import verifiers.v1 as vf


class CompatHarnessConfig(vf.HarnessConfig):
    id: str = "compat-harness-v1"
    runtime: vf.RuntimeConfig = vf.SubprocessConfig()


class CompatHarness(vf.Harness[CompatHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True

    async def launch(
        self,
        ctx: vf.RolloutContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        del runtime, mcp_urls
        messages = []
        system, instruction = self.resolve_prompt(trace.task)
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": str(instruction)})
        body = {
            "model": ctx.model,
            "messages": messages,
            "temperature": 0,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {secret}"},
            ) as response:
                text = await response.text()
                if response.status >= 400:
                    return vf.ProgramResult(
                        exit_code=1, stdout="", stderr=f"{response.status}: {text}"
                    )
        return vf.ProgramResult(exit_code=0, stdout=text, stderr="")


def load_harness(config: CompatHarnessConfig) -> CompatHarness:
    return CompatHarness(config)
