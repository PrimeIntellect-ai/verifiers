"""Pi harness using the upstream pi-acp adapter."""

import json
import logging
import shlex
from pathlib import Path
from typing import Literal

from pydantic import Field

from verifiers.v1.acp import ACPConfig, ACPHarness
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harnesses.node import (
    NODE_BIN_DIR,
    ensure_node,
    node_patch_id,
    prepare_node_patch,
)
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

KEY_VAR = "PI_INTERCEPT_KEY"

PI_DIR = "/var/tmp/vf-pi"
PACKAGES_DIR = f"{PI_DIR}/mcp"
PI_BIN = f"{PACKAGES_DIR}/node_modules/.bin/pi"
SKILLS_DIR = ".agents/skills"
MCP_VERSION = "2.20.1"
ACP_VERSION = "0.0.33"
MCP_ADAPTER = f"{PACKAGES_DIR}/node_modules/pi-mcp-adapter/index.ts"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/pi-acp"
ACP_COMMAND = [f"{NODE_BIN_DIR}/node", ACP_BIN]
SESSION_IMPORT_PATCH = Path(__file__).with_name("session_import.patch").read_bytes()
SESSION_IMPORT_PATCH_ID = node_patch_id(SESSION_IMPORT_PATCH)

INSTALL = r"""
set -e
packages=/var/tmp/vf-pi/mcp
export PATH="/var/tmp/vf-node/bin:$PATH"

rm -rf "$packages/node_modules/pi-acp"
npm install --prefix "$packages" --ignore-scripts --no-audit --no-fund --omit=dev \
    "@earendil-works/pi-coding-agent@$VF_PI_VERSION" \
    "pi-mcp-adapter@$VF_PI_MCP_VERSION" \
    "pi-acp@$VF_PI_ACP_VERSION" >/dev/null
$VF_PI_ACP_PATCH
"""


class PiHarnessConfig(HarnessConfig):
    version: str = Field(default="0.84.0", pattern=r"^[A-Za-z0-9._+-]+$")
    """Pi release to install, pinned for reproducibility."""
    transport: Literal["chat_completions", "responses", "anthropic_messages"] = (
        "chat_completions"
    )
    """Model API transport."""


class PiHarness(ACPHarness[PiHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    # Pi's project skill discovery is trust-gated (a prompt print mode can't answer),
    # so the installed skills are passed explicitly via `--skill` at launch.
    SUPPORTS_SKILLS = True

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        await ensure_node(runtime)
        logger.info(
            "pi: ensuring Pi %s and pi-acp %s are installed",
            self.config.version,
            ACP_VERSION,
        )
        ready = (
            f"{PI_DIR}/.ready-{self.config.version}-{MCP_VERSION}-{ACP_VERSION}-"
            f"{SESSION_IMPORT_PATCH_ID}"
        )
        patch = await prepare_node_patch(
            runtime,
            PI_DIR,
            f"{PACKAGES_DIR}/node_modules/pi-acp/dist/index.js",
            SESSION_IMPORT_PATCH,
        )
        lock = f"{PI_DIR}/install.lock"
        guarded = (
            f"mkdir -p {PI_DIR} && "
            f'until ln -s "$$" {lock} 2>/dev/null; do '
            f"owner=$(readlink {lock}); "
            f'if ! kill -0 "$owner" 2>/dev/null; then '
            f'[ "$(readlink {lock})" != "$owner" ] || rm -f {lock}; fi; '
            f"sleep 0.1; done; "
            f'trap \'[ "$(readlink {lock})" != "$$" ] || rm -f {lock}\' EXIT; '
            f"[ -f {ready} ] || (sh -c {shlex.quote(INSTALL)} && touch {ready})"
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                "VF_PI_VERSION": self.config.version,
                "VF_PI_MCP_VERSION": MCP_VERSION,
                "VF_PI_ACP_VERSION": ACP_VERSION,
                "VF_PI_ACP_PATCH": patch,
            },
        )
        if install.exit_code != 0:
            raise RuntimeError(f"pi install failed: {install.stderr.strip()[-500:]}")
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
        system_prompt, prompt = self.resolve_prompt(data)
        agent_dir = f".vf-pi-agent-{trace.id}"
        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        provider, separator, model = ctx.model.partition("/")
        if not separator:
            provider, model = "openai", ctx.model
        api = {
            "chat_completions": "openai-completions",
            "responses": "openai-responses",
            "anthropic_messages": "anthropic-messages",
        }[self.config.transport]
        base_url = (
            endpoint.removesuffix("/v1")
            if self.config.transport == "anthropic_messages"
            else endpoint
        )
        model_config = {
            "id": model,
            "reasoning": reasoning,
            "input": ["text", "image"],
            **(
                {"compat": {"sessionAffinityFormat": "openai-nosession"}}
                if self.config.transport == "responses"
                else {}
            ),
        }
        models = {
            "providers": {
                provider: {
                    "baseUrl": base_url,
                    "api": api,
                    "apiKey": f"${KEY_VAR}",
                    "models": [model_config],
                }
            }
        }
        await runtime.write(f"{agent_dir}/models.json", json.dumps(models).encode())

        mcp_args: list[str] = []
        if mcp_urls:
            extension_path = f"{agent_dir}/mcp.js"
            mcp = {
                "mcpServers": {
                    name: {"url": url, "lifecycle": "eager"}
                    for name, url in mcp_urls.items()
                }
            }
            extension = (
                f'import {{ createMcpAdapter }} from "{MCP_ADAPTER}";\n'
                "export default createMcpAdapter({ config: "
                f"JSON.parse({json.dumps(json.dumps(mcp))}) }});\n"
            )
            await runtime.write(extension_path, extension.encode())
            mcp_args = ["--extension", extension_path]

        env = {
            **self.config.resolved_env,
            KEY_VAR: secret,
            "PI_CODING_AGENT_DIR": agent_dir,
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
            "PI_ACP_CODING_AGENT_MODULE": f"{PACKAGES_DIR}/node_modules/@earendil-works/pi-coding-agent/dist/index.js",
        }
        skill_args = [
            arg
            for skill in self.config.skills
            # Resolve like `install_skills` so the path matches what it wrote.
            for arg in ("--skill", f"{SKILLS_DIR}/{skill.resolve().name}")
        ]
        pi_args = [
            PI_BIN,
            "--no-approve",
            "--provider",
            provider,
            "--model",
            model,
            *mcp_args,
            *skill_args,
        ]
        if self.config.disabled_tools:
            pi_args += ["--exclude-tools", ",".join(self.config.disabled_tools)]
        if system_prompt:
            pi_args += ["--append-system-prompt", system_prompt]
        pi_wrapper = f"{agent_dir}/pi"
        await runtime.write(
            pi_wrapper,
            f'#!/bin/sh\nexec {NODE_BIN_DIR}/node {shlex.join(pi_args)} "$@"\n'.encode(),
        )
        await runtime.run(["chmod", "+x", pi_wrapper], {})
        env["PI_ACP_PI_COMMAND"] = pi_wrapper
        return ACPConfig(
            env=env,
            command=ACP_COMMAND,
            prompt=prompt,
            # Pi's extension owns the task-scoped MCP configuration.
            mcp_urls={},
            # Pi can end after its final tool completes without a text message.
            allow_empty_tool_reply=True,
        )
