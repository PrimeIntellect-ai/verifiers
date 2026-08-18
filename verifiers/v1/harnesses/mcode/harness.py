"""Run MiniMax Code's headless CLI as a v1 harness."""

import json
import logging
import shlex

from pydantic import Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.node import NODE_BIN_DIR, ensure_node
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages, content_text

logger = logging.getLogger(__name__)

MCODE_DIR = "/var/tmp/vf-mcode-{version}"
PACKAGES_DIR = f"{MCODE_DIR}/packages"
MCODE_BIN = f"{PACKAGES_DIR}/node_modules/.bin/mcode"
MCODE_VERSION = "0.1.4"

INSTALL = r"""
set -e
export PATH="/var/tmp/vf-node/bin:$PATH"
rm -f {ready}
npm install --prefix {packages} --no-audit --no-fund --omit=dev \
    "@minimax-ai/code@$VF_MCODE_VERSION" >/dev/null
touch {ready}
"""

# MiniMax Code 0.1.4 requires a literal custom-provider key in config. Keep only a
# harmless placeholder there, then replace its request header from a Node preload.
# The preload consumes and unlinks a one-use credential file before the CLI module
# loads, and removes its control variables before any model-directed tool can run.
PRELOAD = r"""
import fs from "node:fs";

const secretPath = process.env.VF_MCODE_SECRET_FILE;
const endpointValue = process.env.VF_MCODE_ENDPOINT;
if (!secretPath || !endpointValue) {
  throw new Error("missing mcode interception preload configuration");
}
const secret = fs.readFileSync(secretPath, "utf8");
fs.unlinkSync(secretPath);
delete process.env.VF_MCODE_SECRET_FILE;
delete process.env.VF_MCODE_ENDPOINT;
delete process.env.NODE_OPTIONS;

const endpoint = new URL(endpointValue);
const endpointPath = endpoint.pathname.replace(/\/$/, "");
const metricsHosts = new Set(["agent.minimaxi.com", "agent.minimax.io"]);
const metricsPath = "/matrix/api/v1/metrics/batch";
const originalFetch = globalThis.fetch;

globalThis.fetch = function interceptedFetch(input, init = undefined) {
  const request = input instanceof Request ? input : undefined;
  const url = new URL(request?.url ?? String(input));
  if (metricsHosts.has(url.hostname) && url.pathname === metricsPath) {
    return Promise.resolve(new Response(null, { status: 204 }));
  }
  const inScope =
    url.origin === endpoint.origin &&
    (url.pathname === endpointPath || url.pathname.startsWith(`${endpointPath}/`));
  if (!inScope) {
    return originalFetch(input, init);
  }

  const headers = new Headers(
    init && Object.hasOwn(init, "headers") ? init.headers : request?.headers,
  );
  headers.set("authorization", `Bearer ${secret}`);
  if (request) {
    return originalFetch(new Request(request, { ...init, headers }));
  }
  return originalFetch(input, { ...init, headers });
};
"""

BUILTIN_TOOLS = ("read", "write", "edit", "bash", "grep", "glob")


class MCodeHarnessConfig(HarnessConfig):
    version: str = Field(default=MCODE_VERSION, pattern=r"^[A-Za-z0-9._+-]+$")
    """MiniMax Code release to install, pinned for reproducibility."""


class MCodeHarness(Harness[MCodeHarnessConfig]):
    """Harness backed by MiniMax Code's non-interactive ``exec`` command."""

    SUPPORTS_MCP = False

    def _tools(self) -> list[str]:
        disabled = set(self.config.disabled_tools or ())
        unsupported = disabled - set(BUILTIN_TOOLS)
        if unsupported:
            raise ValueError(
                "MiniMax Code does not recognize disabled tools: "
                + ", ".join(sorted(unsupported))
            )
        return [tool for tool in BUILTIN_TOOLS if tool not in disabled]

    async def setup(self, runtime: Runtime) -> None:
        await ensure_node(runtime)
        versions = {"version": self.config.version}
        directory = MCODE_DIR.format(**versions)
        packages = PACKAGES_DIR.format(**versions)
        mcode_bin = MCODE_BIN.format(**versions)
        ready = f"{directory}/.ready"
        script = INSTALL.replace("{packages}", packages).replace("{ready}", ready)
        ensure = shlex.quote(f"[ -f {ready} ] && [ -x {mcode_bin} ] || ({script})")
        lock = f"{directory}/install.lock"
        guarded = (
            f"mkdir -p {directory} && "
            f'until ln -s "$$" {lock} 2>/dev/null; do '
            f"owner=$(readlink {lock}); "
            f'if ! kill -0 "$owner" 2>/dev/null; then '
            f'[ "$(readlink {lock})" != "$owner" ] || rm -f {lock}; fi; '
            f"sleep 0.1; done; "
            f'trap \'[ "$(readlink {lock})" != "$$" ] || rm -f {lock}\' EXIT; '
            f"sh -c {ensure}"
        )
        logger.info("mcode: ensuring MiniMax Code %s is installed", self.config.version)
        result = await runtime.run(
            ["sh", "-c", guarded], {"VF_MCODE_VERSION": self.config.version}
        )
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"MiniMax Code install failed: {detail}")

    @staticmethod
    def _home(trace: Trace) -> str:
        return f".vf-mcode/{trace.id}"

    async def _write_config(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
    ) -> tuple[str, str, str]:
        home = self._home(trace)
        data_dir = f"{home}/data"
        config_path = f"{home}/config.yaml"
        preload_path = f"{home}/interception-preload.mjs"
        tools = self._tools()
        config = {
            "logLevel": "warn",
            "agents": {
                "default": {
                    "persona": {"enabled": False},
                    "tools": tools,
                    "builtinTools": [],
                    "skills": [],
                    "features": {
                        "mavis": False,
                        "delegation": False,
                        "webSearch": False,
                    },
                }
            },
            "memory": {"enabled": False, "proactive": False},
            "askUser": {"enabled": False},
            "skills": {"external": {"enabled": False}},
            "skillEvolve": {"enabled": False},
            "beta": {
                "autoMemory": False,
                "skillEvolve": False,
                "skillEvolveBuiltinMr": False,
                "skillProposal": False,
                "browserBridge": False,
                "filePanelBrowser": False,
                "filePanelBrowserMultiTab": False,
                "browserUseTooling": False,
                "browserAgentCursor": False,
                "desktopPlanMode": False,
                "peek": False,
                "keepAlive": False,
                "promptOverride": False,
                "cuMode": False,
                "asr": False,
                "teamPlanWorkspaceCard": False,
                "teamPlanDrilldown": False,
                "taskHistoryProjectGrouping": False,
                "threadGoal": False,
                "mcodeTools": False,
                "codexOAuth": False,
            },
            "custom_provider": {
                "verifiers": {
                    "name": "verifiers",
                    "kind": "custom",
                    "enabled": True,
                    "api": "openai-completions",
                    "options": {
                        "apiKey": "verifiers-interception",
                        "baseURL": endpoint,
                        "authMode": "api-key",
                    },
                    "models": {
                        ctx.model: {"reasoning": False, "tool_call": bool(tools)}
                    },
                }
            },
            "defaultModel": f"custom_provider:verifiers/{ctx.model}",
        }
        created = await runtime.run(["mkdir", "-m", "700", "-p", home, data_dir], {})
        if created.exit_code != 0:
            raise RuntimeError(f"failed to create mcode home: {created.stderr.strip()}")
        await runtime.write(config_path, json.dumps(config).encode())
        await runtime.write(preload_path, PRELOAD.encode())
        secured = await runtime.run(["chmod", "600", config_path, preload_path], {})
        if secured.exit_code != 0:
            raise RuntimeError(f"failed to secure mcode config: {secured.stderr.strip()}")
        return data_dir, config_path, preload_path

    async def _write_secret(self, trace: Trace, runtime: Runtime, secret: str) -> str:
        path = f"{self._home(trace)}/interception-secret"
        await runtime.write(path, secret.encode())
        secured = await runtime.run(["chmod", "600", path], {})
        if secured.exit_code != 0:
            raise RuntimeError(
                f"failed to secure mcode interception secret: {secured.stderr.strip()}"
            )
        return path

    @staticmethod
    def _prompt_text(prompt: str | Messages | None) -> str:
        if isinstance(prompt, str):
            if not prompt:
                raise ValueError("mcode exec requires a non-empty prompt")
            return prompt
        if prompt is None:
            raise ValueError("mcode exec requires a prompt")
        parts = []
        for message in prompt:
            text = content_text(message.content)
            if text:
                parts.append(f"[{message.role}]\n{text}")
        if not parts:
            raise ValueError("mcode exec requires a non-empty prompt")
        return "\n\n".join(parts)

    async def _exec(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        data: TaskData,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if mcp_urls:
            raise ValueError("mcode headless harness does not support MCP URLs yet")
        _, prompt = self.resolve_prompt(data)
        prompt_text = self._prompt_text(prompt)
        data_dir, config_path, preload_path = await self._write_config(
            ctx, trace, runtime, endpoint
        )
        secret_path = await self._write_secret(trace, runtime, secret)
        versions = {"version": self.config.version}
        argv = [
            NODE_BIN_DIR + "/node",
            MCODE_BIN.format(**versions),
            "exec",
            "--config",
            config_path,
            "--permission",
            "off",
            "--max-steps",
            "1000",
            "--output-format",
            "json",
        ]
        argv.append(prompt_text)
        result = await runtime.run_program(
            argv,
            {
                **self.config.resolved_env,
                "MINIMAX_DATA_DIR": data_dir,
                "NO_COLOR": "1",
                "NODE_OPTIONS": f"--import=./{preload_path}",
                "VF_MCODE_ENDPOINT": endpoint,
                "VF_MCODE_SECRET_FILE": secret_path,
            },
        )
        if result.exit_code != 0:
            return result
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "MiniMax Code returned non-JSON output: "
                + result.stdout.strip()[-500:]
            ) from error
        if payload.get("status") != "succeeded":
            raise RuntimeError(
                "MiniMax Code run failed: "
                + str(payload.get("error") or payload.get("status") or "unknown error")
            )
        return result

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        return await self._exec(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            data,
            mcp_urls,
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        result = await runtime.run(["rm", "-rf", self._home(trace)], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up MiniMax Code data: {result.stderr.strip()[-500:]}"
            )
