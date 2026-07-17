"""Pi harness using the upstream pi-acp adapter."""

import json
import logging
import shlex

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

PROVIDER = "intercept"
KEY_VAR = "PI_INTERCEPT_KEY"
HOME_VAR = "VF_PI_ORIGINAL_HOME"

PI_DIR = "/tmp/vf-pi"
PACKAGES_DIR = f"{PI_DIR}/mcp"
PI_BIN = f"{PACKAGES_DIR}/node_modules/.bin/pi"
MCP_VERSION = "2.11.0"
ACP_VERSION = "0.0.31"
NODE_VERSION = "22.19.0"
MCP_ADAPTER = f"{PACKAGES_DIR}/node_modules/pi-mcp-adapter/index.ts"
ACP_BIN = f"{PACKAGES_DIR}/node_modules/.bin/pi-acp"
ACP_COMMAND = [
    "sh",
    "-c",
    f'export {HOME_VAR}="$HOME"; '
    'PI_CODING_AGENT_DIR="$PWD/$PI_CODING_AGENT_DIR"; '
    'export PI_CODING_AGENT_DIR HOME="$PI_CODING_AGENT_DIR"; '
    f'export PATH="{PI_DIR}/node/bin:$PATH"; exec {ACP_BIN}',
]

INSTALL = r"""
set -e
packages=/tmp/vf-pi/mcp
node=/tmp/vf-pi/node

if [ -f /etc/alpine-release ]; then
    apk add --no-cache curl ca-certificates nodejs-current npm >/dev/null
    if ! node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=19 ? 0 : 1)'; then
        sed -E -i 's/v[0-9]+\.[0-9]+/v3.22/g' /etc/apk/repositories
        apk upgrade --available --no-cache >/dev/null
        apk add --no-cache nodejs-current npm >/dev/null
    fi
    node_bin="$(dirname "$(command -v node)")"
else
    command -v curl >/dev/null 2>&1 \
        || { apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; }
    if [ ! -x "$node/bin/node" ] || [ "$("$node/bin/node" --version 2>/dev/null)" != "v$VF_PI_NODE_VERSION" ]; then
        case "$(uname -m)" in aarch64|arm64) node_arch=arm64 ;; *) node_arch=x64 ;; esac
        rm -rf "$node"
        mkdir -p "$node"
        curl -fsSL "https://nodejs.org/dist/v$VF_PI_NODE_VERSION/node-v$VF_PI_NODE_VERSION-linux-${node_arch}.tar.gz" \
            | tar -xz -C "$node" --strip-components=1
    fi
    node_bin="$node/bin"
fi
export PATH="$node_bin:$PATH"
node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=19 ? 0 : 1)' \
    || { echo "pi-acp requires Node.js 22.19 or newer" >&2; exit 1; }

pi_version="$(node -p "require('$packages/node_modules/@earendil-works/pi-coding-agent/package.json').version" 2>/dev/null || true)"
mcp_version="$(node -p "require('$packages/node_modules/pi-mcp-adapter/package.json').version" 2>/dev/null || true)"
acp_version="$(node -p "require('$packages/node_modules/pi-acp/package.json').version" 2>/dev/null || true)"
if [ "$pi_version" != "$VF_PI_VERSION" ] || [ "$mcp_version" != "$VF_PI_MCP_VERSION" ] || [ "$acp_version" != "$VF_PI_ACP_VERSION" ]; then
    npm install --prefix "$packages" --ignore-scripts --no-audit --no-fund --omit=dev \
        "@earendil-works/pi-coding-agent@$VF_PI_VERSION" \
        "pi-mcp-adapter@$VF_PI_MCP_VERSION" \
        "pi-acp@$VF_PI_ACP_VERSION" >/dev/null
fi
"""

# Isolate pi-mcp-adapter discovery while it registers, then restore the task home.
MCP_WRAPPER = f"""
export default async function isolatedMcp(pi) {{
  const agentDir = process.env.PI_CODING_AGENT_DIR;
  const cwd = process.cwd();
  process.chdir(agentDir);
  process.env.HOME = agentDir;
  try {{
    const {{ default: mcpAdapter }} = await import("{MCP_ADAPTER}");
    const isolatedPi = {{
      ...pi,
      on(event, handler) {{
        pi.on(
          event,
          event === "session_start"
            ? (event, ctx) => handler(event, {{ ...ctx, cwd: agentDir }})
            : handler,
        );
      }},
    }};
    mcpAdapter(isolatedPi);
  }} finally {{
    process.chdir(cwd);
    process.env.HOME = process.env.{HOME_VAR};
    delete process.env.{HOME_VAR};
  }}
}}
""".strip()

PI_ACP = ACP()


class PiHarnessConfig(HarnessConfig):
    version: str = "0.80.10"
    """Pi release to install, pinned for reproducibility."""


class PiHarness(Harness[PiHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "pi: ensuring Pi %s and pi-acp %s are installed",
            self.config.version,
            ACP_VERSION,
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
            f"sh -c {shlex.quote(INSTALL)}"
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                "VF_PI_VERSION": self.config.version,
                "VF_PI_MCP_VERSION": MCP_VERSION,
                "VF_PI_ACP_VERSION": ACP_VERSION,
                "VF_PI_NODE_VERSION": NODE_VERSION,
            },
        )
        if install.exit_code != 0:
            raise RuntimeError(f"pi install failed: {install.stderr.strip()[-500:]}")
        await PI_ACP.setup(self, runtime)

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
        system_prompt, prompt = self.resolve_prompt(data)
        agent_dir = f".vf-pi-agent-{trace.id}"
        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    "apiKey": f"${KEY_VAR}",
                    "models": [
                        {
                            "id": ctx.model,
                            "reasoning": reasoning,
                            "input": ["text", "image"],
                        }
                    ],
                }
            }
        }
        await runtime.write(f"{agent_dir}/models.json", json.dumps(models).encode())

        mcp_args: list[str] = []
        restore_home = f'export HOME="${HOME_VAR}"; unset {HOME_VAR}; '
        if mcp_urls:
            extension_path = f"{agent_dir}/mcp.js"
            mcp = {
                "mcpServers": {
                    name: {"url": url, "lifecycle": "eager"}
                    for name, url in mcp_urls.items()
                }
            }
            await runtime.write(f"{agent_dir}/mcp.json", json.dumps(mcp).encode())
            await runtime.write(extension_path, MCP_WRAPPER.encode())
            mcp_args = [
                "--extension",
                extension_path,
                "--mcp-config",
                f"{agent_dir}/mcp.json",
            ]
            restore_home = ""

        env = {
            **self.config.resolved_env,
            KEY_VAR: secret,
            "PI_CODING_AGENT_DIR": agent_dir,
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
        }
        pi_args = [
            PI_BIN,
            "--no-approve",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
            *mcp_args,
            *(
                ["--exclude-tools", ",".join(self.config.disabled_tools)]
                if self.config.disabled_tools
                else []
            ),
            *(["--append-system-prompt", system_prompt] if system_prompt else []),
        ]
        pi_wrapper = f"{agent_dir}/pi"
        await runtime.write(
            pi_wrapper,
            f'#!/bin/sh\n{restore_home}exec {shlex.join(pi_args)} "$@"\n'.encode(),
        )
        chmod = await runtime.run(["chmod", "+x", pi_wrapper], {})
        if chmod.exit_code != 0:
            raise RuntimeError(f"Pi wrapper setup failed: {chmod.stderr}")
        env["PI_ACP_PI_COMMAND"] = pi_wrapper
        return await PI_ACP.run(
            runtime,
            env,
            ACP_COMMAND,
            prompt,
            session_path=f"{agent_dir}/acp-session",
        )
