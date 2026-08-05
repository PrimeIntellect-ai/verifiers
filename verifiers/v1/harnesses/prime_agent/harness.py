"""Prime Agent harness driving prime-agent's native ACP mode."""

import json
import logging
import shlex

from pydantic import Field

from verifiers.v1.acp import ACP
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

# Model traffic is routed through the interception endpoint, which prime-agent
# sees as an ordinary OpenAI-compatible provider.
#
# The agent-dir env var is derived from the package's own piConfig name, so the
# published prime-agent release reads PRIME_AGENT_CODING_AGENT_DIR, not the
# upstream PI_ prefix.
ENV_AGENT_DIR = "PRIME_AGENT_CODING_AGENT_DIR"
PROVIDER = "intercept"
# Stand-in written to models.json; the launch wrapper swaps in the real secret.
APIKEY_PLACEHOLDER = "__VF_PRIME_AGENT_KEY__"
KEY_VAR = "PRIME_AGENT_INTERCEPT_KEY"

PRIME_AGENT_DIR = "/tmp/vf-prime-agent"
PRIME_AGENT_BIN = f"{PRIME_AGENT_DIR}/node_modules/.bin/prime-agent"
SKILLS_DIR = ".agents/skills"
INSTALLER_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/install.sh"
NODE_VERSION = "22.19.0"
NODE_BIN = f"{PRIME_AGENT_DIR}/node/bin"

INSTALL = r"""
set -e
node="$VF_PRIME_AGENT_DIR/node"
node_ok() { node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=8 ? 0 : 1)'; }

# Containers do not ship Node, and prime-agent requires >=22.8. Prefer the
# distro package, otherwise fetch the pinned official build.
if [ -f /etc/alpine-release ]; then
    apk add --no-cache curl ca-certificates nodejs-current npm >/dev/null
    if ! node_ok; then
        # The official Node build is glibc-only, so an Alpine whose own
        # nodejs-current is too old has to move its repos forward and retry.
        sed -E -i 's/v[0-9]+\.[0-9]+/v3.22/g' /etc/apk/repositories
        apk upgrade --available --no-cache >/dev/null
        apk add --no-cache nodejs-current npm >/dev/null
    fi
    node_bin="$(dirname "$(command -v node)")"
else
    if ! command -v curl >/dev/null 2>&1; then
        # Only Debian-family images are provisioned here; say so instead of
        # failing with "apt-get: not found" on a distro this cannot bootstrap.
        command -v apt-get >/dev/null 2>&1 \
            || { echo "prime-agent setup needs curl, or apt-get to install it" >&2; exit 1; }
        apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null
    fi
    case "$(uname -s)" in Linux) node_os=linux ;; Darwin) node_os=darwin ;; *) echo "unsupported os: $(uname -s)" >&2; exit 1 ;; esac
    if [ ! -x "$node/bin/node" ] || [ "$("$node/bin/node" --version 2>/dev/null)" != "v$VF_PRIME_AGENT_NODE_VERSION" ]; then
        # Reject an unknown machine like the OS check does: guessing x64 would
        # download an archive whose node cannot exec, failing much later.
        case "$(uname -m)" in
            aarch64|arm64) node_arch=arm64 ;;
            x86_64|amd64) node_arch=x64 ;;
            *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
        esac
        rm -rf "$node"
        mkdir -p "$node"
        curl -fsSL "https://nodejs.org/dist/v$VF_PRIME_AGENT_NODE_VERSION/node-v$VF_PRIME_AGENT_NODE_VERSION-${node_os}-${node_arch}.tar.gz" \
            | tar -xz -C "$node" --strip-components=1
    fi
    node_bin="$node/bin"
fi
export PATH="$node_bin:$PATH"
node_ok || { echo "prime-agent requires Node.js 22.8 or newer" >&2; exit 1; }

# Concurrent rollouts share the install, so it is keyed on the requested
# tarball: a changed `version` or `tarball_url` must reinstall rather than
# silently reuse whatever an earlier rollout left in the shared directory.
stamp="$VF_PRIME_AGENT_DIR/.installed"
if [ -x "$VF_PRIME_AGENT_BIN" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$VF_PRIME_AGENT_TARBALL" ]; then
    exit 0
fi
mkdir -p "$VF_PRIME_AGENT_DIR"
# Drop the stamp first so a failed install is never mistaken for a good one.
rm -f "$stamp"
# The published release tarball is a plain npm package, so install it directly
# instead of running the interactive installer script.
npm install --no-audit --no-fund --prefix "$VF_PRIME_AGENT_DIR" \
    "$VF_PRIME_AGENT_TARBALL" >/dev/null
printf %s "$VF_PRIME_AGENT_TARBALL" > "$stamp"
"""

PRIME_AGENT_ACP = ACP()


class PrimeAgentHarnessConfig(HarnessConfig):
    version: str = "0.6.0"
    """Prime Agent release to install, pinned for reproducibility.

    0.6.0 is the first release with native ACP mode (`--mode acp`).
    """

    tarball_url: str | None = None
    """Override the release tarball URL (defaults to the pinned public release)."""

    autonomous: bool = False
    """Run with `--autonomous`, so the agent continues until its limits or gates stop it."""

    gates: list[str] = Field(default_factory=list)
    """Autonomous quality-gate commands. Requires `autonomous`."""


class PrimeAgentHarness(Harness[PrimeAgentHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    # prime-agent's ACP mode ignores the `mcpServers` of `session/new`, and its own
    # MCP integrations are authored Python-backed skills the model imports in its
    # kernel, not tools an ACP client can inject. Claiming support would let a
    # tool-bearing taskset run with its tools silently absent, so declare it false
    # and let `validate_pairing` reject that pairing up front.
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = True
    SUPPORTS_SKILLS = True

    def _tarball(self) -> str:
        if self.config.tarball_url:
            return self.config.tarball_url
        version = self.config.version
        base = INSTALLER_URL.rsplit("/", 1)[0]
        return f"{base}/releases/v{version}/prime-agent-{version}.tgz"

    async def setup(self, runtime: Runtime) -> None:
        await self.install_skills(runtime, SKILLS_DIR)
        logger.info("prime-agent: ensuring %s is installed", self.config.version)
        lock = f"{PRIME_AGENT_DIR}/install.lock"
        # Concurrent rollouts share one runtime; serialize the install like the
        # other node-based harnesses do.
        guarded = (
            f"mkdir -p {PRIME_AGENT_DIR} && "
            f'"$(command -v flock || command -v lockf)" {lock} '
            f"sh -c {shlex.quote(INSTALL)}"
        )
        install = await runtime.run(
            ["sh", "-c", guarded],
            {
                "VF_PRIME_AGENT_DIR": PRIME_AGENT_DIR,
                "VF_PRIME_AGENT_BIN": PRIME_AGENT_BIN,
                "VF_PRIME_AGENT_TARBALL": self._tarball(),
                "VF_PRIME_AGENT_NODE_VERSION": NODE_VERSION,
            },
        )
        if install.exit_code != 0:
            raise RuntimeError(
                f"prime-agent install failed: {install.stderr.strip()[-500:]}"
            )
        await PRIME_AGENT_ACP.setup(self, runtime)

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
        # A resumed segment replays the accreted conversation, and the ACP runner
        # renders that transcript into the prompt — including the `[system]` block
        # it rendered on the first segment. Re-emitting the system prompt here
        # would hand the model the same instructions twice.
        if trace.branches and trace.branches[-1].messages:
            system_prompt = None
        agent_dir = f".vf-prime-agent-{trace.id}"
        reasoning = ctx.sampling.reasoning_effort not in (
            None,
            "none",
        ) or ctx.model.rsplit("/", 1)[-1].startswith(("gpt-5", "o1", "o3", "o4"))
        models = {
            "providers": {
                PROVIDER: {
                    "baseUrl": endpoint,
                    "api": "openai-completions",
                    # prime-agent does not expand "$VAR" in models.json: it sends the
                    # literal string as the bearer token (verified against a capturing
                    # server), so the pi-style indirection cannot be used here. The
                    # secret is substituted by the launch wrapper at exec time instead
                    # of being written to disk, because concurrent rollouts share a
                    # runtime and model-executed code could otherwise read another
                    # rollout's credential out of its models.json.
                    "apiKey": APIKEY_PLACEHOLDER,
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
        models_path = f"{agent_dir}/models.json"
        await runtime.write(models_path, json.dumps(models).encode())

        env = {
            **self.config.resolved_env,
            KEY_VAR: secret,
            ENV_AGENT_DIR: agent_dir,
            "PRIME_AGENT_OFFLINE": "1",
            "PRIME_AGENT_TELEMETRY": "0",
        }

        args = [
            PRIME_AGENT_BIN,
            "--mode",
            "acp",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
        ]
        for skill in self.config.skills:
            # Resolve like `install_skills` so the path matches what it wrote.
            args += ["--skill", f"{SKILLS_DIR}/{skill.resolve().name}"]
        if self.config.disabled_tools:
            raise ValueError(
                "prime-agent has no per-tool disable flag; its model-facing tool is "
                "ipython. Use --no-builtin-tools upstream if that is ever needed."
            )
        if self.config.autonomous:
            args.append("--autonomous")
            for gate in self.config.gates:
                args += ["--autonomous-gate", gate]
        elif self.config.gates:
            raise ValueError("prime-agent gates require autonomous=true")
        # The ACP runner seeds the system prompt into the conversation itself, so
        # passing --append-system-prompt as well would apply it twice.

        # ACP owns stdout, so the agent must be exec'd directly with HOME pinned
        # to the per-trace agent dir: prime-agent writes sessions and kernel state
        # under it, and rollouts must not share either.
        wrapper = f"{agent_dir}/prime-agent"
        await runtime.write(
            wrapper,
            (
                "#!/bin/sh\n"
                "set -e\n"
                # The bundled Node lives beside the install, not on the container PATH.
                f'export PATH="{NODE_BIN}:$PATH"\n'
                f'{ENV_AGENT_DIR}="$PWD/${ENV_AGENT_DIR}"\n'
                f'export {ENV_AGENT_DIR} HOME="${ENV_AGENT_DIR}"\n'
                # Substitute the bearer token into models.json here, so the plaintext
                # credential is only readable by this rollout's own agent process.
                # Node rewrites the parsed document rather than a text substitution:
                # a token carrying a backslash, quote, or `sed` metacharacter would
                # otherwise corrupt the key or abort the wrapper before exec.
                f'models="${ENV_AGENT_DIR}/models.json"\n'
                f"umask 077\n"
                "node -e '"
                'const fs=require("fs"),p=process.argv[1],'
                'c=JSON.parse(fs.readFileSync(p,"utf8"));'
                f'c.providers["{PROVIDER}"].apiKey=process.env.{KEY_VAR}||"";'
                'fs.writeFileSync(p,JSON.stringify(c))\' "$models"\n'
                f'exec {shlex.join(args)} "$@"\n'
            ).encode(),
        )
        await runtime.run(["chmod", "+x", wrapper], {})
        # models.json is world-readable as written; restrict it before it holds a secret.
        await runtime.run(["chmod", "700", agent_dir], {})
        await runtime.run(["chmod", "600", models_path], {})

        return await PRIME_AGENT_ACP.run(
            runtime,
            env,
            ["sh", "-c", f'exec "$PWD/{wrapper}"'],
            prompt,
            system_prompt=system_prompt,
        )
