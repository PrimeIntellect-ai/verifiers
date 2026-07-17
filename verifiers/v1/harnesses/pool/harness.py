"""Pool (Poolside Agent CLI) reaches interception as an OpenAI-compatible provider.

`pool exec` runs a single prompt non-interactively and then exits — the non-interactive mode
targeted by every agent-CLI harness (Claude Code's `--print`, Codex's `exec`, Kimi's `--prompt`).

Pool reaches an arbitrary OpenAI-compatible provider (here, the interception server) through its
**Standalone provider mode**, the env-var idiom the install docs describe for CI:

| Var                            | Role                                             |
| ------------------------------ | ------------------------------------------------ |
| `POOLSIDE_API_KEY`             | Provider bearer token — set to the interception `secret` (Mirrors how `kimi_code`/`rlm`/
|                                | `claude_code` carry the secret through `KIMI_MODEL_API_KEY`/`RLM_API_KEY`/`ANTHROPIC_API_KEY`). |
| `POOLSIDE_STANDALONE_BASE_URL` | Provider base URL — set to the interception `endpoint` (Carries the `/v1` the chat dialect
|                                | serves, e.g. `http://127.0.0.1:port/v1`). |
| `POOLSIDE_STANDALONE_MODEL`    | Model name — set to `ctx.model` (Interception exposes no `/models`, so Pool can't enumerate one;
|                                | the proxy overlays `ctx.model` on the actual provider request anyway). |

This is **not** pool's ACP/`--agent-server` path: `POOLSIDE_API_URL`/`--api-url` target Pool's
*Platform* (the `POST /v0/me/access` bootstrap + tenant `auto-approve-commands` permission that
`--unsafe-auto-allow` checks), which is a separate Poolside identity from the model provider — so
the harness deliberately leaves the Platform alone and authenticates to the provider directly via
the Standalone vars above, the same one-shot, no-login auth CI uses (`POOLSIDE_API_KEY` as the
provider bearer + `POOLSIDE_STANDALONE_BASE_URL`/`POOLSIDE_STANDALONE_MODEL`).

Pool's project settings (`.poolside/settings.yaml`) and its per-run MCP config live next to the
directory `pool exec` runs in. The harness runs Pool inside the Verifiers runtime workdir (CWD), so
project settings written there are picked up automatically — the home-relative scheme `kimi_code`
uses for `mcp.json`.
"""

import json
import logging
import shlex

from pydantic import Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages, TextContentPart

logger = logging.getLogger(__name__)


# Pool is distributed as a per-release tarball on GitHub (`poolsideai/pool` releases), the same
# source codex and pi download from (`github.com/.../releases/download/v<version>/...tar.gz`). The
# `downloads.poolside.ai/pool/install.sh` helper only serves the *latest* release, so a pinned
# `--harness.version` fetches the versioned release tarball directly — like codex/pi pin theirs.
# The install dir is version-stamped (like claude_code's `CLAUDE_HOME`) so bumping
# `--harness.version` lands the binary in a fresh directory and forces a clean reinstall — no
# need to parse `pool --version` output to validate the pin.
POOLSIDE_DIR = "/tmp/vf-pool-{version}"
POOL_BIN = "{dir}/pool"
INSTALL = r"""
set -e
command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null)
command -v tar >/dev/null || (apt-get update -qq && apt-get install -y -qq tar >/dev/null)
case "$(uname -s)" in Linux) os=linux ;; Darwin) os=darwin ;; *) echo "unsupported os: $(uname -s)" >&2; exit 1 ;; esac
case "$(uname -m)" in aarch64|arm64) arch=arm64 ;; x86_64|amd64) arch=amd64 ;; *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;; esac
mkdir -p {dir}
curl -fsSL "https://github.com/poolsideai/pool/releases/download/v{version}/pool-$os-$arch.tar.gz" | tar -xz -C {dir}
mv "{dir}/pool-$os-$arch" "{dir}/pool"
chmod +x "{dir}/pool"
"""


class PoolHarnessConfig(HarnessConfig):
    version: str = Field(default="1.0.11", pattern=r"^[A-Za-z0-9._+-]+$")
    """Poolside Agent CLI release to install, pinned for reproducibility. Override per run with
    `--harness.version <release>` (fetched directly from the GitHub release tarball)."""


class PoolHarness(Harness[PoolHarnessConfig]):
    # `pool exec` exposes no `--append-system-prompt` / system-message flag, so — like kimi_code
    # and codex — a task `system_prompt` is folded into the prompt text (APPENDS_SYSTEM_PROMPT stays
    # at its False default; `resolve_prompt` prepends it).
    APPENDS_SYSTEM_PROMPT = False
    SUPPORTS_MCP = True
    # Structured (Messages) prompts are flattened to text for `pool exec`'s text-only `-p`/`-f`,
    # the same shape codex supports (minus images, which pool exec cannot ingest).
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        pool_dir = POOLSIDE_DIR.format(version=self.config.version)
        pool_bin = POOL_BIN.format(dir=pool_dir)
        # `.replace` (codex/kimi/pi style) because `INSTALL` carries shell `${os}`/`${arch}`.
        script = INSTALL.replace("{version}", self.config.version).replace(
            "{dir}", pool_dir
        )
        # Cache the pinned binary across local rollouts; Linux has flock, macOS has lockf.
        install = shlex.quote(f"[ -x {pool_bin} ] || ({script})")
        guarded = (
            f"mkdir -p {pool_dir} && "
            f'"$(command -v flock || command -v lockf)" {pool_dir}/install.lock '
            f"bash -o pipefail -c {install}"
        )
        result = await runtime.run(["sh", "-c", guarded], self.config.resolved_env)
        if result.exit_code != 0:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError(f"Poolside Agent CLI install failed: {detail}")

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if self.config.disabled_tools:
            raise ValueError(
                "the pool harness runs Pool's own tool set (shell, edit, MCP) and does not "
                "support `disabled_tools`; map tool denials through `settings.yaml` instead."
            )
        task = trace.task.data
        # Mirror codex: bypass `resolve_prompt`'s system-fold for a structured prompt that also
        # carries a `system_prompt` (it would raise folding a system prompt into a Messages
        # prompt); otherwise let resolve_prompt fold the system into the text for us.
        if (
            task.system_prompt is not None
            and task.prompt is not None
            and not isinstance(task.prompt, str)
        ):
            system_prompt, prompt = task.system_prompt, task.prompt
        else:
            system_prompt, prompt = self.resolve_prompt(task)
        text = _prompt_to_text(system_prompt, prompt)

        # Standalone OpenAI-compatible provider auth to interception (mirrors kimi_code's
        # `KIMI_MODEL_*`/`RLM_*`/`ANTHROPIC_*` env-based provider wiring — NOT pool's Platform,
        # whose `POOLSIDE_API_URL`/ACP path is a separate identity). `--api-url`/`-a` select the
        # ACP/Platform session; the harness pins the *provider* via the Standalone vars so the
        # interception `secret` is the bearer pool sends to /v1/chat/completions. `POOLSIDE_API_KEY`
        # here is Pool's provider bearer (its CLI reference: "Auth token for ... an OpenAI-
        # compatible provider"), and the subprocess runtime strips host `*API_KEY` vars and
        # re-applies this explicit one, so a host key can't shadow the interception secret.
        env = {
            **self.config.resolved_env,
            "POOLSIDE_API_KEY": secret,
            "POOLSIDE_STANDALONE_BASE_URL": endpoint,
            "POOLSIDE_STANDALONE_MODEL": ctx.model,
        }

        # `pool exec` reads project settings (`.poolside/settings.yaml`) from its CWD, which the
        # runtime sets to the working directory; hand the task's tool/user servers to Pool as MCP
        # servers in the documented `mcp_servers` block. JSON is a YAML subset, so the payload
        # parses as settings.yaml (kimi_code/claude write an equivalent `mcp.json`).
        settings_path = ".poolside/settings.yaml"
        if mcp_urls:
            await runtime.write(
                settings_path,
                json.dumps(
                    {
                        "mcp_servers": {
                            name: {"transport": {"type": "http", "url": url}}
                            for name, url in mcp_urls.items()
                        }
                    }
                ).encode(),
            )

        # `pool exec -p -` reads the prompt from standard input, but the Runtime launches the
        # harness program without wiring child stdin — so feed the prompt through a file with
        # `pool exec --prompt-file` (`-f`), the file equivalent of `-p -`. This mirrors how the
        # default/null harnesses hand non-string or image-bearing prompts off through a file to
        # dodge exec-arg limits and quoting, and is the natural Pool idiom for a prompt sourced
        # outside the inline `-p`.
        prompt_path = f".vf-pool-prompt-{trace.id}"
        await runtime.write(prompt_path, text.encode())
        pool_bin = POOL_BIN.format(dir=POOLSIDE_DIR.format(version=self.config.version))
        argv = [
            pool_bin,
            "exec",
            "--unsafe-auto-allow",  # non-interactive: auto-approve tool actions without prompts
            "--sandbox",
            "disabled",  # the Verifiers runtime is the sandbox; don't nest Pool's own
            "--prompt-file",
            prompt_path,
        ]
        return await runtime.run_program(argv, env)


def _prompt_to_text(system_prompt: str | None, prompt: str | Messages | None) -> str:
    """Render the opening prompt as the single text `pool exec` seeds.

    `pool exec` only accepts a text prompt (`-p` / `--prompt-file`), so a `Messages` prompt is
    flattened by concatenating its system + user messages (codex keeps images and seeds them via
    `-i`; Pool's exec has no image input). `system_prompt` is already folded in for the string
    case (APPENDS_SYSTEM_PROMPT=False -> resolve_prompt prepends it); for a structured prompt the
    caller leaves it separate and we prepend it here."""
    if isinstance(prompt, str):
        text = prompt
    elif prompt is None:
        raise ValueError(
            "the pool harness requires a task prompt (it has no user simulator)"
        )
    else:
        blocks: list[str] = []
        for message in prompt:
            if message.role not in ("system", "user"):
                raise ValueError(
                    "pool exec only supports system and user initial messages "
                    "(assistant/tool turns can't be replayed into the agent)"
                )
            content = message.content
            if isinstance(content, str):
                blocks.append(content)
                continue
            # system/user messages carry `str | list[ContentPart]`; the role guard above keeps us
            # in that lane, but typing sees the full `Message` union (incl. `None` content), so
            # narrow to a list before iterating.
            if not isinstance(content, list):
                continue
            for part in content:  # list[ContentPart]
                if isinstance(part, TextContentPart):
                    blocks.append(part.text)
                else:
                    raise ValueError(
                        "pool exec is text-only (its `-p`/`--prompt-file` take no image flag); "
                        "image prompts are unsupported"
                    )
        text = "\n\n".join(b for b in blocks if b)
    if system_prompt:
        text = f"{system_prompt}\n\n{text}" if text else system_prompt
    return text
