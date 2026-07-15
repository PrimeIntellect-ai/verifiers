"""Pi reaches interception through a custom OpenAI-compatible provider.

Pi intentionally leaves MCP to extensions, so the harness always installs and loads the
pi-mcp-adapter package.
"""

import base64
import json
import logging
import re
import shlex

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.types import SystemMessage, TextContentPart, UserMessage

logger = logging.getLogger(__name__)

PROVIDER = "intercept"
KEY_VAR = "PI_INTERCEPT_KEY"

PI_DIR = "/tmp/vf-pi"
PI_BIN = f"{PI_DIR}/pi"
MCP_VERSION = "2.11.0"
MCP_EXTENSION = f"{PI_DIR}/mcp/node_modules/pi-mcp-adapter/index.ts"

INSTALL = r"""
set -e
dir=/tmp/vf-pi
bin="$dir/pi"
mcp="$dir/mcp"

if ! command -v curl >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    { apt-get update -qq && apt-get install -y -qq curl ca-certificates nodejs npm >/dev/null; } \
        || apk add --no-cache curl ca-certificates nodejs npm >/dev/null
fi

if [ ! -x "$bin" ] || [ "$("$bin" --version 2>/dev/null)" != "{version}" ]; then
    case "$(uname -m)" in aarch64|arm64) arch=arm64 ;; *) arch=x64 ;; esac
    curl -fsSL "https://github.com/earendil-works/pi/releases/download/v{version}/pi-linux-${arch}.tar.gz" \
        | tar -xz -C "$dir" --strip-components=1
fi

[ "$(node -p "require('$mcp/node_modules/pi-mcp-adapter/package.json').version" 2>/dev/null)" = "{mcp_version}" ] \
    || npm install --prefix "$mcp" --ignore-scripts --no-audit --no-fund --omit=dev \
        "pi-mcp-adapter@{mcp_version}" >/dev/null
"""


class PiHarnessConfig(HarnessConfig):
    version: str = "0.80.6"
    """Pi release to install, pinned for reproducibility."""


class PiHarness(Harness[PiHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info(
            "pi: ensuring Pi %s and pi-mcp-adapter %s are installed",
            self.config.version,
            MCP_VERSION,
        )
        script = INSTALL.replace("{version}", self.config.version).replace(
            "{mcp_version}", MCP_VERSION
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
            f"sh -c {shlex.quote(script)}"
        )
        install = await runtime.run(["sh", "-c", guarded], {})
        if install.exit_code != 0:
            raise RuntimeError(f"pi install failed: {install.stderr.strip()[-500:]}")

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

        agent_dir = f"{PI_DIR}/agent-{trace.id}"
        image_args: list[str] = []
        if prompt is not None and not isinstance(prompt, str):
            system_texts = [system_prompt] if system_prompt else []
            texts: list[str] = []
            image_index = 0
            for message in prompt:
                if not isinstance(message, (SystemMessage, UserMessage)):
                    raise ValueError(
                        "Pi print mode only supports system and user initial messages"
                    )
                parts = (
                    [TextContentPart(text=message.content)]
                    if isinstance(message.content, str)
                    else message.content
                )
                for part in parts:
                    if isinstance(part, TextContentPart):
                        (system_texts if message.role == "system" else texts).append(
                            part.text
                        )
                        continue
                    metadata, separator, encoded = part.image_url.url.partition(",")
                    media_type, *parameters = metadata.removeprefix("data:").split(";")
                    if (
                        not separator
                        or not metadata.startswith("data:image/")
                        or not any(p.lower() == "base64" for p in parameters)
                    ):
                        raise ValueError(
                            "Pi image prompts require base64 data:image URLs"
                        )
                    extension = re.sub(
                        r"[^a-zA-Z0-9]+", "_", media_type.removeprefix("image/")
                    ).strip("_")
                    path = (
                        f"{agent_dir}/images/image_{image_index}.{extension or 'image'}"
                    )
                    await runtime.write(path, base64.b64decode(encoded))
                    image_args.append(f"@{path}")
                    image_index += 1
            system_prompt = "\n\n".join(system_texts) or None
            prompt = "\n\n".join(texts)
        if prompt is None:
            raise ValueError("Pi requires a task prompt (it has no user simulator)")

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
        mcp = {
            "mcpServers": {
                name: {"url": url, "lifecycle": "eager"}
                for name, url in mcp_urls.items()
            }
        }
        prompt_path = f"{agent_dir}/prompt.txt"
        await runtime.write(prompt_path, prompt.encode())
        await runtime.write(f"{agent_dir}/models.json", json.dumps(models).encode())
        await runtime.write(f"{agent_dir}/mcp.json", json.dumps(mcp).encode())

        env = {
            **self.config.resolved_env,
            KEY_VAR: secret,
            "PI_CODING_AGENT_DIR": agent_dir,
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
        }
        tool_args = (
            ["--exclude-tools", ",".join(self.config.disabled_tools)]
            if self.config.disabled_tools
            else []
        )
        system_args = ["--append-system-prompt", system_prompt] if system_prompt else []
        argv = [
            "sh",
            "-c",
            # Pi has no `--` terminator, so the prompt must not be parsed as argv.
            'exec "$@" < "$0"',
            prompt_path,
            PI_BIN,
            "--print",
            "--no-session",
            "--no-approve",
            "--offline",
            "--provider",
            PROVIDER,
            "--model",
            ctx.model,
            "--extension",
            MCP_EXTENSION,
            "--mcp-config",
            f"{agent_dir}/mcp.json",
            *tool_args,
            *system_args,
            *image_args,
        ]
        try:
            return await runtime.run_program(argv, env)
        finally:
            try:
                await runtime.run(["rm", "-rf", agent_dir], {})
            except Exception:
                logger.warning("failed to clean up Pi agent directory", exc_info=True)
