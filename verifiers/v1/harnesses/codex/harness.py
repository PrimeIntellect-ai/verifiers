"""Codex reaches interception as a Responses API provider using the session secret.

The static musl binary runs in Linux containers without additional runtime dependencies.
"""

import base64
import logging
import re
import shlex

from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace
from verifiers.v1.types import TextContentPart
from verifiers.v1.task import TaskData

logger = logging.getLogger(__name__)

# The provider id we register on the fly via `-c` overrides — arbitrary, internal to codex.
PROVIDER = "intercept"
# The env var codex reads the provider api key (its bearer = the session secret) from.
KEY_VAR = "CODEX_INTERCEPT_KEY"

CODEX_DIR = "/tmp/vf-codex"
CODEX_BIN = f"{CODEX_DIR}/bin/codex"
INSTALL = r"""
set -e
mkdir -p {dir}/bin
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl >/dev/null; }
case "$(uname -m)" in aarch64|arm64) arch=aarch64 ;; *) arch=x86_64 ;; esac
triple="${arch}-unknown-linux-musl"
curl -fsSL "https://github.com/openai/codex/releases/download/rust-v{version}/codex-${triple}.tar.gz" | tar -xz -C {dir}/bin
mv "{dir}/bin/codex-${triple}" {bin}
chmod +x {bin}
"""


class CodexHarnessConfig(HarnessConfig):
    version: str = "0.144.5"
    """Codex release to install (the `rust-v<version>` GitHub release); pinned for reproducibility."""
    multi_agent: bool = False
    """Enable Codex's native multi-agent v2 tools."""


class CodexHarness(Harness[CodexHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = False  # TODO
    SUPPORTS_MCP = False  # TODO
    SUPPORTS_RESUME = True

    async def setup(self, runtime: Runtime) -> None:
        logger.info("codex: ensuring codex %s is installed", self.config.version)
        script = (
            INSTALL.replace("{version}", self.config.version)
            .replace("{dir}", CODEX_DIR)
            .replace("{bin}", CODEX_BIN)
        )
        ensure = shlex.quote(f"[ -x {CODEX_BIN} ] || ({script})")
        # Shared local runtimes may provision concurrently; only the first downloads.
        guarded = (
            f"mkdir -p {CODEX_DIR} && flock {CODEX_DIR}/install.lock sh -c {ensure}"
        )
        install = await runtime.run(["sh", "-c", guarded], {})
        if install.exit_code != 0:
            raise RuntimeError(f"codex install failed: {install.stderr.strip()[-500:]}")

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
        task = data
        if (
            task.system_prompt is not None
            and task.prompt is not None
            and not isinstance(task.prompt, str)
        ):
            system_prompt, prompt = task.system_prompt, task.prompt
        else:
            system_prompt, prompt = self.resolve_prompt(task)
        image_args: list[str] = []
        image_dir = f".vf-codex-images-{trace.id}"
        if prompt is not None and not isinstance(prompt, str):
            # Codex seeds one initial turn, so Messages system text joins its prompt.
            texts = [system_prompt] if system_prompt else []
            image_index = 0
            for message in prompt:
                if message.role not in ("system", "user"):
                    raise ValueError(
                        "codex exec only supports system and user initial messages"
                    )
                parts = (
                    [TextContentPart(text=message.content)]
                    if isinstance(message.content, str)
                    else message.content
                )
                for part in parts:
                    if isinstance(part, TextContentPart):
                        texts.append(part.text)
                        continue
                    metadata, separator, encoded = part.image_url.url.partition(",")
                    media_type, *parameters = metadata.removeprefix("data:").split(";")
                    if (
                        not separator
                        or not metadata.startswith("data:image/")
                        or not any(p.lower() == "base64" for p in parameters)
                    ):
                        raise ValueError(
                            "codex image prompts require base64 data:image URLs"
                        )
                    extension = re.sub(
                        r"[^a-zA-Z0-9]+", "_", media_type.removeprefix("image/")
                    ).strip("_")
                    path = f"{image_dir}/image_{image_index}.{extension or 'image'}"
                    await runtime.write(path, base64.b64decode(encoded))
                    image_args += ["-i", path]
                    image_index += 1
            prompt = "\n\n".join(texts)
        argv = [
            CODEX_BIN,
            "exec",
            *self._config_args(ctx, endpoint),
            *image_args,
            "--",
            prompt,
        ]
        try:
            return await runtime.run_program(
                argv, await self._env(trace, runtime, secret)
            )
        finally:
            if image_args:
                try:
                    await runtime.run(["rm", "-rf", image_dir], {})
                except Exception:
                    # Runtime teardown is the fallback; preserve the rollout result.
                    logger.warning(
                        "failed to clean up Codex prompt images", exc_info=True
                    )

    async def resume(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        messages,
    ) -> ProgramResult:
        """Native continuation: `codex exec resume --last` re-opens the session the
        previous segment recorded — codex's own context, compaction included — and
        takes the user's next turn. Nothing is replayed into its prompt (replaying
        our view of the conversation would fight codex's own session state, which is
        exactly why the default relaunch-resume is wrong for it). The rollout's
        per-trace `CODEX_HOME` (see `_env`) makes `--last` unambiguous even when a
        borrowed runtime hosts several rollouts."""
        texts: list[str] = []
        for message in messages:
            if message.role != "user":
                raise ValueError("codex resume takes user turns only")
            parts = (
                [TextContentPart(text=message.content)]
                if isinstance(message.content, str)
                else message.content
            )
            for part in parts:
                if not isinstance(part, TextContentPart):
                    raise ValueError(
                        "codex resume supports text user turns only (images go in "
                        "the opening prompt)"
                    )
                texts.append(part.text)
        if trace.num_turns == 0:
            return await self.launch(
                ctx,
                trace,
                runtime,
                endpoint,
                secret,
                mcp_urls,
                data.model_copy(update={"prompt": messages}),
            )
        argv = [
            CODEX_BIN,
            "exec",
            "resume",
            "--last",
            *self._config_args(ctx, endpoint),
            "--",
            "\n\n".join(texts),
        ]
        return await runtime.run_program(argv, await self._env(trace, runtime, secret))

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        home = self._home(trace)
        result = await runtime.run(["rm", "-rf", home], {})
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to clean up Codex home: {result.stderr.strip()[-500:]}"
            )

    @staticmethod
    def _home(trace: Trace) -> str:
        return f"/tmp/vf-codex-home-{trace.id}"

    async def _env(self, trace: Trace, runtime: Runtime, secret: str) -> dict[str, str]:
        # codex authenticates to the interception server with the session secret (its
        # provider api key) and posts Responses calls to `{endpoint}/responses`. The
        # per-trace CODEX_HOME scopes its recorded sessions to this rollout, so
        # `exec resume --last` continues THIS exchange and never a neighbor's.
        # codex refuses a home that doesn't exist, so make it before every segment.
        home = self._home(trace)
        await runtime.run(["mkdir", "-p", home], {})
        return {
            **self.config.resolved_env,
            KEY_VAR: secret,
            "CODEX_HOME": home,
        }

    def _config_args(self, ctx: ModelContext, endpoint: str) -> list[str]:
        # Values are Codex feature names such as `shell_tool`; Codex owns validation.
        # https://developers.openai.com/codex/config-reference#features
        tool_config = [
            arg
            for tool in self.config.disabled_tools or []
            for arg in ("--disable", tool)
        ]
        # `-c` values parse as TOML, falling back to a raw string (so the url / `responses`
        # come through literally); `requires_openai_auth=false` parses as a bool.
        return [
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            # Apps/plugins can flip on remotely and advertise definitions custom providers reject.
            "--disable",
            "apps",
            "--disable",
            "plugins",
            "--disable",
            "multi_agent",
            # Preserve any user-supplied multi-agent v2 tool and limit settings.
            "-c",
            f"features.multi_agent_v2.enabled={str(self.config.multi_agent).lower()}",
            "-m",
            ctx.model,
            "-c",
            f"model_provider={PROVIDER}",
            "-c",
            f"model_providers.{PROVIDER}.name={PROVIDER}",
            "-c",
            f"model_providers.{PROVIDER}.base_url={endpoint}",
            "-c",
            f"model_providers.{PROVIDER}.env_key={KEY_VAR}",
            "-c",
            f"model_providers.{PROVIDER}.wire_api=responses",
            "-c",
            f"model_providers.{PROVIDER}.requires_openai_auth=false",
            *tool_config,
        ]
