"""Codex CLI harness configuration."""

from pathlib import PurePosixPath
import json
import shlex

from verifiers.envs.experimental.composable import Harness

DEFAULT_CODEX_VERSION = "0.132.0"
DEFAULT_CODEX_RELEASE_TAG = f"rust-v{DEFAULT_CODEX_VERSION}"
DEFAULT_CODEX_INSTALL_DIR = "/opt/codex"
DEFAULT_CODEX_BIN = f"{DEFAULT_CODEX_INSTALL_DIR}/codex"
DEFAULT_INSTRUCTION_PATH = "/codex/instruction.md"
DEFAULT_SYSTEM_PROMPT_PATH = "/codex/system.md"
DEFAULT_LOG_PATH = "/logs/agent/codex.log"
DEFAULT_GOAL_PATH = "/codex/goal.md"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_TIMEOUT_SECONDS = 3600


def build_codex_install_script(
    codex_version: str = DEFAULT_CODEX_VERSION,
    install_dir: str = DEFAULT_CODEX_INSTALL_DIR,
    codex_bin: str = DEFAULT_CODEX_BIN,
) -> str:
    """Build the shell script that installs the Codex CLI."""
    release_tag = f"rust-v{codex_version}"
    return f"""\
set -eu
mkdir -p {shlex.quote(install_dir)} /logs/agent /codex
case "$(uname -m)" in
  x86_64|amd64) CODEX_TARGET=x86_64-unknown-linux-musl ;;
  aarch64|arm64) CODEX_TARGET=aarch64-unknown-linux-musl ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac
CODEX_URL="https://github.com/openai/codex/releases/download/{release_tag}/codex-$CODEX_TARGET.tar.gz"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
python3 - "$CODEX_URL" "$tmpdir/codex.tar.gz" <<'PY'
import sys, time, urllib.request

url, dest = sys.argv[1], sys.argv[2]
delay = 1.0
for attempt in range(1, 6):
    try:
        urllib.request.urlretrieve(url, dest)
        break
    except Exception:
        if attempt == 5:
            raise
        time.sleep(delay)
        delay = min(delay * 2, 8.0)
PY
tar -xzf "$tmpdir/codex.tar.gz" -C "$tmpdir"
install -m 755 "$tmpdir"/codex-* {shlex.quote(codex_bin)}
ln -sf {shlex.quote(codex_bin)} /usr/local/bin/codex
codex --version
"""


def build_codex_run_command(
    *,
    goal_mode: bool = False,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    codex_bin: str = DEFAULT_CODEX_BIN,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    goal_path: str = DEFAULT_GOAL_PATH,
    goal_prompt: str | None = None,
    model_reasoning_effort: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Build the shell command that runs Codex against intercepted model traffic."""
    log_dir = str(PurePosixPath(log_path).parent)
    if agent_workdir == DEFAULT_AGENT_WORKDIR:
        workdir_assignment = f"CODEX_AGENT_WORKDIR={DEFAULT_AGENT_WORKDIR}"
    else:
        workdir_assignment = f"CODEX_AGENT_WORKDIR={shlex.quote(agent_workdir)}"

    if goal_mode:
        prompt = goal_prompt or f"/goal Read {goal_path} and complete the task."
        prompt_setup = (
            f"cat {shlex.quote(system_prompt_path)} {shlex.quote(instruction_path)} > "
            f"{shlex.quote(goal_path)}\n"
            f"CODEX_PROMPT={shlex.quote(prompt)}"
        )
        prompt_redirect = ""
        goals_arg = "  --enable goals \\\n"
    else:
        prompt_path = "/codex/prompt.md"
        prompt_setup = (
            f"cat {shlex.quote(system_prompt_path)} {shlex.quote(instruction_path)} > "
            f"{prompt_path}\nCODEX_PROMPT=-"
        )
        prompt_redirect = f" < {prompt_path}"
        goals_arg = ""

    reasoning_config = ""
    if model_reasoning_effort:
        config_value = f"model_reasoning_effort={json.dumps(model_reasoning_effort)}"
        reasoning_config = f"  -c {shlex.quote(config_value)} \\\n"

    extra = ""
    if extra_args:
        extra = "".join(f"  {shlex.quote(arg)} \\\n" for arg in extra_args)

    script = f"""\
set -euo pipefail
export HOME="${{HOME:-/root}}"
export CODEX_HOME="${{CODEX_HOME:-$HOME/.codex}}"
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export CODEX_UNSAFE_ALLOW_NO_SANDBOX=1
{workdir_assignment}
mkdir -p "$CODEX_HOME" /codex {shlex.quote(log_dir)} "$CODEX_AGENT_WORKDIR"
{prompt_setup}
cd "$CODEX_AGENT_WORKDIR"
timeout --kill-after=30s {int(timeout_seconds)}s {shlex.quote(codex_bin)} \\
{goals_arg}\
  --dangerously-bypass-approvals-and-sandbox \\
  -c 'web_search="disabled"' \\
  -c 'tools.web_search=false' \\
  -c 'model_provider="vf_proxy"' \\
  -c 'model_providers.vf_proxy.name="Verifiers Proxy"' \\
  -c "model_providers.vf_proxy.base_url=\\"$OPENAI_BASE_URL\\"" \\
  -c 'model_providers.vf_proxy.env_key="OPENAI_API_KEY"' \\
{reasoning_config}\
{extra}\
  --model model \\
  exec --ignore-user-config --ignore-rules --skip-git-repo-check --ephemeral --cd "$CODEX_AGENT_WORKDIR" "$CODEX_PROMPT"{prompt_redirect} \\
  2>&1 | tee -a {shlex.quote(log_path)}
"""
    return f"bash -lc {shlex.quote(script)}"


def codex_harness(
    system_prompt: str | None = None,
    goal_mode: bool = False,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    model_reasoning_effort: str | None = None,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    goal_path: str = DEFAULT_GOAL_PATH,
    goal_prompt: str | None = None,
    codex_version: str = DEFAULT_CODEX_VERSION,
    extra_args: list[str] | None = None,
) -> Harness:
    """Create a Harness configured for the Codex CLI."""
    return Harness(
        install_script=build_codex_install_script(codex_version=codex_version),
        install_timeout=300,
        run_command=build_codex_run_command(
            goal_mode=goal_mode,
            agent_workdir=agent_workdir,
            timeout_seconds=timeout_seconds,
            instruction_path=instruction_path,
            system_prompt_path=system_prompt_path,
            log_path=log_path,
            goal_path=goal_path,
            goal_prompt=goal_prompt,
            model_reasoning_effort=model_reasoning_effort,
            extra_args=extra_args,
        ),
        system_prompt=system_prompt,
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_path=log_path,
    )
