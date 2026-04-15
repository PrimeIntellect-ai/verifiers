"""RLM agent harness: install script, run command, and harness factory."""

from __future__ import annotations

import shlex

from verifiers.envs.experimental.composable import Harness

DEFAULT_RLM_REPO_URL = "github.com/PrimeIntellect-ai/rlm.git"
DEFAULT_RLM_TOOLS = "bash,edit"
DEFAULT_RLM_MAX_TURNS = 100
DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/task/append_to_system_prompt.txt"
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"


def build_install_script(
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    checkout_path: str = DEFAULT_RLM_CHECKOUT_PATH,
) -> str:
    script = f"""\
set -eo pipefail

apt-get update -qq
apt-get install -y -qq ca-certificates curl git > /dev/null 2>&1

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"
RLM_REPO_URL={shlex.quote(rlm_repo_url)}
RLM_CHECKOUT_PATH={shlex.quote(checkout_path)}

case "$RLM_REPO_URL" in
  http://*|https://*|git@*) CLONE_URL="$RLM_REPO_URL" ;;
  *) CLONE_URL="https://$RLM_REPO_URL" ;;
esac

if [ -n "${{GH_TOKEN:-}}" ] && printf '%s' "$CLONE_URL" | grep -q '^https://github.com/'; then
  CLONE_URL="${{CLONE_URL/https:\\/\\/github.com\\//https:\\/\\/${{GH_TOKEN}}@github.com/}}"
fi

rm -rf "$RLM_CHECKOUT_PATH"
git clone --depth 1 "$CLONE_URL" "$RLM_CHECKOUT_PATH"
if [ -d /task/rlm-skills ] && find /task/rlm-skills -mindepth 1 -maxdepth 1 | read -r _; then
  mkdir -p "$RLM_CHECKOUT_PATH/skills"
  cp -R /task/rlm-skills/. "$RLM_CHECKOUT_PATH/skills/"
fi
uv sync --project "$RLM_CHECKOUT_PATH" --all-packages
"""
    return f"bash -lc {shlex.quote(script)}"


def build_run_command(
    instruction_path: str = "/task/instruction.md",
    workdir: str = "/testbed",
    checkout_path: str = DEFAULT_RLM_CHECKOUT_PATH,
) -> str:
    script = f"""\
set -eo pipefail
export PATH="$HOME/.local/bin:$PATH"
export RLM_MODEL=$OPENAI_MODEL
export OPENAI_API_KEY=intercepted
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH)} 2>/dev/null || true)"
cd {workdir}
uv run --project {shlex.quote(checkout_path)} --all-packages rlm "$(cat {instruction_path})"
"""
    return f"bash -lc {shlex.quote(script)}"


def rlm_harness(
    workdir: str = "/testbed",
    instruction_path: str = "/task/instruction.md",
    rlm_repo_url: str = DEFAULT_RLM_REPO_URL,
    checkout_path: str = DEFAULT_RLM_CHECKOUT_PATH,
    append_to_system_prompt: str | None = None,
) -> Harness:
    return Harness(
        install_script=build_install_script(rlm_repo_url, checkout_path),
        run_command=build_run_command(instruction_path, workdir, checkout_path),
        system_prompt=append_to_system_prompt,
        system_prompt_path=DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
        instruction_path=instruction_path,
    )
