"""Terminus 2 harness configuration.

This adapts Harbor's Terminus 2 agent shape to the ComposableEnv ``Harness``
primitive: install a small sandbox-side runner and invoke it as ``run_command``.
The Harbor prompt templates are installed with the runner; ``system_prompt`` is
reserved for an optional extra first chat message.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

DEFAULT_INSTALL_DIR = "/opt/terminus_2"
DEFAULT_RUNNER_PATH = f"{DEFAULT_INSTALL_DIR}/runner.py"
DEFAULT_TEMPLATE_DIR = f"{DEFAULT_INSTALL_DIR}/templates"
DEFAULT_INSTRUCTION_PATH = "/task/instruction.md"
DEFAULT_SYSTEM_PROMPT_PATH = "/terminus_2/system_prompt.txt"
DEFAULT_LOG_DIR = "/logs/agent"
DEFAULT_LOG_PATH = f"{DEFAULT_LOG_DIR}/terminus_2.log"
DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
DEFAULT_SKILLS_PATH = "/task/skills"
DEFAULT_PARSER_NAME = "json"
DEFAULT_MAX_TURNS = 100
DEFAULT_TMUX_WIDTH = 160
DEFAULT_TMUX_HEIGHT = 40
DEFAULT_SUMMARIZATION_THRESHOLD_CHARS = 120_000
DEFAULT_SUMMARIZATION_KEEP_MESSAGES = 4
DEFAULT_JSON_PROMPT_TEMPLATE = (
    Path(__file__).parent / "templates" / "terminus-json-plain.txt"
).read_text()
DEFAULT_XML_PROMPT_TEMPLATE = (
    Path(__file__).parent / "templates" / "terminus-xml-plain.txt"
).read_text()
DEFAULT_TIMEOUT_TEMPLATE = (
    Path(__file__).parent / "templates" / "timeout.txt"
).read_text()
RUNNER_SOURCE = (Path(__file__).parent / "runner.py").read_text()
TERMINUS_2_CLI_PACKAGE = "embedded-terminus-2-runner"
TERMINUS_2_CLI_VERSION = "harbor-port-1"
TERMINUS_2_CLI_SHA256 = hashlib.sha256(
    "\n".join(
        [
            RUNNER_SOURCE,
            DEFAULT_JSON_PROMPT_TEMPLATE,
            DEFAULT_XML_PROMPT_TEMPLATE,
            DEFAULT_TIMEOUT_TEMPLATE,
        ]
    ).encode()
).hexdigest()
DEFAULT_RUNNER_SHA256 = TERMINUS_2_CLI_SHA256


def build_terminus_2_install_script(
    runner_path: str = DEFAULT_RUNNER_PATH,
    install_dir: str = DEFAULT_INSTALL_DIR,
    template_dir: str = DEFAULT_TEMPLATE_DIR,
    runner_sha256: str = DEFAULT_RUNNER_SHA256,
    install_tmux: bool = True,
) -> str:
    """Build the shell script that embeds the Terminus 2 runner and templates.

    This does not install Harbor as a package; it writes the dependency-light
    runner and prompt templates directly into the sandbox.
    """
    # Terminus is shipped as a tiny dependency-light runner plus prompt templates
    # instead of installing Harbor inside the sandbox.
    install_tools = ""
    if install_tmux:
        install_tools = """\
if ! command -v python3 >/dev/null 2>&1 || ! command -v tmux >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bash python3 tmux ca-certificates
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache bash python3 tmux ca-certificates
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y bash python3 tmux ca-certificates
  elif command -v yum >/dev/null 2>&1; then
    yum install -y bash python3 tmux ca-certificates
  elif command -v pacman >/dev/null 2>&1; then
    pacman -Sy --noconfirm bash python tmux ca-certificates
  else
    echo "Could not install python3/tmux: unsupported package manager" >&2
    exit 1
  fi
fi
"""

    quoted_install_dir = shlex.quote(install_dir)
    quoted_template_dir = shlex.quote(template_dir)
    runner_file = shlex.quote(runner_path)
    runner_python_path = json.dumps(runner_path)
    json_template_python_path = json.dumps(f"{template_dir}/terminus-json-plain.txt")
    xml_template_python_path = json.dumps(f"{template_dir}/terminus-xml-plain.txt")
    timeout_template_python_path = json.dumps(f"{template_dir}/timeout.txt")
    runner_source = RUNNER_SOURCE.removesuffix("\n")
    json_prompt_template = DEFAULT_JSON_PROMPT_TEMPLATE.removesuffix("\n")
    xml_prompt_template = DEFAULT_XML_PROMPT_TEMPLATE.removesuffix("\n")
    timeout_template = DEFAULT_TIMEOUT_TEMPLATE.removesuffix("\n")
    return f"""\
set -e
{install_tools}
mkdir -p {quoted_install_dir} {quoted_template_dir} {shlex.quote(DEFAULT_LOG_DIR)} /terminus_2
cat > {runner_file} <<'TERMINUS_2_RUNNER'
{runner_source}
TERMINUS_2_RUNNER
cat > {quoted_template_dir}/terminus-json-plain.txt <<'TERMINUS_2_JSON_TEMPLATE'
{json_prompt_template}
TERMINUS_2_JSON_TEMPLATE
cat > {quoted_template_dir}/terminus-xml-plain.txt <<'TERMINUS_2_XML_TEMPLATE'
{xml_prompt_template}
TERMINUS_2_XML_TEMPLATE
cat > {quoted_template_dir}/timeout.txt <<'TERMINUS_2_TIMEOUT_TEMPLATE'
{timeout_template}
TERMINUS_2_TIMEOUT_TEMPLATE
chmod +x {runner_file}
python3 - <<'TERMINUS_2_VERIFY'
import hashlib
from pathlib import Path

expected = "{runner_sha256}"
payload = "\\n".join(
    [
        Path({runner_python_path}).read_text(),
        Path({json_template_python_path}).read_text(),
        Path({xml_template_python_path}).read_text(),
        Path({timeout_template_python_path}).read_text(),
    ]
)
actual = hashlib.sha256(payload.encode()).hexdigest()
if actual != expected:
    raise SystemExit(f"Terminus 2 runner hash mismatch: {{actual}} != {{expected}}")
TERMINUS_2_VERIFY
"""


def build_terminus_2_run_command(
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    runner_path: str = DEFAULT_RUNNER_PATH,
    template_dir: str = DEFAULT_TEMPLATE_DIR,
    parser_name: str = DEFAULT_PARSER_NAME,
    max_turns: int = DEFAULT_MAX_TURNS,
    tmux_width: int = DEFAULT_TMUX_WIDTH,
    tmux_height: int = DEFAULT_TMUX_HEIGHT,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    skills_dir: str | None = None,
    enable_summarize: bool = True,
    summarization_threshold_chars: int = DEFAULT_SUMMARIZATION_THRESHOLD_CHARS,
    summarization_keep_messages: int = DEFAULT_SUMMARIZATION_KEEP_MESSAGES,
) -> str:
    """Build the shell command that delegates to the sandbox Terminus runner.

    MCP servers, skills, tmux dimensions, parser choice, and summarization
    thresholds are passed as runner CLI args.
    """
    # Keep the default workdir shell-expanded so AGENT_WORKDIR can be supplied by
    # the environment at runtime.
    if agent_workdir == DEFAULT_AGENT_WORKDIR:
        workdir_assignment = f"TERMINUS_2_AGENT_WORKDIR={DEFAULT_AGENT_WORKDIR}"
    else:
        workdir_assignment = f"TERMINUS_2_AGENT_WORKDIR={shlex.quote(agent_workdir)}"

    args = [
        "python3",
        shlex.quote(runner_path),
        "--instruction-path",
        shlex.quote(instruction_path),
        "--agent-workdir",
        '"$TERMINUS_2_AGENT_WORKDIR"',
        "--log-path",
        shlex.quote(log_path),
        "--mcp-servers-json",
        shlex.quote(json.dumps(mcp_servers or [])),
        "--template-dir",
        shlex.quote(template_dir),
        "--system-prompt-path",
        shlex.quote(system_prompt_path),
        "--parser-name",
        shlex.quote(parser_name),
        "--max-turns",
        str(max_turns),
        "--tmux-width",
        str(tmux_width),
        "--tmux-height",
        str(tmux_height),
        "--summarization-threshold-chars",
        str(summarization_threshold_chars),
        "--summarization-keep-messages",
        str(summarization_keep_messages),
    ]
    # MCP and skills are represented as prompt context for Terminus, unlike Codex
    # and Claude Code where the CLI consumes native config files.
    if skills_dir:
        args.extend(["--skills-dir", shlex.quote(skills_dir)])
    if not enable_summarize:
        args.append("--disable-summarize")

    script = f"""\
set -eo pipefail
export TERM=xterm-256color
{workdir_assignment}
mkdir -p {shlex.quote(log_path.rsplit("/", 1)[0])} "$TERMINUS_2_AGENT_WORKDIR"
cd "$TERMINUS_2_AGENT_WORKDIR"
{" ".join(args)} 2>&1 | tee -a {shlex.quote(log_path)}
"""
    return f"bash -lc {shlex.quote(script)}"


def terminus_2_harness(
    system_prompt: str | None = None,
    agent_workdir: str = DEFAULT_AGENT_WORKDIR,
    instruction_path: str = DEFAULT_INSTRUCTION_PATH,
    system_prompt_path: str = DEFAULT_SYSTEM_PROMPT_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    runner_path: str = DEFAULT_RUNNER_PATH,
    runner_sha256: str = DEFAULT_RUNNER_SHA256,
    parser_name: str = DEFAULT_PARSER_NAME,
    max_turns: int = DEFAULT_MAX_TURNS,
    tmux_width: int = DEFAULT_TMUX_WIDTH,
    tmux_height: int = DEFAULT_TMUX_HEIGHT,
    mcp_servers: list[dict[str, Any] | str] | None = None,
    skills_dir: str | None = None,
    enable_summarize: bool = True,
    summarization_threshold_chars: int = DEFAULT_SUMMARIZATION_THRESHOLD_CHARS,
    summarization_keep_messages: int = DEFAULT_SUMMARIZATION_KEEP_MESSAGES,
):
    """Create a Harness configured for a Terminus 2-style tmux agent."""
    from harnesses.base import make_native_harness

    # The factory keeps Harbor-style behavior behind the generic Harness
    # primitive: install a runner, write prompt files, then run one shell command.
    install_script = build_terminus_2_install_script(
        runner_path=runner_path,
        runner_sha256=runner_sha256,
    )

    return make_native_harness(
        build_run_command=build_terminus_2_run_command,
        run_kwargs={
            "agent_workdir": agent_workdir,
            "instruction_path": instruction_path,
            "system_prompt_path": system_prompt_path,
            "log_path": log_path,
            "runner_path": runner_path,
            "parser_name": parser_name,
            "max_turns": max_turns,
            "tmux_width": tmux_width,
            "tmux_height": tmux_height,
            "enable_summarize": enable_summarize,
            "summarization_threshold_chars": summarization_threshold_chars,
            "summarization_keep_messages": summarization_keep_messages,
        },
        install_script=install_script,
        system_prompt=system_prompt,
        instruction_path=instruction_path,
        system_prompt_path=system_prompt_path,
        log_path=log_path,
        default_skills_path=DEFAULT_SKILLS_PATH,
        mcp_servers=mcp_servers,
        skills_dir=skills_dir,
    )


TERMINUS_2_INSTALL_SCRIPT = build_terminus_2_install_script()
TERMINUS_2_CONFIG = {
    "install_script": TERMINUS_2_INSTALL_SCRIPT,
    "cli_package": TERMINUS_2_CLI_PACKAGE,
    "cli_version": TERMINUS_2_CLI_VERSION,
    "cli_sha256": TERMINUS_2_CLI_SHA256,
}
