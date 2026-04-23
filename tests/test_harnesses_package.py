"""Coverage for optional harness package builders and sandbox adapters."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent.parent / "packages" / "harnesses"
# Import the local optional package directly, without installing it into the repo
# environment.
sys.path.insert(0, str(PACKAGE_DIR))

import harnesses  # noqa: E402  # ty: ignore[unresolved-import]
from verifiers.envs.composable_skills import TaskSkills  # noqa: E402
from verifiers.envs.composable_tools import TaskTools  # noqa: E402
from harnesses.terminus_2.runner import (  # noqa: E402  # ty: ignore[unresolved-import]
    build_augmented_instruction,
    parse_response,
    should_summarize,
)


def test_make_configurable_harness_rebuilds_tools_and_skills():
    def build_harness(mcp_servers, skills_dir):
        return harnesses.Harness(
            run_command=f"mcp={len(mcp_servers)} skills={skills_dir}",
            skills_path=skills_dir or "/task/skills",
        )

    harness = harnesses.make_configurable_harness(
        build_harness,
        mcp_servers=["base"],
        skills_dir="/base/skills",
    )

    configured = harness.with_tools(
        TaskTools(mcp_servers=[{"name": "task", "command": "server"}])
    ).with_skills(TaskSkills(skills_dir="/task/skills"))

    assert configured.run_command == "mcp=2 skills=/task/skills"


def test_make_native_harness_passes_capabilities_to_run_command_builder():
    def build_run_command(**kwargs):
        return json.dumps(
            {
                "mcp_servers": kwargs["mcp_servers"],
                "skills_dir": kwargs["skills_dir"],
                "instruction_path": kwargs["instruction_path"],
            }
        )

    harness = harnesses.make_native_harness(
        build_run_command=build_run_command,
        run_kwargs={"instruction_path": "/task/instruction.md"},
        instruction_path="/task/instruction.md",
        default_skills_path="/task/skills",
    )

    configured = harness.with_tools(
        TaskTools(mcp_servers=[{"name": "task", "command": "server"}])
    ).with_skills(TaskSkills(skills_dir="/task/skills"))
    run_config = json.loads(configured.run_command)

    assert run_config["mcp_servers"] == [{"name": "task", "command": "server"}]
    assert run_config["skills_dir"] == "/task/skills"
    assert run_config["instruction_path"] == "/task/instruction.md"


def test_opencode_config_renders_valid_json_after_shell_expansion():
    run_command = harnesses.build_opencode_run_command(
        disabled_tools=["webfetch", "question"],
        system_prompt_path="/opencode/system.txt",
        mcp_servers=[
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
                "env": {"ROOT": "/workspace"},
            },
            {
                "name": "sentry",
                "transport": "streamable-http",
                "url": "https://mcp.sentry.dev/mcp",
                "headers": {"Authorization": "Bearer token"},
            },
        ],
    )

    prefix = "cat > ~/.config/opencode/opencode.json << EOFCONFIG\n"
    suffix = "\nEOFCONFIG"
    config_block = run_command.split(prefix, 1)[1].split(suffix, 1)[0]

    # Render through bash because opencode.json intentionally contains shell-time
    # placeholders, including SCHEMA_DOLLAR for the literal "$schema" key.
    script = (
        f"OPENAI_BASE_URL=https://example.invalid "
        f"OPENAI_MODEL=intercepted/model "
        f"SCHEMA_DOLLAR='$' "
        f"bash -lc 'cat <<EOFCONFIG\n{config_block}\nEOFCONFIG'"
    )
    rendered = subprocess.run(
        script,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    config = json.loads(rendered)
    assert config["$schema"] == "https://opencode.ai/config.json"
    assert config["model"] == "intercepted/model"
    assert config["provider"]["intercepted"]["options"]["baseURL"] == (
        "https://example.invalid"
    )
    assert config["agent"]["build"]["prompt"] == "{file:/opencode/system.txt}"
    assert config["agent"]["build"]["tools"]["webfetch"] is False
    assert config["agent"]["build"]["tools"]["question"] is False
    assert config["compaction"] == {"auto": False, "prune": False}
    assert config["mcp"]["filesystem"]["type"] == "local"
    assert config["mcp"]["filesystem"]["command"] == ["node", "server.js"]
    assert config["mcp"]["filesystem"]["environment"]["ROOT"] == "/workspace"
    assert config["mcp"]["sentry"]["type"] == "remote"
    assert config["mcp"]["sentry"]["url"] == "https://mcp.sentry.dev/mcp"
    assert config["mcp"]["sentry"]["headers"]["Authorization"] == "Bearer token"


def test_opencode_mcp_config_renders_json():
    config = harnesses.build_opencode_mcp_config(
        [
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "uvx mcp-server",
                "env": {"ROOT": "/workspace"},
                "timeout": 7000,
            },
            {
                "name": "sentry",
                "transport": "http",
                "url": "https://mcp.sentry.dev/mcp",
                "http_headers": {"Authorization": "Bearer token"},
                "enabled": False,
            },
        ]
    )

    parsed = json.loads(config)
    filesystem = parsed["filesystem"]
    assert filesystem["type"] == "local"
    assert filesystem["command"] == ["uvx", "mcp-server"]
    assert filesystem["environment"]["ROOT"] == "/workspace"
    assert filesystem["timeout"] == 7000
    sentry = parsed["sentry"]
    assert sentry["type"] == "remote"
    assert sentry["url"] == "https://mcp.sentry.dev/mcp"
    assert sentry["headers"]["Authorization"] == "Bearer token"
    assert sentry["enabled"] is False


def test_opencode_harness_factory_returns_composable_harness():
    opencode = harnesses.opencode_harness(
        system_prompt="base",
        task_system_prompt="task",
        disabled_tools=["webfetch"],
        allow_git=True,
        mcp_servers=[
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
            }
        ],
    )

    assert opencode.system_prompt == "base\ntask"
    assert opencode.instruction_path == "/opencode/prompt.txt"
    assert opencode.system_prompt_path == "/opencode/system.txt"
    assert opencode.log_path == "/opencode/logs.txt"
    assert "sha256sum -c -" in opencode.install_script
    assert "export ALLOW_GIT=1" in opencode.run_command
    assert '"webfetch": false' in opencode.run_command
    assert '"mcp"' in opencode.run_command
    assert '"filesystem"' in opencode.run_command


def test_opencode_harness_registers_task_mcp_servers():
    opencode = harnesses.opencode_harness(
        mcp_servers=[
            {
                "name": "base",
                "transport": "stdio",
                "command": "base-server",
            }
        ]
    )

    configured = opencode.with_tools(
        TaskTools(
            mcp_servers=[
                {
                    "name": "task",
                    "transport": "streamable-http",
                    "url": "http://127.0.0.1:8000/mcp",
                }
            ]
        )
    )

    assert '"base"' in configured.run_command
    assert '"base-server"' in configured.run_command
    assert '"task"' in configured.run_command
    assert '"url": "http://127.0.0.1:8000/mcp"' in configured.run_command
    assert '"task"' not in opencode.run_command


def test_opencode_harness_registers_task_skills():
    opencode = harnesses.opencode_harness()

    configured = opencode.with_skills(TaskSkills(skills_dir="/task/skills"))

    assert opencode.skills_path == "/task/skills"
    assert "cp -r /task/skills/* ~/.config/opencode/skills/" in (configured.run_command)
    assert '"skill": false' not in configured.run_command
    assert "cp -r /task/skills/* ~/.config/opencode/skills/" not in (
        opencode.run_command
    )


def test_harness_integrity_configs_expose_install_metadata():
    configs = [
        harnesses.OPENCODE_CONFIG,
        harnesses.CODEX_CONFIG,
        harnesses.CLAUDE_CODE_CONFIG,
        harnesses.OPENCLAW_CONFIG,
        harnesses.MINI_SWE_AGENT_CONFIG,
        harnesses.PI_MONO_CONFIG,
        harnesses.TERMINUS_2_CONFIG,
    ]

    for config in configs:
        assert config["install_script"]
        assert config["cli_package"]
        assert config["cli_version"]
        assert len(config["cli_sha256"]) == 64
        int(config["cli_sha256"], 16)
        assert config["cli_sha256"] in config["install_script"]
        if "node_installer_sha256" in config:
            assert len(config["node_installer_sha256"]) == 64
            assert config["node_installer_sha256"] in config["install_script"]
            assert "| bash" not in config["install_script"]


def test_harnesses_opencode_import():
    from harnesses.opencode import opencode_harness  # ty: ignore[unresolved-import]

    assert opencode_harness(system_prompt=None).system_prompt is None


def test_codex_mcp_config_renders_toml():
    config = harnesses.build_codex_mcp_config(
        [
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
                "env": {"ROOT": "/workspace"},
                "tools": {"read": "approve"},
            },
            {
                "name": "sentry",
                "transport": "http",
                "url": "https://mcp.sentry.dev/mcp",
                "enabled": False,
            },
        ]
    )

    parsed = tomllib.loads(config)
    filesystem = parsed["mcp_servers"]["filesystem"]
    assert filesystem["command"] == "node"
    assert filesystem["args"] == ["server.js"]
    assert filesystem["env"]["ROOT"] == "/workspace"
    assert filesystem["tools"]["read"]["approval_mode"] == "approve"
    assert parsed["mcp_servers"]["sentry"]["url"] == "https://mcp.sentry.dev/mcp"
    assert parsed["mcp_servers"]["sentry"]["enabled"] is False


def test_codex_harness_factory_returns_composable_harness():
    codex = harnesses.codex_harness(
        system_prompt="custom codex system",
        package_version="0.1.0",
        package_sha256="0" * 64,
        reasoning_effort="medium",
        reasoning_summary="auto",
        mcp_servers=[
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
            }
        ],
        skills_dir="/skills",
    )

    assert codex.system_prompt == "custom codex system"
    assert codex.instruction_path == "/codex/prompt.txt"
    assert codex.system_prompt_path == "/codex/system.txt"
    assert codex.log_path == "/logs/agent/codex.txt"
    assert "CODEX_NPM_PACKAGE=@openai/codex@0.1.0" in codex.install_script
    assert 'npm pack "${CODEX_NPM_PACKAGE}"' in codex.install_script
    assert "sha256sum -c -" in codex.install_script
    assert 'npm install -g "${CODEX_NPM_TARBALL}"' in codex.install_script
    assert "nvm install 22" in codex.install_script
    assert "CODEX_AGENT_WORKDIR=${AGENT_WORKDIR:-/app}" in codex.run_command
    assert 'CODEX_MODEL="${OPENAI_MODEL##*/}"' in codex.run_command
    assert "codex exec" in codex.run_command
    assert "--dangerously-bypass-approvals-and-sandbox" in codex.run_command
    assert "--skip-git-repo-check" in codex.run_command
    assert "--json" in codex.run_command
    assert "--enable" in codex.run_command
    assert "unified_exec" in codex.run_command
    assert '--model \\\n  "$CODEX_MODEL"' in codex.run_command
    assert "model_reasoning_effort=medium" in codex.run_command
    assert "model_reasoning_summary=auto" in codex.run_command
    assert "[mcp_servers.filesystem]" in codex.run_command
    assert 'args = ["server.js"]' in codex.run_command
    assert 'cp -r /skills/* "$CODEX_HOME/skills/"' in codex.run_command


def test_codex_harness_registers_task_mcp_servers():
    codex = harnesses.codex_harness(
        mcp_servers=[
            {
                "name": "base",
                "transport": "stdio",
                "command": "base-server",
            }
        ]
    )

    configured = codex.with_tools(
        TaskTools(
            mcp_servers=[
                {
                    "name": "task",
                    "transport": "streamable-http",
                    "url": "http://127.0.0.1:8000/mcp",
                }
            ]
        )
    )

    assert "[mcp_servers.base]" in configured.run_command
    assert 'command = "base-server"' in configured.run_command
    assert "[mcp_servers.task]" in configured.run_command
    assert 'url = "http://127.0.0.1:8000/mcp"' in configured.run_command
    assert "[mcp_servers.task]" not in codex.run_command


def test_codex_harness_registers_task_skills():
    codex = harnesses.codex_harness()

    configured = codex.with_skills(TaskSkills(skills_dir="/task/skills"))

    assert codex.skills_path == "/task/skills"
    assert 'cp -r /task/skills/* "$CODEX_HOME/skills/"' in configured.run_command
    assert 'cp -r /task/skills/* "$CODEX_HOME/skills/"' not in codex.run_command


def test_codex_custom_paths_and_import():
    from harnesses.codex import codex_harness  # ty: ignore[unresolved-import]

    codex = codex_harness(
        system_prompt=None,
        agent_workdir="/workspace",
        instruction_path="/task/instruction.md",
        log_path="/tmp/codex.log",
        last_message_path="/tmp/codex-last-message.txt",
        codex_home="/tmp/codex-home",
        sandbox_bypass=False,
        skip_git_repo_check=False,
        json_output=False,
        enable_unified_exec=False,
    )

    assert codex.system_prompt is None
    assert codex.instruction_path == "/task/instruction.md"
    assert "CODEX_AGENT_WORKDIR=/workspace" in codex.run_command
    assert "export CODEX_HOME=/tmp/codex-home" in codex.run_command
    assert (
        "--output-last-message \\\n  /tmp/codex-last-message.txt" in codex.run_command
    )
    assert "tee -a /tmp/codex.log" in codex.run_command
    assert "--dangerously-bypass-approvals-and-sandbox" not in codex.run_command
    assert "--skip-git-repo-check" not in codex.run_command


def test_claude_code_mcp_config_renders_json():
    config = harnesses.build_claude_code_mcp_config(
        [
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
                "env": {"ROOT": "/workspace"},
            },
            {
                "name": "sentry",
                "transport": "streamable-http",
                "url": "https://mcp.sentry.dev/mcp",
                "headers": {"Authorization": "Bearer token"},
            },
        ]
    )

    parsed = json.loads(config)
    filesystem = parsed["mcpServers"]["filesystem"]
    assert filesystem["type"] == "stdio"
    assert filesystem["command"] == "node"
    assert filesystem["args"] == ["server.js"]
    assert filesystem["env"]["ROOT"] == "/workspace"
    sentry = parsed["mcpServers"]["sentry"]
    assert sentry["type"] == "http"
    assert sentry["url"] == "https://mcp.sentry.dev/mcp"
    assert sentry["headers"]["Authorization"] == "Bearer token"


def test_claude_code_harness_factory_returns_composable_harness():
    claude = harnesses.claude_code_harness(
        system_prompt="custom claude system",
        package_version="0.1.0",
        package_sha256="0" * 64,
        max_turns=5,
        reasoning_effort="high",
        max_thinking_tokens=8000,
        allowed_tools=["Bash", "Read"],
        disallowed_tools="WebFetch",
        mcp_servers=[
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
            }
        ],
        skills_dir="/skills",
        memory_dir="/memory",
    )

    assert claude.system_prompt == "custom claude system"
    assert claude.instruction_path == "/claude-code/prompt.txt"
    assert claude.system_prompt_path == "/claude-code/system.txt"
    assert claude.log_path == "/logs/agent/claude-code.txt"
    assert (
        "CLAUDE_CODE_NPM_PACKAGE=@anthropic-ai/claude-code@0.1.0"
        in claude.install_script
    )
    assert 'npm pack "${CLAUDE_CODE_NPM_PACKAGE}"' in claude.install_script
    assert "sha256sum -c -" in claude.install_script
    assert 'npm install -g "${CLAUDE_CODE_NPM_TARBALL}"' in claude.install_script
    assert "nvm install 22" in claude.install_script
    assert "CLAUDE_CONFIG_DIR=/logs/agent/claude-code-sessions" in claude.run_command
    assert "CLAUDE_CODE_AGENT_WORKDIR=${AGENT_WORKDIR:-/app}" in claude.run_command
    assert "ANTHROPIC_API_KEY" in claude.run_command
    assert "CLAUDE_ARGS+=(--output-format=stream-json)" in claude.run_command
    assert "CLAUDE_ARGS+=(--permission-mode bypassPermissions)" in claude.run_command
    assert "CLAUDE_ARGS+=(--max-turns 5)" in claude.run_command
    assert "CLAUDE_ARGS+=(--effort high)" in claude.run_command
    assert "CLAUDE_ARGS+=(--allowedTools Bash)" in claude.run_command
    assert "CLAUDE_ARGS+=(--allowedTools Read)" in claude.run_command
    assert "CLAUDE_ARGS+=(--disallowedTools WebFetch)" in claude.run_command
    assert "MAX_THINKING_TOKENS=8000" in claude.run_command
    assert '"mcpServers"' in claude.run_command
    assert '"filesystem"' in claude.run_command
    assert 'cp -r /skills/* "$CLAUDE_CONFIG_DIR/skills/"' in claude.run_command
    assert 'cp -r /memory/* "$CLAUDE_CONFIG_DIR/projects/-app/memory/"' in (
        claude.run_command
    )
    assert 'claude "${CLAUDE_ARGS[@]}" -- "$CLAUDE_CODE_PROMPT"' in claude.run_command


def test_claude_code_harness_registers_task_mcp_servers():
    claude = harnesses.claude_code_harness(
        mcp_servers=[
            {
                "name": "base",
                "transport": "stdio",
                "command": "base-server",
            }
        ]
    )

    configured = claude.with_tools(
        TaskTools(
            mcp_servers=[
                {
                    "name": "task",
                    "transport": "streamable-http",
                    "url": "http://127.0.0.1:8000/mcp",
                }
            ]
        )
    )

    assert '"base"' in configured.run_command
    assert '"command": "base-server"' in configured.run_command
    assert '"task"' in configured.run_command
    assert '"url": "http://127.0.0.1:8000/mcp"' in configured.run_command
    assert '"task"' not in claude.run_command


def test_claude_code_harness_registers_task_skills():
    claude = harnesses.claude_code_harness()

    configured = claude.with_skills(TaskSkills(skills_dir="/task/skills"))

    assert claude.skills_path == "/task/skills"
    assert 'cp -r /task/skills/* "$CLAUDE_CONFIG_DIR/skills/"' in (
        configured.run_command
    )
    assert 'cp -r /task/skills/* "$CLAUDE_CONFIG_DIR/skills/"' not in (
        claude.run_command
    )


def test_claude_code_custom_paths_and_import():
    from harnesses.claude_code import (  # ty: ignore[unresolved-import]
        claude_code_harness,
    )

    claude = claude_code_harness(
        system_prompt=None,
        agent_workdir="/workspace",
        instruction_path="/task/instruction.md",
        log_path="/tmp/claude.log",
        config_dir="/tmp/claude-config",
        output_format="json",
        permission_mode=None,
    )

    assert claude.system_prompt is None
    assert claude.instruction_path == "/task/instruction.md"
    assert "CLAUDE_CODE_AGENT_WORKDIR=/workspace" in claude.run_command
    assert "CLAUDE_CONFIG_DIR=/tmp/claude-config" in claude.run_command
    assert "CLAUDE_ARGS+=(--output-format=json)" in claude.run_command
    assert "tee -a /tmp/claude.log" in claude.run_command
    assert "--permission-mode" not in claude.run_command


def test_openclaw_mcp_config_renders_json():
    config = harnesses.build_openclaw_mcp_config(
        [
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
                "env": {"ROOT": "/workspace"},
            },
            {
                "name": "sentry",
                "transport": "http",
                "url": "https://mcp.sentry.dev/mcp",
                "headers": {"Authorization": "Bearer token"},
                "connectionTimeoutMs": 5000,
            },
        ]
    )

    parsed = json.loads(config)
    filesystem = parsed["servers"]["filesystem"]
    assert filesystem["command"] == "node"
    assert filesystem["args"] == ["server.js"]
    assert filesystem["env"]["ROOT"] == "/workspace"
    sentry = parsed["servers"]["sentry"]
    assert sentry["transport"] == "streamable-http"
    assert sentry["url"] == "https://mcp.sentry.dev/mcp"
    assert sentry["headers"]["Authorization"] == "Bearer token"
    assert sentry["connectionTimeoutMs"] == 5000


def test_openclaw_config_renders_valid_json_after_shell_expansion():
    config = harnesses.build_openclaw_config(
        provider_name="vf",
        model_id="$OPENCLAW_MODEL",
        base_url="$OPENAI_BASE_URL",
        api_key="OPENAI_API_KEY",
        reasoning=True,
        compat={"supportsStore": False},
    )

    # OpenClaw config intentionally contains shell-time placeholders so the
    # rollout can inject the intercepted endpoint and model.
    script = (
        "OPENAI_BASE_URL=https://example.invalid/v1 "
        "OPENCLAW_MODEL=gpt-test "
        "OPENCLAW_AGENT_WORKDIR=/workspace "
        f"bash -lc 'cat <<OPENCLAW_CONFIG_JSON\n{config}\nOPENCLAW_CONFIG_JSON'"
    )
    rendered = subprocess.run(
        script,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    parsed = json.loads(rendered)
    provider = parsed["models"]["providers"]["vf"]
    assert provider["baseUrl"] == "https://example.invalid/v1"
    assert provider["apiKey"] == "OPENAI_API_KEY"
    assert provider["models"][0]["id"] == "gpt-test"
    assert provider["models"][0]["reasoning"] is True
    assert provider["models"][0]["compat"]["supportsStore"] is False
    assert parsed["agents"]["defaults"]["model"]["primary"] == "vf/gpt-test"
    assert parsed["agents"]["defaults"]["workspace"] == "/workspace"


def test_openclaw_harness_factory_returns_composable_harness():
    openclaw = harnesses.openclaw_harness(
        system_prompt="custom openclaw system",
        package_version="0.1.0",
        package_sha256="0" * 64,
        reasoning=True,
        thinking="high",
        timeout_seconds=30,
        skills_dir="/skills",
        append_system_prompt="extra system",
        mcp_servers=[
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
            }
        ],
        extra_args=["--verbose"],
    )

    assert openclaw.system_prompt == "custom openclaw system"
    assert openclaw.instruction_path == "/openclaw/prompt.txt"
    assert openclaw.system_prompt_path == "/openclaw/system.txt"
    assert openclaw.log_path == "/logs/agent/openclaw.txt"
    assert "OPENCLAW_NPM_PACKAGE=openclaw@0.1.0" in openclaw.install_script
    assert 'npm pack "${OPENCLAW_NPM_PACKAGE}"' in openclaw.install_script
    assert "sha256sum -c -" in openclaw.install_script
    assert 'npm install -g "${OPENCLAW_NPM_TARBALL}"' in openclaw.install_script
    assert "nvm install 22" in openclaw.install_script
    assert "OPENCLAW_STATE_DIR=/logs/agent/openclaw-state" in openclaw.run_command
    assert "OPENCLAW_AGENT_WORKDIR=${AGENT_WORKDIR:-/app}" in openclaw.run_command
    assert 'OPENCLAW_MODEL="$OPENAI_MODEL"' in openclaw.run_command
    assert 'cat > "$OPENCLAW_CONFIG_PATH"' in openclaw.run_command
    assert '"primary": "intercepted/$OPENCLAW_MODEL"' in openclaw.run_command
    assert '"api": "openai-completions"' in openclaw.run_command
    assert "OPENCLAW_ARGS+=(agent)" in openclaw.run_command
    assert "OPENCLAW_ARGS+=(--local)" in openclaw.run_command
    assert "OPENCLAW_ARGS+=(--agent main)" in openclaw.run_command
    assert "OPENCLAW_ARGS+=(--thinking high)" in openclaw.run_command
    assert "OPENCLAW_ARGS+=(--timeout 30)" in openclaw.run_command
    assert "OPENCLAW_ARGS+=(--json)" in openclaw.run_command
    assert "OPENCLAW_ARGS+=(--verbose)" in openclaw.run_command
    assert '"mcp"' in openclaw.run_command
    assert '"filesystem"' in openclaw.run_command
    assert 'cp -r /skills/* "$OPENCLAW_AGENT_WORKDIR/skills/"' in (openclaw.run_command)
    assert 'openclaw "${OPENCLAW_ARGS[@]}"' in openclaw.run_command


def test_openclaw_harness_registers_task_mcp_servers():
    openclaw = harnesses.openclaw_harness(
        mcp_servers=[
            {
                "name": "base",
                "transport": "stdio",
                "command": "base-server",
            }
        ]
    )

    configured = openclaw.with_tools(
        TaskTools(
            mcp_servers=[
                {
                    "name": "task",
                    "transport": "streamable-http",
                    "url": "http://127.0.0.1:8000/mcp",
                }
            ]
        )
    )

    assert '"base"' in configured.run_command
    assert '"command": "base-server"' in configured.run_command
    assert '"task"' in configured.run_command
    assert '"url": "http://127.0.0.1:8000/mcp"' in configured.run_command
    assert '"task"' not in openclaw.run_command


def test_openclaw_harness_registers_task_skills():
    openclaw = harnesses.openclaw_harness()

    configured = openclaw.with_skills(TaskSkills(skills_dir="/task/skills"))

    assert openclaw.skills_path == "/task/skills"
    assert 'cp -r /task/skills/* "$OPENCLAW_AGENT_WORKDIR/skills/"' in (
        configured.run_command
    )
    assert 'cp -r /task/skills/* "$OPENCLAW_AGENT_WORKDIR/skills/"' not in (
        openclaw.run_command
    )


def test_openclaw_custom_paths_and_import():
    from harnesses.openclaw import openclaw_harness  # ty: ignore[unresolved-import]

    openclaw = openclaw_harness(
        system_prompt=None,
        agent_workdir="/workspace",
        instruction_path="/task/instruction.md",
        log_path="/tmp/openclaw.log",
        state_dir="/tmp/openclaw-state",
        agent_id="worker",
        provider_name="custom",
        model_id="custom-model",
        thinking=None,
        json_output=False,
    )

    assert openclaw.system_prompt is None
    assert openclaw.instruction_path == "/task/instruction.md"
    assert "OPENCLAW_AGENT_WORKDIR=/workspace" in openclaw.run_command
    assert "OPENCLAW_STATE_DIR=/tmp/openclaw-state" in openclaw.run_command
    assert 'OPENCLAW_AGENT_DIR="$OPENCLAW_STATE_DIR/agents/worker/agent"' in (
        openclaw.run_command
    )
    assert "OPENCLAW_MODEL=custom-model" in openclaw.run_command
    assert '"primary": "custom/$OPENCLAW_MODEL"' in openclaw.run_command
    assert "OPENCLAW_ARGS+=(--agent worker)" in openclaw.run_command
    assert "tee -a /tmp/openclaw.log" in openclaw.run_command
    assert "--thinking" not in openclaw.run_command
    assert "--json" not in openclaw.run_command


def test_mini_swe_agent_harness_factory_returns_composable_harness():
    mini = harnesses.mini_swe_agent_harness(
        system_prompt="custom mini system",
        environment_timeout=60,
        extra_config_specs=["agent.step_limit=10"],
    )

    assert mini.system_prompt == "custom mini system"
    assert mini.instruction_path == "/mini-swe-agent/prompt.txt"
    assert mini.system_prompt_path == "/mini-swe-agent/system.txt"
    assert mini.log_path == "/logs/agent/mini-swe-agent.log"
    assert "mini-swe-agent==2.2.8" in mini.install_script
    assert "UV_UNMANAGED_INSTALL=/opt/mini-swe-agent/uv-bin" in mini.install_script
    assert "uv --quiet venv --python 3.11 --seed /opt/mini-swe-agent/venv" in (
        mini.install_script
    )
    assert "pip download --quiet --only-binary=:all: --no-deps" in (mini.install_script)
    assert "sha256sum -c -" in mini.install_script
    assert "cat > /opt/mini-swe-agent/openai_text_model.py" in mini.install_script
    assert "MSWEA_CONFIGURED=true" in mini.run_command
    assert 'PYTHONPATH=/opt/mini-swe-agent:"${PYTHONPATH:-}"' in mini.run_command
    assert '--model "$OPENAI_MODEL"' in mini.run_command
    assert "-c mini_textbased" in mini.run_command
    assert "model.model_class=openai_text_model.OpenAITextModel" in mini.run_command
    assert "litellm" not in mini.run_command
    assert "environment.timeout=60" in mini.run_command
    assert "agent.step_limit=10" in mini.run_command
    assert "MINI_SWE_AGENT_WORKDIR=${AGENT_WORKDIR:-/app}" in mini.run_command


def test_mini_swe_agent_custom_paths_and_import():
    from harnesses.mini_swe_agent import (  # ty: ignore[unresolved-import]
        mini_swe_agent_harness,
    )

    mini = mini_swe_agent_harness(
        system_prompt=None,
        agent_workdir="/workspace",
        instruction_path="/task/instruction.md",
        log_path="/tmp/mini.log",
        trajectory_path="/tmp/mini.traj.json",
    )

    assert mini.system_prompt is None
    assert mini.instruction_path == "/task/instruction.md"
    assert "MINI_SWE_AGENT_WORKDIR=/workspace" in mini.run_command
    assert "--output /tmp/mini.traj.json" in mini.run_command
    assert "tee -a /tmp/mini.log" in mini.run_command


def test_pi_mono_models_config_renders_json():
    config = harnesses.build_pi_mono_models_config(
        provider_name="vf",
        model_id="openai/gpt-test",
        base_url="https://example.invalid/v1",
        api_key="OPENAI_API_KEY",
        reasoning=True,
        compat={"supportsStore": False},
    )

    parsed = json.loads(config)
    provider = parsed["providers"]["vf"]
    assert provider["baseUrl"] == "https://example.invalid/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == "OPENAI_API_KEY"
    assert provider["models"][0]["id"] == "openai/gpt-test"
    assert provider["models"][0]["reasoning"] is True
    assert provider["models"][0]["compat"]["supportsStore"] is False


def test_pi_mono_harness_factory_returns_composable_harness():
    pi = harnesses.pi_mono_harness(
        system_prompt="custom pi system",
        package_version="0.1.0",
        package_sha256="0" * 64,
        reasoning=True,
        thinking="high",
        skills_dir="/skills",
        append_system_prompt="extra system",
        extra_args=["--verbose"],
    )

    assert pi.system_prompt == "custom pi system"
    assert pi.instruction_path == "/pi-mono/prompt.txt"
    assert pi.system_prompt_path == "/pi-mono/system.txt"
    assert pi.log_path == "/logs/agent/pi-mono.txt"
    assert (
        "PI_MONO_NPM_PACKAGE=@mariozechner/pi-coding-agent@0.1.0" in pi.install_script
    )
    assert 'npm pack "${PI_MONO_NPM_PACKAGE}"' in pi.install_script
    assert "sha256sum -c -" in pi.install_script
    assert 'npm install -g "${PI_MONO_NPM_TARBALL}"' in pi.install_script
    assert "nvm install 22" in pi.install_script
    assert "PI_CODING_AGENT_DIR=/logs/agent/pi-mono-agent" in pi.run_command
    assert "PI_MONO_AGENT_WORKDIR=${AGENT_WORKDIR:-/app}" in pi.run_command
    assert 'PI_MONO_MODEL="$OPENAI_MODEL"' in pi.run_command
    assert 'cat > "$PI_CODING_AGENT_DIR/models.json"' in pi.run_command
    assert '"id": "$PI_MONO_MODEL"' in pi.run_command
    assert '"api": "openai-completions"' in pi.run_command
    assert "PI_MONO_ARGS+=(--provider intercepted)" in pi.run_command
    assert 'PI_MONO_ARGS+=(--model "$PI_MONO_MODEL")' in pi.run_command
    assert "PI_MONO_ARGS+=(--mode json)" in pi.run_command
    assert "PI_MONO_ARGS+=(--thinking high)" in pi.run_command
    assert "PI_MONO_ARGS+=(--tools read,bash,edit,write,grep,find,ls)" in pi.run_command
    assert "PI_MONO_ARGS+=(--no-session)" in pi.run_command
    assert "PI_MONO_ARGS+=(--offline)" in pi.run_command
    assert "export PI_OFFLINE=1" in pi.run_command
    assert 'cp -r /skills/* "$PI_CODING_AGENT_DIR/skills/"' in pi.run_command
    assert 'PI_MONO_ARGS+=(--skill "$PI_CODING_AGENT_DIR/skills")' in pi.run_command
    assert 'PI_MONO_PROMPT" | pi "${PI_MONO_ARGS[@]}"' in pi.run_command


def test_pi_mono_harness_registers_task_skills():
    pi = harnesses.pi_mono_harness()

    configured = pi.with_skills(TaskSkills(skills_dir="/task/skills"))

    assert pi.skills_path == "/task/skills"
    assert 'cp -r /task/skills/* "$PI_CODING_AGENT_DIR/skills/"' in (
        configured.run_command
    )
    assert 'cp -r /task/skills/* "$PI_CODING_AGENT_DIR/skills/"' not in (pi.run_command)


def test_pi_mono_custom_paths_and_import():
    from harnesses.pi_mono import pi_mono_harness  # ty: ignore[unresolved-import]

    pi = pi_mono_harness(
        system_prompt=None,
        agent_workdir="/workspace",
        instruction_path="/task/instruction.md",
        log_path="/tmp/pi.log",
        agent_dir="/tmp/pi-agent",
        provider_name="custom",
        model_id="custom-model",
        output_mode="text",
        tools="read,bash",
        no_extensions=False,
        no_prompt_templates=False,
        no_themes=False,
        no_session=False,
        offline=False,
    )

    assert pi.system_prompt is None
    assert pi.instruction_path == "/task/instruction.md"
    assert "PI_MONO_AGENT_WORKDIR=/workspace" in pi.run_command
    assert "PI_CODING_AGENT_DIR=/tmp/pi-agent" in pi.run_command
    assert "PI_MONO_MODEL=custom-model" in pi.run_command
    assert "PI_MONO_ARGS+=(--provider custom)" in pi.run_command
    assert "PI_MONO_ARGS+=(--print)" in pi.run_command
    assert "PI_MONO_ARGS+=(--tools read,bash)" in pi.run_command
    assert "tee -a /tmp/pi.log" in pi.run_command
    assert "--mode json" not in pi.run_command
    assert "--no-extensions" not in pi.run_command
    assert "--no-prompt-templates" not in pi.run_command
    assert "--no-themes" not in pi.run_command
    assert "--no-session" not in pi.run_command
    assert "--offline" not in pi.run_command
    assert "PI_OFFLINE=1" not in pi.run_command


def test_terminus_2_harness_factory_returns_composable_harness():
    terminus = harnesses.terminus_2_harness(max_turns=7, parser_name="xml")

    assert terminus.system_prompt is None
    assert terminus.instruction_path == "/task/instruction.md"
    assert terminus.system_prompt_path == "/terminus_2/system_prompt.txt"
    assert terminus.log_path == "/logs/agent/terminus_2.log"
    assert "cat > /opt/terminus_2/runner.py" in terminus.install_script
    assert "terminus-json-plain.txt" in terminus.install_script
    assert "TERMINUS_2_VERIFY" in terminus.install_script
    assert "Terminus 2 runner hash mismatch" in terminus.install_script
    assert "--parser-name xml" in terminus.run_command
    assert "--max-turns 7" in terminus.run_command
    assert "TERMINUS_2_AGENT_WORKDIR=${AGENT_WORKDIR:-/app}" in terminus.run_command


def test_terminus_2_custom_system_prompt_is_optional_extra_message():
    terminus = harnesses.terminus_2_harness(system_prompt="extra system prompt")

    assert terminus.system_prompt == "extra system prompt"
    assert "terminus-json-plain.txt" not in terminus.system_prompt
    assert "--system-prompt-path /terminus_2/system_prompt.txt" in terminus.run_command


def test_terminus_2_harness_passes_mcp_skills_and_summarization_options():
    terminus = harnesses.terminus_2_harness(
        mcp_servers=[
            {
                "name": "filesystem",
                "transport": "stdio",
                "command": "node",
                "args": ["server.js"],
            }
        ],
        skills_dir="/skills",
        enable_summarize=False,
        summarization_threshold_chars=50_000,
    )

    assert "--mcp-servers-json" in terminus.run_command
    assert "filesystem" in terminus.run_command
    assert "--skills-dir /skills" in terminus.run_command
    assert "--disable-summarize" in terminus.run_command
    assert "--summarization-threshold-chars 50000" in terminus.run_command


def test_terminus_2_harness_registers_task_skills():
    terminus = harnesses.terminus_2_harness()

    configured = terminus.with_skills(TaskSkills(skills_dir="/task/skills"))

    assert terminus.skills_path == "/task/skills"
    assert "--skills-dir /task/skills" in configured.run_command
    assert "--skills-dir /task/skills" not in terminus.run_command


def test_terminus_2_runner_augments_instruction_with_mcp_and_skills(tmp_path):
    skill_dir = tmp_path / "skills" / "debugging"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: Debugging\n"
        "description: Inspect failures and isolate root causes.\n"
        "---\n"
        "# Debugging\n"
    )

    instruction = build_augmented_instruction(
        "Fix the bug.",
        [
            {
                "name": "repo",
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server"],
            }
        ],
        str(tmp_path / "skills"),
    )

    assert "Fix the bug." in instruction
    assert "MCP Servers:" in instruction
    assert "- repo: stdio transport, command: uvx mcp-server" in instruction
    assert "<available_skills>" in instruction
    assert "<name>Debugging</name>" in instruction
    assert str(skill_dir / "SKILL.md") in instruction


def test_terminus_2_runner_summarization_threshold_uses_message_history():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "a" * 20},
        {"role": "assistant", "content": "b" * 20},
        {"role": "user", "content": "c" * 20},
    ]

    assert should_summarize(messages, threshold_chars=30, keep_messages=2)
    assert not should_summarize(messages, threshold_chars=30, keep_messages=3)


def test_terminus_2_runner_parses_json_and_xml_commands():
    json_result = parse_response(
        '{"analysis":"a","plan":"p","commands":[{"keystrokes":"ls\\n","duration":0.1}],"task_complete":"true"}'
    )
    xml_result = parse_response(
        "<response><analysis>a</analysis><plan>p</plan><commands>"
        '<keystrokes duration="0.2">pwd\n</keystrokes>'
        "</commands><task_complete>true</task_complete></response>",
        parser_name="xml",
    )

    assert json_result["commands"] == [{"keystrokes": "ls\n", "duration": 0.1}]
    assert json_result["task_complete"] is True
    assert xml_result["commands"] == [{"keystrokes": "pwd\n", "duration": 0.2}]
    assert xml_result["task_complete"] is True
