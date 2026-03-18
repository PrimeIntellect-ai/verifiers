from __future__ import annotations

from typing import Any, cast

from verifiers.envs.experimental.harnesses.base import Harness


def build_harness_from_config(
    config: dict[str, Any] | None,
    *,
    agent_workdir: str | None = None,
) -> Harness | None:
    if not isinstance(config, dict):
        return None

    agent_config = config.get("agent", {})
    if not isinstance(agent_config, dict):
        return None

    harness_config = agent_config.get("harness", {})
    if not isinstance(harness_config, dict) or not harness_config:
        return None

    transport = str(harness_config.get("transport") or "").strip().lower()
    agent_name = str(harness_config.get("agent") or "").strip().lower()
    timeout_seconds = float(
        harness_config.get("timeout_seconds", agent_config.get("timeout_sec", 3600.0))
    )

    if transport == "acp":
        from verifiers.envs.experimental.harnesses.acp_agent import ACPHarness

        default_commands = {
            "opencode": ("opencode", "acp"),
            "claude": ("claude", "acp"),
            "claude-code": ("claude", "acp"),
        }
        raw_command = harness_config.get("command")
        if raw_command is None:
            raw_command = default_commands.get(agent_name)
        if raw_command is None:
            raise ValueError(
                f"Unsupported ACP harness agent '{agent_name}'. "
                "Specify agent='opencode' or agent='claude-code', or provide command."
            )

        if isinstance(raw_command, str):
            command = (raw_command,)
        else:
            command = tuple(str(part) for part in cast(list[Any], raw_command))

        mcp_servers = harness_config.get("mcp_servers")
        return ACPHarness(
            command=command,
            cwd=str(harness_config.get("cwd") or agent_workdir or ""),
            session_model_id=cast(str | None, harness_config.get("session_model_id")),
            mcp_servers=cast(list[Any] | None, mcp_servers),
            system_prompt=str(harness_config.get("system_prompt") or ""),
            timeout_seconds=timeout_seconds,
        )

    if transport == "interceptor":
        if agent_name not in {"", "opencode"}:
            raise ValueError(
                "Only OpenCode interceptor harnesses are currently supported."
            )

        from verifiers.envs.experimental.harnesses.opencode import (
            DEFAULT_INSTALL_COMMAND,
            DEFAULT_RUN_COMMAND_TEMPLATE,
            OpenCodeHarness,
        )

        disabled_tools = harness_config.get("disabled_tools")
        return OpenCodeHarness(
            asset_dir=str(
                harness_config.get("asset_dir", OpenCodeHarness.DEFAULT_ASSET_DIR)
            ),
            agent_workdir=str(
                harness_config.get("agent_workdir") or agent_workdir or "/app"
            ),
            disabled_tools=cast(list[str] | None, disabled_tools),
            system_prompt=cast(str | None, harness_config.get("system_prompt")),
            install_command=str(
                harness_config.get("install_command") or DEFAULT_INSTALL_COMMAND
            ),
            run_command_template=str(
                harness_config.get("run_command_template")
                or DEFAULT_RUN_COMMAND_TEMPLATE
            ),
            disable_compaction=bool(
                harness_config.get(
                    "disable_compaction",
                    OpenCodeHarness.DEFAULT_DISABLE_COMPACTION,
                )
            ),
            enable_interleaved=bool(
                harness_config.get(
                    "enable_interleaved",
                    OpenCodeHarness.DEFAULT_ENABLE_INTERLEAVED,
                )
            ),
            provider_timeout_ms=int(
                harness_config.get(
                    "provider_timeout_ms",
                    OpenCodeHarness.DEFAULT_PROVIDER_TIMEOUT_MS,
                )
            ),
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(
        f"Unsupported harness transport '{transport}'. Expected 'acp' or 'interceptor'."
    )
