from __future__ import annotations

import os
from collections.abc import Mapping
import time
from typing import Literal, cast

from prime_sandboxes import CommandTimeoutError, SandboxImagePullError

from verifiers.decorators import cleanup
from verifiers.errors import SandboxError
from verifiers.types import State

from verifiers.envs.experimental.channels import (
    ChannelMap,
    SandboxResources,
    SandboxSeed,
    SandboxSpec,
)
from verifiers.envs.experimental.channels.sandbox_channel import (
    SANDBOX_FAILURES,
    resolve_sandbox_spec,
)
from verifiers.envs.experimental.resources import Resources
from verifiers.envs.experimental.task import Task

SandboxUse = Literal["auto", "existing", "new"]


class SandboxTool:
    """Base class for tools that execute inside Prime sandboxes."""

    def __init__(
        self,
        name: str,
        sandbox: SandboxSpec | None = None,
        sandbox_use: SandboxUse = "auto",
        sandbox_key: str | None = None,
        command_timeout: int = 60,
        setup_timeout: int = 300,
        setup_commands: list[str] | None = None,
        sandbox_runtime: dict[str, object] | None = None,
        sandbox_wait_for_creation_max_attempts: int = 120,
        set_default_sandbox_id: bool = True,
    ):
        self.name = name
        self.sandbox = sandbox or SandboxSpec()
        self.sandbox_use = sandbox_use
        self.sandbox_key = sandbox_key or name
        self.command_timeout = command_timeout
        self.setup_timeout = setup_timeout
        self.setup_commands = list(setup_commands or [])
        self.sandbox_runtime = dict(sandbox_runtime or {})
        self.sandbox_wait_for_creation_max_attempts = (
            sandbox_wait_for_creation_max_attempts
        )
        self.set_default_sandbox_id = set_default_sandbox_id

    def channels(self) -> ChannelMap:
        return {
            "sandbox": {
                "runtime": self.sandbox_runtime,
                "tool_sandboxes": {
                    self.sandbox_key: {
                        "sandbox_use": self.sandbox_use,
                        "sandbox": self.sandbox,
                    }
                },
            },
            "cleanup": self.cleanup,
        }

    def sandbox_state(self, state: State) -> dict[str, object]:
        tool_states = cast(dict[str, object], state.setdefault("sandbox_tools", {}))
        sandbox_state = cast(
            dict[str, object], tool_states.setdefault(self.sandbox_key, {})
        )
        sandbox_state.setdefault("command_execution_times", [])
        sandbox_state.setdefault("ready", False)
        return sandbox_state

    async def ensure_sandbox(
        self, task: Task, state: State, resources: Resources
    ) -> str:
        sandbox_use, sandbox_spec = self.resolved_sandbox_config(resources)
        sandbox_state = self.sandbox_state(state)
        existing_id = sandbox_state.get("sandbox_id")
        if existing_id:
            return str(existing_id)

        if sandbox_use in {"auto", "existing"} and state.get("sandbox_id"):
            sandbox_id = str(state["sandbox_id"])
            sandbox_state["sandbox_id"] = sandbox_id
            sandbox_state["owns_sandbox"] = False
            sandbox_state["ready"] = True
            try:
                await self.run_setup_commands(sandbox_id, sandbox_state, resources)
            except SandboxError:
                raise
            except Exception as e:
                raise SandboxError(
                    f"Sandbox setup failed for tool {self.name!r}: {e}"
                ) from e
            return sandbox_id

        if sandbox_use == "existing":
            raise SandboxError(
                f"Tool {self.name!r} requires an existing sandbox_id in state."
            )

        runtime = resources.require("sandbox_runtime", SandboxResources)
        seed = resources.get("sandbox_request")
        spec = resolve_sandbox_spec(sandbox_spec, seed)
        try:
            sandbox_id = await runtime.create_from_spec(
                f"{self.sandbox_key}-{task.example_id}-{os.urandom(4).hex()}",
                spec,
                max_attempts=self.sandbox_wait_for_creation_max_attempts,
            )
        except SandboxImagePullError as e:
            state["sandbox_image_pull_error"] = True
            raise SandboxError(f"Failed to pull sandbox image {spec.image}: {e}") from e
        except Exception as e:
            raise SandboxError(
                f"Sandbox creation failed for image {spec.image}: {e}"
            ) from e
        sandbox_state["sandbox_id"] = sandbox_id
        sandbox_state["owns_sandbox"] = True
        if self.set_default_sandbox_id and sandbox_use == "auto":
            state["sandbox_id"] = sandbox_id
        setup_commands = []
        if isinstance(seed, SandboxSeed):
            setup_commands.extend(seed.setup_commands)
        setup_commands.extend(self.setup_commands)
        try:
            await self.run_setup_commands(
                sandbox_id, sandbox_state, resources, setup_commands=setup_commands
            )
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError(
                f"Sandbox setup failed for image {spec.image}: {e}"
            ) from e

        sandbox_state["ready"] = True
        return sandbox_id

    async def run_setup_commands(
        self,
        sandbox_id: str,
        sandbox_state: dict[str, object],
        resources: Resources,
        setup_commands: list[str] | None = None,
    ) -> None:
        if sandbox_state.get("setup_complete"):
            return
        commands = self.setup_commands if setup_commands is None else setup_commands
        if commands:
            runtime = resources.require("sandbox_runtime", SandboxResources)
            await runtime.run_setup_commands(
                sandbox_id, commands, timeout=self.setup_timeout
            )
        sandbox_state["setup_complete"] = True

    def resolved_sandbox_config(
        self, resources: Resources
    ) -> tuple[SandboxUse, SandboxSpec]:
        configs = resources.require("tool_sandboxes", dict)
        config = cast(Mapping[str, object], configs[self.sandbox_key])
        sandbox_use = config["sandbox_use"]
        sandbox = config["sandbox"]
        if sandbox_use not in {"auto", "existing", "new"}:
            raise SandboxError(f"Invalid sandbox_use for tool {self.name!r}.")
        if not isinstance(sandbox, SandboxSpec):
            raise SandboxError(f"Invalid sandbox spec for tool {self.name!r}.")
        return cast(SandboxUse, sandbox_use), sandbox

    async def execute_command(
        self,
        command: str,
        task: Task,
        state: State,
        resources: Resources,
        timeout: int | None = None,
        working_dir: str | None = None,
    ) -> tuple[int, str]:
        runtime = resources.require("sandbox_runtime", SandboxResources)
        sandbox_id = await self.ensure_sandbox(task, state, resources)
        start = time.time()
        try:
            result = await runtime.with_retry(runtime.client.execute_command)(
                sandbox_id,
                command,
                timeout=timeout or self.command_timeout,
                working_dir=working_dir,
            )
        except SANDBOX_FAILURES as e:
            if e.__class__.__name__ == "SandboxOOMError":
                state["sandbox_oom"] = True
            if e.__class__.__name__ == "SandboxTimeoutError":
                state["sandbox_timeout"] = True
            raise SandboxError(f"Sandbox failed while executing command: {e}") from e
        except CommandTimeoutError:
            state["command_timeout_count"] = (
                int(state.get("command_timeout_count", 0)) + 1
            )
            self.record_command_time(state, time.time() - start)
            return (
                -1,
                f"The last command <command>{command}</command> timed out and has been killed.\n"
                "Try another command and avoid interactive input.",
            )
        self.record_command_time(state, time.time() - start)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        parts = [stdout] if stdout else []
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return result.exit_code, "\n".join(parts) if parts else "(no output)"

    def record_command_time(self, state: State, seconds: float) -> None:
        sandbox_state = self.sandbox_state(state)
        command_times = cast(list[float], sandbox_state["command_execution_times"])
        command_times.append(seconds)

    @cleanup(priority=-100)
    async def cleanup(self, task: Task, state: State, resources: Resources) -> None:
        sandbox_state = self.sandbox_state(state)
        if not sandbox_state.get("owns_sandbox"):
            return
        sandbox_id = sandbox_state.get("sandbox_id")
        if not sandbox_id:
            return
        runtime = resources.require("sandbox_runtime", SandboxResources)
        await runtime.delete(str(sandbox_id))
        sandbox_state["owns_sandbox"] = False
