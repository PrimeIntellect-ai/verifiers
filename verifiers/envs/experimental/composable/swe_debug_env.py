"""No-agent debugger for SWE-style SandboxTaskSet instances."""

import shlex
import time
from typing import Any, Literal

import verifiers as vf
from prime_sandboxes import CreateSandboxRequest
from verifiers.envs.experimental.sandbox_mixin import (
    SandboxMixin,
    SandboxMonitorRubric,
    SandboxSetupError,
)
from verifiers.types import Messages, State

from .task import SandboxTaskSet

DebugStep = Literal["none", "gold_patch", "command", "script"]


class SWEDebugRubric(SandboxMonitorRubric):
    """Reads the reward set by SWEDebugEnv during setup."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_reward_func(self.debug_reward, weight=1.0)

    async def debug_reward(self, state: vf.State, **kwargs: Any) -> float:
        return float(state.get("reward") or 0.0)


class SWEDebugEnv(SandboxMixin, vf.MultiTurnEnv):
    """Create a task sandbox, optionally mutate it, optionally run tests.

    Pipeline:
    - entry: create sandbox and optionally run ``taskset.setup(state)``
    - debug step: ``none``, ``gold_patch``, ``command``, or ``script``
    - exit: optionally run task tests and score them
    """

    def __init__(
        self,
        taskset: SandboxTaskSet,
        dataset: Any = None,
        *,
        run_setup: bool = True,
        debug_step: DebugStep = "gold_patch",
        run_tests: bool = True,
        debug_command: str | None = None,
        debug_script: str | None = None,
        debug_script_path: str | None = None,
        debug_timeout: int | None = None,
        test_timeout: int = 900,
        cpu_cores: int | None = None,
        memory_gb: int | None = None,
        disk_size_gb: int | None = None,
        labels: list[str] | None = None,
        timeout_seconds: float = 1800.0,
        output_tail_chars: int = 2000,
        **sandbox_kwargs: Any,
    ):
        if debug_step not in ("none", "gold_patch", "command", "script"):
            raise ValueError(f"Unsupported debug_step: {debug_step!r}")
        if debug_step == "command" and not debug_command:
            raise ValueError("debug_command is required when debug_step='command'")
        if debug_step == "script" and not (debug_script or debug_script_path):
            raise ValueError(
                "debug_script or debug_script_path is required when debug_step='script'"
            )

        self.taskset = taskset
        self.run_setup = run_setup
        self.debug_step = debug_step
        self.run_tests = run_tests
        self.debug_command = debug_command
        self.debug_script = debug_script
        self.debug_script_path = debug_script_path
        self.debug_timeout = debug_timeout
        self.test_timeout = test_timeout
        self._cpu_cores = cpu_cores
        self._memory_gb = memory_gb
        self._disk_size_gb = disk_size_gb
        self.labels = labels or ["swe-debug"]
        self.timeout_seconds = timeout_seconds
        self.output_tail_chars = output_tail_chars

        super().__init__(
            dataset=dataset or taskset.get_dataset,
            rubric=SWEDebugRubric(),
            timeout_seconds=timeout_seconds,
        )
        self.init_sandbox_client(**sandbox_kwargs)

    async def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Messages:
        raise NotImplementedError("SWEDebugEnv does not use multi-turn interaction")

    async def setup_state(self, state: State) -> None:
        await super().setup_state(state)
        state["attempts"] = state.get("attempts", 0) + 1
        state["debug_step"] = self.debug_step
        state["run_setup"] = self.run_setup
        state["run_tests"] = self.run_tests

        t0 = time.perf_counter()
        valid = False
        exc: BaseException | None = None
        try:
            await self._create_task_sandbox(state)
            valid = await self._run_debug_pipeline(state)
        except Exception as e:  # noqa: BLE001
            exc = e
            state["error"] = vf.SandboxError(f"SWE debug failed: {repr(e)}")
            state["reward"] = 0.0
        finally:
            state["elapsed_s"] = time.perf_counter() - t0

        reason, tail = self._classify_outcome(valid, exc, state)
        state.setdefault("reason", reason)
        if tail:
            state["test_output_tail"] = tail

    async def _create_task_sandbox(self, state: State) -> None:
        info = state["info"]
        spec = self.taskset.get_sandbox_spec(info)
        timeout_minutes = (
            spec.timeout_minutes
            if spec.timeout_minutes is not None
            else self.compute_sandbox_timeout_minutes()
        )
        request = CreateSandboxRequest(
            name=f"swe-debug-{state.get('example_id', 'unknown')}",
            docker_image=spec.image,
            cpu_cores=spec.cpu_cores if self._cpu_cores is None else self._cpu_cores,
            memory_gb=spec.memory_gb if self._memory_gb is None else self._memory_gb,
            disk_size_gb=spec.disk_size_gb
            if self._disk_size_gb is None
            else self._disk_size_gb,
            gpu_count=spec.gpu_count,
            gpu_type=spec.gpu_type,
            vm=spec.gpu_count > 0,
            timeout_minutes=timeout_minutes,
            environment_vars=self.taskset.get_env_vars() or None,
            labels=self.labels,
        )
        t0 = time.perf_counter()
        await self.create_sandbox(state, request)
        state["sandbox_create_s"] = time.perf_counter() - t0

    async def post_sandbox_setup(self, state: State) -> None:
        state["sandbox_client"] = self.sandbox_client
        state["test_timeout"] = self.test_timeout
        state["run_background_job"] = self.run_background_job
        if not self.run_setup:
            state["setup_s"] = 0.0
            return
        t0 = time.perf_counter()
        await self.taskset.setup(state)
        state["setup_s"] = time.perf_counter() - t0

    async def _run_debug_pipeline(self, state: State) -> bool:
        t0 = time.perf_counter()
        if self.debug_step == "gold_patch":
            await self._apply_gold_patch(state)
        elif self.debug_step == "command":
            valid = await self._run_debug_command(state, self.debug_command or "")
            if not valid:
                state["body_s"] = time.perf_counter() - t0
                return False
        elif self.debug_step == "script":
            valid = await self._run_debug_script(state)
            if not valid:
                state["body_s"] = time.perf_counter() - t0
                return False
        state["body_s"] = time.perf_counter() - t0

        if self.run_tests:
            return await self._run_tests(state)

        state["reward"] = 1.0
        state["reason"] = "pass"
        return True

    async def _apply_gold_patch(self, state: State) -> None:
        apply_gold_patch = getattr(self.taskset, "_apply_gold_patch", None)
        if apply_gold_patch is None:
            raise RuntimeError("Taskset does not support gold patch application")
        t0 = time.perf_counter()
        await apply_gold_patch(state["sandbox_client"], state["sandbox_id"], state)
        state["gold_apply_s"] = time.perf_counter() - t0

    async def _run_debug_command(self, state: State, command: str) -> bool:
        return await self._execute_debug_command(state, command)

    async def _run_debug_script(self, state: State) -> bool:
        sandbox_id = state["sandbox_id"]
        remote_path = "/tmp/swe_debug_script.sh"
        if self.debug_script_path:
            await self.upload_file(sandbox_id, remote_path, self.debug_script_path)
        else:
            await self.upload_content(sandbox_id, self.debug_script or "", remote_path)
        command = f"chmod +x {remote_path} && {shlex.quote(remote_path)}"
        return await self._execute_debug_command(state, command)

    async def _execute_debug_command(self, state: State, command: str) -> bool:
        t0 = time.perf_counter()
        result = await self.sandbox_client.execute_command(
            state["sandbox_id"],
            command,
            working_dir=self._workdir(state),
            timeout=(
                self.debug_timeout
                if self.debug_timeout is not None
                else self.test_timeout
            ),
        )
        state["debug_run_s"] = time.perf_counter() - t0
        state["debug_exit_code"] = result.exit_code
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if stdout:
            state["debug_stdout_tail"] = stdout[-self.output_tail_chars :]
        if stderr:
            state["debug_stderr_tail"] = stderr[-self.output_tail_chars :]
        if result.exit_code == 0:
            return True
        state["reward"] = 0.0
        state["reason"] = "debug_command_failed"
        return False

    async def _run_tests(self, state: State) -> bool:
        run_tests = getattr(self.taskset, "_run_tests", None)
        calculate_reward = getattr(self.taskset, "_calculate_reward", None)
        if run_tests is None or calculate_reward is None:
            raise RuntimeError("Taskset does not support direct test execution")
        t0 = time.perf_counter()
        test_output = await run_tests(
            state["sandbox_client"],
            state["sandbox_id"],
            state,
            state.get("test_timeout", self.test_timeout),
        )
        state["test_run_s"] = time.perf_counter() - t0
        state["test_output"] = test_output
        reward = float(calculate_reward(test_output, state.get("info") or {}))
        state["reward"] = reward
        valid = reward > 0
        state["reason"] = "pass" if valid else "test_failed"
        return valid

    def _workdir(self, state: State) -> str:
        return self.taskset.get_workdir(state.get("info") or {})

    def _classify_outcome(
        self, valid: bool, exc: BaseException | None, state: State
    ) -> tuple[str, str | None]:
        test_output = state.get("test_output")
        tail = (
            test_output[-self.output_tail_chars :]
            if isinstance(test_output, str) and test_output
            else None
        )
        if valid:
            return "pass", tail
        if state.get("reason"):
            return str(state["reason"]), tail
        if exc is None:
            return "test_failed", tail

        from prime_sandboxes import (
            APIError,
            APITimeoutError,
            CommandTimeoutError,
            DownloadTimeoutError,
            PaymentRequiredError,
            SandboxImagePullError,
            SandboxNotRunningError,
            SandboxTimeoutError,
            UploadTimeoutError,
        )

        if isinstance(exc, PaymentRequiredError):
            return "billing_error", tail
        if isinstance(
            exc,
            (
                TimeoutError,
                APITimeoutError,
                CommandTimeoutError,
                DownloadTimeoutError,
                SandboxTimeoutError,
                UploadTimeoutError,
            ),
        ):
            return "timeout", tail
        if isinstance(exc, SandboxSetupError):
            return "setup_failed", tail
        if isinstance(
            exc,
            (
                vf.InfraError,
                APIError,
                SandboxImagePullError,
                SandboxNotRunningError,
            ),
        ):
            return "sandbox_error", tail

        msg = str(exc).lower()
        if "apply failed" in msg or "patch failed" in msg or "no gold patch" in msg:
            return "gold_apply_failed", tail
        if "does not support" in msg:
            return "unsupported_action", tail
        return "setup_failed", tail

    @vf.stop
    async def debug_completed(self, state: State) -> bool:
        return True

    @vf.cleanup
    async def destroy_sandbox(self, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            await self.delete_sandbox(sandbox_id)
