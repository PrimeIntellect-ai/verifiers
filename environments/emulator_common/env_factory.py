import json
import logging
from pathlib import Path

import verifiers.v1 as vf

from .loader import load_manifest, manifest_to_rows
from .runners import VERIFIER_SCRIPT, build_program_config, hidden_manifest_json
from .sandbox import WORKSPACE, build_sandbox_config
from .task_schema import EmulatorManifest
from .test_sources import PublicTestManifest, load_public_test_manifest

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are implementing a hardware emulator for a deterministic coding benchmark.
Prioritize correct CPU semantics, memory maps, rendering determinism, timing,
clear module boundaries, and tests that make debugging tractable.
"""

AGENT_EXIT_STATUS_PATH = "/tmp/emulator_agent_exit_status"


def _allow_verifier_after_agent_exit(harness: vf.MiniSWEAgent) -> vf.MiniSWEAgent:
    """Keep verifier scoring reachable after a nonzero agent command exit."""
    program = getattr(harness, "program", None)
    if not isinstance(program, dict):
        return harness
    command = program.get("command")
    if not isinstance(command, list) or len(command) < 3:
        return harness
    run_script = command[2]
    if not isinstance(run_script, str):
        return harness

    wrapped_command = list(command)
    wrapped_command[2] = (
        f"{run_script}\n"
        "status=$?\n"
        f"printf '%s\\n' \"$status\" > {AGENT_EXIT_STATUS_PATH}\n"
        'if [ "$status" -ne 0 ]; then\n'
        '  echo "[emulator-benchmark] mini-swe-agent exited with status '
        '$status; running verifier anyway." >&2\n'
        "fi\n"
        "exit 0\n"
    )
    program["command"] = wrapped_command

    artifacts = dict(program.get("artifacts") or {})
    artifacts["emulator_agent_exit_status"] = {
        "path": AGENT_EXIT_STATUS_PATH,
        "format": "text",
        "optional": True,
    }
    program["artifacts"] = artifacts
    return harness


def _helper_source(filename: str) -> str:
    return (Path(__file__).resolve().parent / filename).read_text(encoding="utf-8")


def _score_component(state: vf.ConfigData, name: str) -> float:
    score = state.get("emulator_score", {})
    if not isinstance(score, dict):
        return 0.0
    components = score.get("components", {})
    if not isinstance(components, dict):
        return 0.0
    return float(components.get(name, 0.0))


class EmulatorTaskset(vf.Taskset):
    def __init__(
        self,
        manifest: EmulatorManifest,
        *,
        public_test_manifest: PublicTestManifest,
        max_tasks: int | None = None,
        cpu_cores: int | None = None,
        memory_gb: int | None = None,
        network_access: bool = True,
        sandbox_timeout_minutes: int | None = None,
        inline_system_prompt: bool = False,
        config: vf.TasksetConfig | None = None,
    ):
        self.manifest = manifest
        self.public_test_manifest = public_test_manifest
        self.max_tasks = max_tasks
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.network_access = network_access
        self.sandbox_timeout_minutes = sandbox_timeout_minutes
        self.inline_system_prompt = inline_system_prompt
        super().__init__(
            source=self.load_rows,
            taskset_id=manifest.environment_id,
            config=config,
        )

    def load_rows(self) -> list[vf.ConfigData]:
        rows = manifest_to_rows(
            self.manifest,
            max_tasks=self.max_tasks,
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            network_access=self.network_access,
            sandbox_timeout_minutes=self.sandbox_timeout_minutes,
        )
        for row in rows:
            if self.inline_system_prompt:
                instruction = f"{SYSTEM_PROMPT.strip()}\n\n{row['instruction']}"
                row["question"] = instruction
                row["instruction"] = instruction
                row["prompt"] = [{"role": "user", "content": instruction}]
            row["info"]["public_test_sources"] = self.public_test_manifest.to_dict()
        return rows

    @vf.setup(priority=250)
    async def setup_emulator_sandbox(self, task, state, sandbox=None) -> None:
        if sandbox is None:
            raise RuntimeError("Emulator benchmark setup requires an active sandbox.")
        state["_emulator_sandbox"] = sandbox
        state["sandbox_id"] = getattr(sandbox, "id", state.get("sandbox_id"))
        result = await sandbox.execute(
            "mkdir -p /workspace/src /workspace/tests /workspace/verification",
            timeout=30,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"workspace setup failed: {result.stderr}")

    @vf.reward(weight=1.0, priority=100)
    async def solved(self, task, state) -> float:
        agent_error = state.get("error")
        sandbox = state.get("_emulator_sandbox")
        if sandbox is None:
            state["emulator_score"] = {"score": 0.0, "error": "missing sandbox"}
            return 0.0

        manifest_json = hidden_manifest_json(task)
        await sandbox.upload_bytes(
            "/tmp/emulator_hidden_manifest.json",
            manifest_json.encode("utf-8"),
            "emulator_hidden_manifest.json",
        )
        await sandbox.upload_bytes(
            "/tmp/emulator_verify.py",
            VERIFIER_SCRIPT.encode("utf-8"),
            "emulator_verify.py",
        )
        await sandbox.upload_bytes(
            "/tmp/emulator_suite_adapters.py",
            _helper_source("suite_adapters.py").encode("utf-8"),
            "emulator_suite_adapters.py",
        )
        source_json = json.dumps(task["info"].get("public_test_sources", {}))
        await sandbox.upload_bytes(
            "/tmp/emulator_public_tests.json",
            source_json.encode("utf-8"),
            "emulator_public_tests.json",
        )
        verify_timeout = int(
            task["info"].get("runtime", {}).get("verify_timeout_seconds", 900)
        )
        command = (
            "python3 /tmp/emulator_verify.py "
            "--workspace /workspace "
            "--manifest /tmp/emulator_hidden_manifest.json "
            "--sources /tmp/emulator_public_tests.json "
            "--output /tmp/emulator_score.json"
        )
        result = await sandbox.run_background_job(
            command,
            timeout=verify_timeout,
            working_dir=WORKSPACE,
        )
        state["emulator_verify_stdout"] = result.stdout or ""
        state["emulator_verify_stderr"] = result.stderr or ""

        output = await sandbox.execute("cat /tmp/emulator_score.json", timeout=30)
        if output.exit_code != 0:
            state["emulator_score"] = {
                "score": 0.0,
                "error": "missing verifier output",
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            }
            return 0.0
        try:
            parsed = json.loads(output.stdout or "{}")
        except json.JSONDecodeError as exc:
            state["emulator_score"] = {"score": 0.0, "error": str(exc)}
            return 0.0
        agent_exit_status = state.get("artifacts", {}).get("emulator_agent_exit_status")
        if agent_exit_status is not None and isinstance(parsed, dict):
            parsed["agent_exit_status"] = str(agent_exit_status).strip()
        if agent_error is not None and isinstance(parsed, dict):
            parsed["agent_error"] = str(agent_error)
        state["emulator_score"] = parsed
        return float(parsed.get("score", 0.0))

    @vf.metric
    async def public_test_score(self, task, state) -> float:
        _ = task
        return _score_component(state, "public_rom_pass_rate")

    @vf.metric
    async def deterministic_runner_score(self, task, state) -> float:
        _ = task
        return _score_component(state, "deterministic_replay_score")

    @vf.metric
    async def public_rom_pass_rate(self, task, state) -> float:
        _ = task
        return _score_component(state, "public_rom_pass_rate")

    @vf.metric
    async def runner_contract_score(self, task, state) -> float:
        _ = task
        return _score_component(state, "runner_contract_score")

    @vf.metric
    async def build_score(self, task, state) -> float:
        _ = task
        return _score_component(state, "build_score")

    @vf.metric
    async def framebuffer_hash_score(self, task, state) -> float:
        _ = task
        return _score_component(state, "framebuffer_hash_score")

    @vf.metric
    async def trace_cpu_score(self, task, state) -> float:
        _ = task
        return _score_component(state, "trace_cpu_score")

    @vf.metric
    async def audio_perf_score(self, task, state) -> float:
        _ = task
        return _score_component(state, "audio_perf_score")

    @vf.metric
    async def runtime_stability_score(self, task, state) -> float:
        _ = task
        return _score_component(state, "runtime_stability_score")

    @vf.metric
    async def emulator_agent_exit_status(self, task, state) -> float:
        _ = task
        score = state.get("emulator_score", {})
        if not isinstance(score, dict):
            return -1.0
        try:
            return float(score.get("agent_exit_status", -1.0))
        except (TypeError, ValueError):
            return -1.0

    @vf.cleanup
    async def cleanup_emulator_sandbox_state(self, task, state) -> None:
        _ = task
        state.pop("_emulator_sandbox", None)


@vf.reward(weight=1.0)
async def eval_smoke_passed(task, state) -> float:
    return float(state.get("answer") == task.get("answer"))


@vf.metric
async def eval_smoke_public_case_count(task, state) -> float:
    _ = state
    public_cases = task["info"].get("public_cases", [])
    return float(len(public_cases) if isinstance(public_cases, list) else 0)


async def eval_smoke_program(task, state) -> vf.ConfigData:
    answer = str(task.get("answer", "pass"))
    state["answer"] = answer
    state["completion"] = [{"role": "assistant", "content": answer}]
    state["emulator_score"] = {
        "mode": "eval_smoke",
        "score": 1.0,
        "environment_id": task["info"].get("environment_id"),
    }
    return state


def load_eval_smoke_taskset(
    manifest: EmulatorManifest,
    *,
    public_test_manifest: PublicTestManifest,
    max_tasks: int | None = None,
    config: vf.TasksetConfig | None = None,
) -> vf.Taskset:
    rows = manifest_to_rows(manifest, max_tasks=max_tasks)
    for row in rows:
        row.pop("sandbox", None)
        row.pop("program", None)
        row["info"]["public_test_sources"] = public_test_manifest.to_dict()
    return vf.Taskset(
        source=rows,
        taskset_id=f"{manifest.environment_id}-eval-smoke",
        rewards=[eval_smoke_passed],
        metrics=[eval_smoke_public_case_count],
        config=config,
    )


def load_eval_smoke_harness(
    *,
    config: vf.HarnessConfig | None = None,
) -> vf.Harness:
    return vf.Harness(program=eval_smoke_program, config=config)


@vf.reward(weight=1.0)
async def prime_smoke_passed(task, state) -> float:
    _ = task
    return float(state.get("error") is None)


@vf.metric
async def prime_smoke_environment_level(task, state) -> float:
    _ = state
    return float(task["info"].get("level", 0))


def prime_smoke_instruction(manifest: EmulatorManifest) -> str:
    return f"""\
This is an infrastructure smoke test for the {manifest.display_name} emulator
benchmark environment, not the emulator implementation task.

Do not inspect files, edit files, or run tests. Finish immediately with exactly
one action:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""


def load_prime_smoke_taskset(
    manifest: EmulatorManifest,
    *,
    public_test_manifest: PublicTestManifest,
    max_tasks: int | None = None,
    cpu_cores: int | None = None,
    memory_gb: int | None = None,
    network_access: bool = True,
    sandbox_timeout_minutes: int | None = None,
    config: vf.TasksetConfig | None = None,
) -> vf.Taskset:
    rows = manifest_to_rows(
        manifest,
        max_tasks=max_tasks,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        network_access=network_access,
        sandbox_timeout_minutes=sandbox_timeout_minutes,
    )
    instruction = prime_smoke_instruction(manifest)
    for row in rows:
        row["task_id"] = f"{manifest.environment_id}__prime_smoke"
        row["question"] = instruction
        row["instruction"] = instruction
        row["prompt"] = [{"role": "user", "content": instruction}]
        row["info"]["public_test_sources"] = public_test_manifest.to_dict()
    return vf.Taskset(
        source=rows,
        taskset_id=f"{manifest.environment_id}-prime-smoke",
        rewards=[prime_smoke_passed],
        metrics=[prime_smoke_environment_level],
        config=config,
    )


def _load_harness(
    manifest: EmulatorManifest,
    *,
    config: vf.HarnessConfig | None = None,
    cpu_cores: int | None = None,
    memory_gb: int | None = None,
    network_access: bool = True,
    sandbox_timeout_minutes: int | None = None,
    environment_timeout: int = 21600,
    agent_step_limit: int = 1000,
    max_turns: int | None = None,
    allow_agent_failure_for_scoring: bool = False,
    inline_system_prompt: bool = False,
) -> vf.MiniSWEAgent:
    sandbox = build_sandbox_config(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        network_access=network_access,
        timeout_minutes=sandbox_timeout_minutes,
    )
    harness = vf.MiniSWEAgent(
        agent_workdir=WORKSPACE,
        environment_timeout=environment_timeout,
        extra_config_specs=[f"agent.step_limit={agent_step_limit}"],
        system_prompt=None if inline_system_prompt else SYSTEM_PROMPT,
        sandbox=sandbox,
        program=build_program_config(manifest),
        max_turns=max_turns,
        config=config,
    )
    if allow_agent_failure_for_scoring:
        return _allow_verifier_after_agent_exit(harness)
    return harness


def load_emulator_environment(
    manifest_path: str | Path,
    *,
    config: vf.EnvConfig | None = None,
    max_tasks: int | None = None,
    cpu_cores: int | None = None,
    memory_gb: int | None = None,
    network_access: bool = True,
    sandbox_timeout_minutes: int | None = None,
    environment_timeout: int = 21600,
    agent_step_limit: int = 1000,
    max_turns: int | None = None,
    eval_smoke: bool = False,
    prime_smoke: bool = False,
    inline_system_prompt: bool = False,
) -> vf.Env:
    config = config or vf.EnvConfig()
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    public_test_manifest = load_public_test_manifest(
        manifest_path.parent / "public_tests.json"
    )
    if eval_smoke:
        return vf.Env(
            taskset=load_eval_smoke_taskset(
                manifest,
                public_test_manifest=public_test_manifest,
                max_tasks=max_tasks,
                config=config.taskset,
            ),
            harness=load_eval_smoke_harness(config=config.harness),
        )

    if prime_smoke:
        return vf.Env(
            taskset=load_prime_smoke_taskset(
                manifest,
                public_test_manifest=public_test_manifest,
                max_tasks=max_tasks,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                network_access=network_access,
                sandbox_timeout_minutes=sandbox_timeout_minutes,
                config=config.taskset,
            ),
            harness=_load_harness(
                manifest,
                config=config.harness,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                network_access=network_access,
                sandbox_timeout_minutes=sandbox_timeout_minutes,
                environment_timeout=environment_timeout,
                agent_step_limit=agent_step_limit,
                max_turns=max_turns,
            ),
        )

    taskset = EmulatorTaskset(
        manifest,
        public_test_manifest=public_test_manifest,
        max_tasks=max_tasks,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        network_access=network_access,
        sandbox_timeout_minutes=sandbox_timeout_minutes,
        inline_system_prompt=inline_system_prompt,
        config=config.taskset,
    )
    harness = _load_harness(
        manifest,
        config=config.harness,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        network_access=network_access,
        sandbox_timeout_minutes=sandbox_timeout_minutes,
        environment_timeout=environment_timeout,
        agent_step_limit=agent_step_limit,
        max_turns=max_turns,
        allow_agent_failure_for_scoring=True,
        inline_system_prompt=inline_system_prompt,
    )
    return vf.Env(taskset=taskset, harness=harness)
