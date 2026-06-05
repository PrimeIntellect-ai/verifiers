"""hello-harbor: Terminal-Bench / Harbor tasks loaded from disk.

Each task dir (harbor format) ships `instruction.md` + `task.toml` (+ `tests/`,
`solution/`, `environment/`). The agent runs in a container runtime and modifies
`/app`; then the per-task verifier (`tests/test.sh`) runs in the SAME container
and the reward it writes to `/logs/verifier/reward.txt` becomes the score.

All of this lives on the taskset (per-task image via `runtime_config`, in-runtime
checks via `verify`), so a harbor task runs under ANY agent — the built-in one
or a custom one like rlm. Docker / prime only (the tests use
container-absolute paths); `allowed_runtimes` hard-blocks anything else.
"""

import base64
import io
import tarfile
import tomllib
from pathlib import Path

import verifiers.nano as vf

TASKS_DIR = Path(__file__).resolve().parent / "harbor_tasks"


class HarborTask(vf.Task):
    task_dir: str
    """Absolute path to the task's harbor directory."""
    docker_image: str
    """The container image the task declares in task.toml."""


def load_harbor_tasks(tasks_dir: Path) -> list[HarborTask]:
    tasks = []
    for task_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        if not (task_dir / "task.toml").exists():
            continue
        config = tomllib.loads((task_dir / "task.toml").read_text())
        tasks.append(
            HarborTask(
                id=task_dir.name,
                instruction=(task_dir / "instruction.md").read_text().strip(),
                task_dir=str(task_dir),
                docker_image=config["environment"]["docker_image"],
            )
        )
    return tasks


def tar_b64(directory: Path) -> str:
    """Tar a directory's contents (flat) and base64-encode it for shell staging."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name)
    return base64.b64encode(buffer.getvalue()).decode()


class HarborConfig(vf.TasksetConfig):
    allowed_runtimes: list[str] | None = ["docker", "prime"]


class HarborTaskset(vf.Taskset[HarborTask, HarborConfig]):
    def load_tasks(self) -> list[HarborTask]:
        return load_harbor_tasks(TASKS_DIR)

    def runtime_config(self, task: HarborTask, base):
        if "image" in type(base).model_fields:
            return base.model_copy(update={"image": task.docker_image})
        return base

    async def verify(self, transcript: vf.Transcript, runtime: vf.Runtime) -> None:
        # Stage the task's tests into the live runtime and run its harbor verifier;
        # test.sh writes the reward to /logs/verifier/reward.txt.
        try:
            tests = tar_b64(Path(transcript.task.task_dir) / "tests")
            await runtime.run(
                [
                    "sh",
                    "-c",
                    f"mkdir -p /logs/verifier /tests && printf %s {tests} "
                    "| base64 -d | tar -xzf - -C /tests",
                ],
                {},
            )
            await runtime.run(["sh", "-c", "cd /tests && bash test.sh"], {})
            result = await runtime.run(
                ["sh", "-c", "cat /logs/verifier/reward.txt 2>/dev/null || true"], {}
            )
            transcript.metrics["verifier_reward"] = float(result.stdout.strip())
        except (ValueError, OSError):
            transcript.metrics["verifier_reward"] = 0.0

    @vf.reward(weight=1.0)
    async def solved(self, task: HarborTask, transcript: vf.Transcript) -> float:
        return transcript.metrics.get("verifier_reward", 0.0)


class EnvConfig(vf.EnvConfig):
    taskset: HarborConfig = HarborConfig()
    # Default: the built-in agent in docker (it installs its own openai).
    runtime: vf.RuntimeConfig = vf.DockerConfig(image="python:3.11-slim")


def load_taskset(config: HarborConfig | None = None) -> HarborTaskset:
    return HarborTaskset(config or HarborConfig())


def load_agent(config: vf.AgentConfig | None = None) -> vf.Agent:
    return vf.make_agent(config or vf.DefaultAgentConfig())


def load_environment(config: EnvConfig | None = None) -> vf.Environment:
    config = config or EnvConfig()
    return vf.Environment(
        taskset=load_taskset(config.taskset),
        agent=load_agent(config.agent),
        runtime=config.runtime,
    )
