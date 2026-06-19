import fcntl
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

os.environ.setdefault(
    "TAU2_DATA_DIR", str(Path.home() / ".cache" / "tau2-bench-v1" / "data")
)

import verifiers.v1 as vf
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Description as TauDescription
from tau2.data_model.tasks import Task as TauTask
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.run import load_tasks
from tau2.utils.utils import DATA_DIR

TAU2_REPOSITORY = "https://github.com/sierra-research/tau2-bench.git"
TAU2_REVISION = "337326e62d8e0ca74c353b004a9c5d748e0ba914"
# Tau2's workflow variant uses the Telecom tasks with its procedural support policy.
Tau2Domain = Literal["airline", "retail", "telecom", "telecom-workflow"]


class Tau2TasksetConfig(vf.TasksetConfig):
    domain: Tau2Domain = "telecom"


class Tau2Task(vf.Task, TauTask):
    domain: Tau2Domain
    tau_description: TauDescription | None = None


class Tau2Taskset(vf.Taskset[Tau2Task, Tau2TasksetConfig]):
    def load_tasks(self) -> list[Tau2Task]:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        marker = DATA_DIR / ".tau2_revision"
        with (DATA_DIR / ".tau2_bootstrap.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not (
                (DATA_DIR / "tau2" / "domains").exists()
                and marker.exists()
                and marker.read_text() == TAU2_REVISION
            ):
                with tempfile.TemporaryDirectory(prefix="tau2_bench_v1_") as temp_dir:
                    subprocess.run(
                        ["git", "init", temp_dir],
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            temp_dir,
                            "fetch",
                            "--depth",
                            "1",
                            TAU2_REPOSITORY,
                            TAU2_REVISION,
                        ],
                        check=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            temp_dir,
                            "checkout",
                            "FETCH_HEAD",
                            "--",
                            "data",
                        ],
                        check=True,
                        capture_output=True,
                    )
                    shutil.copytree(
                        Path(temp_dir) / "data", DATA_DIR, dirs_exist_ok=True
                    )
                    marker.write_text(TAU2_REVISION)

        tasks = load_tasks(
            task_set_name=self.config.domain,
            task_split_name="base",
        )
        return [
            Tau2Task(
                **task.model_dump(exclude={"description"}),
                idx=index,
                name=task.id,
                description=str(task.description) if task.description else None,
                prompt=DEFAULT_FIRST_AGENT_MESSAGE.content or "",
                domain=self.config.domain,
                tau_description=task.description,
            )
            for index, task in enumerate(tasks)
        ]

    @vf.reward
    async def tau2_reward(self, trace: vf.Trace) -> float:
        if "tau2" not in trace.info:
            return 0.0
        simulation = SimulationRun.model_validate(trace.info["tau2"]["simulation"])
        reward = simulation.reward_info
        if reward is None:
            return 0.0
        trace.info["tau2"]["evaluation"] = reward.model_dump(mode="json")
        return float(reward.reward)


__all__ = ["Tau2Domain", "Tau2Task", "Tau2Taskset", "Tau2TasksetConfig"]
