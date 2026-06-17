"""openswe-v1 - GAIR/OpenSWE as a v1 taskset.

OpenSWE rows carry a task-specific image and an ``eval_script`` that emits
``OPENSWE_EXIT_CODE=<n>``. Setup links the prebuilt conda environment into the
standard agent venv location, and the reward runs the row eval script in the
same runtime the agent edited.
"""

import re

import verifiers.v1 as vf

DATASET = "GAIR/OpenSWE"
CONFIG = "openswe_oss"
PRIME_TEAM = "team-clyvldofb0000gg1kx39rgzjq"
REPO_PATH = "/testbed"

ENV = {
    "PATH": (
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

EXIT_CODE_RE = re.compile(r"OPENSWE_EXIT_CODE=(\d+)")


def build_image_ref(image_name: str) -> str:
    """OpenSWE stores short mixed-case aliases; Prime pulls team-scoped refs."""
    image_ref = image_name.lower()
    if "/" in image_ref:
        return image_ref
    return f"{PRIME_TEAM}/{image_ref}:latest"


class OpenSWETask(vf.Task):
    eval_script: str
    """Row-provided verifier script. Reward is based on its OPENSWE_EXIT_CODE marker."""
    gold_patch: str = ""
    """Gold patch for configs that ship a patch column."""


class OpenSWEConfig(vf.TasksetConfig):
    dataset_name: str = DATASET
    dataset_config: str = CONFIG
    split: str = "train"


class OpenSWETaskset(vf.Taskset[OpenSWETask, OpenSWEConfig]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[OpenSWETask]:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download

        data_file = f"{self.config.dataset_config}.jsonl"
        path = hf_hub_download(
            repo_id=self.config.dataset_name,
            repo_type="dataset",
            filename=data_file,
        )
        rows = load_dataset(
            "json",
            data_files=path,
            split=self.config.split,
        )
        return [
            OpenSWETask(
                idx=i,
                name=row.get("instance_id") or row.get("repo") or f"openswe-{i}",
                instruction=row["problem_statement"],
                image=build_image_ref(row["image_name"]),
                workdir=REPO_PATH,
                resources=vf.Resources(cpu=4, memory=4, disk=10),
                eval_script=row.get("eval_script") or "",
                gold_patch=row.get("patch") or "",
            )
            for i, row in enumerate(rows)
        ]

    async def setup(self, task: OpenSWETask, runtime: vf.Runtime) -> None:
        result = await runtime.run(
            ["sh", "-c", "ln -sfn /opt/conda/envs/testbed /root/.venv"], ENV
        )
        if result.exit_code != 0:
            raise vf.ProgramError(
                f"openswe setup failed ({task.name}): {result.stderr.strip()[-500:]}"
            )

    @vf.reward(weight=1.0)
    async def solved(self, task: OpenSWETask, runtime: vf.Runtime) -> float:
        if not task.eval_script.strip():
            return 0.0
        await runtime.write("/eval.sh", task.eval_script.encode())
        await runtime.run(["chmod", "+x", "/eval.sh"], ENV)
        result = await runtime.run(
            ["sh", "-c", "bash /eval.sh > /test_output.txt 2>&1"], ENV
        )
        if result.exit_code > 1:
            raise vf.ProgramError(
                f"openswe tests errored ({task.name}): exit={result.exit_code}"
            )
        output = await runtime.run(["cat", "/test_output.txt"], ENV)
        match = EXIT_CODE_RE.search(output.stdout or "")
        return 1.0 if match and match.group(1) == "0" else 0.0

    async def validate(self, task: OpenSWETask, runtime: vf.Runtime) -> bool:
        await self.apply_gold_patch(task, runtime)
        return await self.solved(task, runtime) == 1.0

    async def apply_gold_patch(self, task: OpenSWETask, runtime: vf.Runtime) -> None:
        if self.config.dataset_config == "openswe_other":
            raise vf.ProgramError("openswe_other rows do not include gold patches")
        if not task.gold_patch.strip():
            raise vf.ProgramError(f"empty gold patch for {task.name}")
        await runtime.write("/tmp/gold.patch", task.gold_patch.encode())
        for cmd in (
            "git apply --whitespace=fix /tmp/gold.patch",
            "patch --fuzz=5 -p1 -i /tmp/gold.patch",
        ):
            result = await runtime.run(["sh", "-c", cmd], ENV)
            if result.exit_code == 0:
                return
        raise vf.ProgramError(
            f"openswe gold apply failed ({task.name}): "
            f"exit={result.exit_code} {result.stderr.strip()[-500:]}"
        )


def load_taskset(config: OpenSWEConfig) -> OpenSWETaskset:
    return OpenSWETaskset(config)
