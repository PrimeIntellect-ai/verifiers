"""swesmith-v1 - SWE-Smith multilingual tasks as a v1 taskset.

By default this taskset loads every official SWE-Smith language dataset. Setup
checks out the per-instance branch at the bug commit with F2P tests visible.
The reward restores known test files, runs the upstream profile test command,
and compares the upstream parser status map against FAIL_TO_PASS/PASS_TO_PASS.
"""

import shlex
import tempfile
from pathlib import Path
from textwrap import dedent

import verifiers.v1 as vf

__all__ = ["SWESmithConfig", "SWESmithTask", "SWESmithTaskset"]

LANGUAGE_TO_DATASET = {
    "py": "SWE-bench/SWE-smith-py",
    "go": "SWE-bench/SWE-smith-go",
    "java": "SWE-bench/SWE-smith-java",
    "js": "SWE-bench/SWE-smith-js",
    "ts": "SWE-bench/SWE-smith-ts",
    "rs": "SWE-bench/SWE-smith-rs",
    "cpp": "SWE-bench/SWE-smith-cpp",
    "php": "SWE-bench/SWE-smith-php",
}
ALL_LANGUAGES = ["py", "go", "java", "js", "ts", "rs", "cpp", "php"]
REPO_PATH = "/testbed"

ENV = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
        "/usr/local/cargo/bin:/usr/local/go/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "CI": "1",
}


def get_profile(row: dict):
    from swesmith.profiles import registry

    return registry.get_from_inst(row)


def build_eval_script(test_cmd: str) -> str:
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START

    return dedent(
        f"""\
        #!/bin/bash
        set -uxo pipefail

        cd {REPO_PATH}

        : '{TEST_OUTPUT_START}'
        {test_cmd}
        : '{TEST_OUTPUT_END}'
        """
    )


def parse_status_map(output: str, profile) -> dict[str, str]:
    from swesmith.harness.grading import read_test_output

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as file:
        file.write(output)
        path = Path(file.name)
    try:
        bounded, found = read_test_output(str(path))
    finally:
        path.unlink(missing_ok=True)
    if not found or not bounded:
        return {}
    return profile.log_parser(bounded)


def is_resolved(
    status_map: dict[str, str], fail_to_pass: list[str], pass_to_pass: list[str]
) -> bool:
    from swebench.harness.constants import TestStatus

    passed = {TestStatus.PASSED.value, TestStatus.XFAIL.value}
    tests = fail_to_pass + pass_to_pass
    if not tests:
        return False
    return all(status_map.get(test) in passed for test in tests)


class SWESmithTask(vf.Task):
    language: str
    row: dict
    """Original SWE-Smith row for upstream profile methods."""
    instance_id: str
    gold_patch: str = ""
    fail_to_pass: list[str] = []
    pass_to_pass: list[str] = []


class SWESmithConfig(vf.TasksetConfig):
    languages: list[str] = list(ALL_LANGUAGES)
    """Languages to load. Default is every official SWE-Smith language."""
    dataset_names: dict[str, str] = {}
    """Optional per-language dataset overrides."""
    split: str = "train"


class SWESmithTaskset(vf.Taskset[SWESmithTask, SWESmithConfig]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[SWESmithTask]:
        from datasets import load_dataset
        from swesmith.profiles import registry

        tasks: list[SWESmithTask] = []
        for language in self.config.languages:
            if language not in LANGUAGE_TO_DATASET:
                raise ValueError(
                    f"unknown SWE-Smith language {language!r}; "
                    f"available: {sorted(LANGUAGE_TO_DATASET)}"
                )
            dataset_name = self.config.dataset_names.get(
                language, LANGUAGE_TO_DATASET[language]
            )
            rows = load_dataset(dataset_name, split=self.config.split)
            for raw in rows:
                row = dict(raw)
                try:
                    registry.get_from_inst(row)
                except KeyError:
                    continue
                idx = len(tasks)
                instance_id = row["instance_id"]
                tasks.append(
                    SWESmithTask(
                        idx=idx,
                        name=f"{language}:{instance_id}",
                        instruction=row["problem_statement"],
                        image=row["image_name"],
                        workdir=REPO_PATH,
                        resources=vf.Resources(cpu=4, memory=4, disk=10),
                        language=language,
                        row=row,
                        instance_id=instance_id,
                        gold_patch=row.get("patch") or "",
                        fail_to_pass=list(row.get("FAIL_TO_PASS") or []),
                        pass_to_pass=list(row.get("PASS_TO_PASS") or []),
                    )
                )
        return tasks

    async def setup(self, task: SWESmithTask, runtime: vf.Runtime) -> None:
        fetch = await runtime.run(["git", "fetch", "origin"], ENV)
        if fetch.exit_code != 0:
            raise vf.ProgramError(
                f"swesmith git fetch failed ({task.name}): "
                f"{fetch.stderr.strip()[-500:]}"
            )
        checkout = await runtime.run(["git", "checkout", task.instance_id], ENV)
        if checkout.exit_code != 0:
            raise vf.ProgramError(
                f"swesmith checkout failed ({task.name}): "
                f"{checkout.stderr.strip()[-500:]}"
            )

        head_msg = await runtime.run(["git", "log", "-1", "--format=%s"], ENV)
        if "Remove F2P Tests" in (head_msg.stdout or ""):
            result = await runtime.run(["git", "checkout", "HEAD~1"], ENV)
            if result.exit_code != 0:
                raise vf.ProgramError(
                    f"swesmith checkout HEAD~1 failed ({task.name}): "
                    f"{result.stderr.strip()[-500:]}"
                )

        if task.language == "py":
            await runtime.run(
                [
                    "sh",
                    "-c",
                    "for d in /opt/miniconda3/envs/testbed /opt/conda/envs/testbed; "
                    'do [ -d "$d" ] && ln -sfn "$d" /root/.venv && break; done; true',
                ],
                ENV,
            )

        rg_install = (
            "command -v rg >/dev/null 2>&1 || { "
            "V=14.1.1; "
            "curl -sSL "
            "https://github.com/BurntSushi/ripgrep/releases/download/"
            "${V}/ripgrep-${V}-x86_64-unknown-linux-musl.tar.gz "
            "| tar xz -C /tmp "
            "&& install -m 755 /tmp/ripgrep-${V}-x86_64-unknown-linux-musl/rg /usr/local/bin/rg; "
            "}"
        )
        await runtime.run(["sh", "-c", rg_install], ENV)

    @vf.reward(weight=1.0)
    async def solved(self, task: SWESmithTask, runtime: vf.Runtime) -> float:
        output = await self.run_tests(task, runtime)
        profile = get_profile(task.row)
        status_map = parse_status_map(output, profile)
        if not status_map:
            return 0.0
        return (
            1.0
            if is_resolved(status_map, task.fail_to_pass, task.pass_to_pass)
            else 0.0
        )

    async def run_tests(self, task: SWESmithTask, runtime: vf.Runtime) -> str:
        profile = get_profile(task.row)
        await self.revert_test_files(runtime, profile, task.row)
        test_cmd, _ = profile.get_test_cmd(task.row, f2p_only=False)
        await runtime.write("/eval.sh", build_eval_script(test_cmd).encode())
        await runtime.run(["chmod", "+x", "/eval.sh"], ENV)
        result = await runtime.run(
            ["sh", "-c", "bash /eval.sh > /test_output.txt 2>&1"], ENV
        )
        if result.exit_code > 1:
            raise vf.ProgramError(
                f"swesmith tests errored ({task.name}): exit={result.exit_code}"
            )
        output = await runtime.run(["cat", "/test_output.txt"], ENV)
        return output.stdout or ""

    async def revert_test_files(self, runtime: vf.Runtime, profile, row: dict) -> None:
        try:
            f2p_files, p2p_files = profile.get_test_files(row)
        except Exception:
            return
        test_files = [str(path) for path in list(f2p_files) + list(p2p_files)]
        if not test_files:
            return
        await runtime.run(
            [
                "sh",
                "-c",
                "git checkout -- " + " ".join(shlex.quote(f) for f in test_files),
            ],
            ENV,
        )

    async def validate(self, task: SWESmithTask, runtime: vf.Runtime) -> bool:
        await self.apply_gold_patch(task, runtime)
        return await self.solved(task, runtime) == 1.0

    async def apply_gold_patch(self, task: SWESmithTask, runtime: vf.Runtime) -> None:
        if not task.gold_patch.strip():
            raise vf.ProgramError(f"empty gold patch for {task.name}")
        await self.apply_patch(runtime, task.gold_patch, "gold", reverse=True)

    async def apply_patch(
        self, runtime: vf.Runtime, patch: str, label: str, reverse: bool = False
    ) -> None:
        path = f"/tmp/{label}.patch"
        await runtime.write(path, patch.encode())
        reverse_flag = "-R " if reverse else ""
        for cmd in (
            f"git apply --whitespace=fix {reverse_flag}{path}",
            f"patch --fuzz=5 -p1 {reverse_flag}-i {path}",
        ):
            result = await runtime.run(["sh", "-c", cmd], ENV)
            if result.exit_code == 0:
                return
        raise vf.ProgramError(
            f"{label} apply failed: exit={result.exit_code} "
            f"{result.stderr.strip()[-500:]}"
        )


def load_taskset(config: SWESmithConfig) -> SWESmithTaskset:
    return SWESmithTaskset(config)
