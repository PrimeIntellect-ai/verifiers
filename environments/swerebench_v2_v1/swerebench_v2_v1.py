"""swerebench-v2-v1 - SWE-rebench-V2 as a v1 taskset.

SWE-rebench-V2 rows carry a task image, gold source patch, validating
``test_patch``, and an ``install_config`` naming a language-specific parser. The
reward restores validating test files, re-applies the test patch, runs the row's
test command, and compares parser output to FAIL_TO_PASS and PASS_TO_PASS.
"""

import json
import re
import shlex

import verifiers.v1 as vf
from verifiers.envs.experimental.composable.tasksets.swe.shared.test_patch import (
    get_modified_files,
    get_new_files,
)
from verifiers.envs.experimental.composable.tasksets.swe.swe_rebench_v2 import (
    log_parsers,
)

DATASET = "PrimeIntellect/SWE-rebench-V2"

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
    "_JAVA_OPTIONS": "-Djava.net.preferIPv6Addresses=false",
}

TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def normalize_test_name(name: str) -> str:
    for pattern in TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def repo_workdir(repo: str) -> str:
    if "/" not in repo:
        raise ValueError(f"expected owner/repo, got {repo!r}")
    return f"/{repo.split('/', 1)[1]}"


def extract_install_config(config: dict | str) -> dict:
    if isinstance(config, str):
        return json.loads(config)
    return dict(config)


def build_eval_script(test_cmds: list[str]) -> str:
    lines = [
        "#!/bin/bash",
        "set -uo pipefail",
        "",
        'echo "SWEREBENCH_V2_TEST_OUTPUT_START"',
        "FAIL=0",
    ]
    for command in test_cmds:
        lines.append(f"{command} || FAIL=1")
    lines += [
        'echo "SWEREBENCH_V2_TEST_OUTPUT_END"',
        "",
        'exit "$FAIL"',
        "",
    ]
    return "\n".join(lines)


def is_resolved(
    status_map: dict[str, str], fail_to_pass: list[str], pass_to_pass: list[str]
) -> bool:
    normalized = {normalize_test_name(k): v for k, v in status_map.items()}
    expected = [normalize_test_name(t) for t in fail_to_pass + pass_to_pass]
    if not expected:
        return False
    return all(
        normalized.get(t) == log_parsers.TestStatus.PASSED.value for t in expected
    )


class SWERebenchV2Task(vf.Task):
    install_config: dict | str
    """Row install_config with test command and parser name."""
    base_commit: str = ""
    test_patch: str = ""
    gold_patch: str = ""
    fail_to_pass: list[str] = []
    pass_to_pass: list[str] = []


class SWERebenchV2Config(vf.TasksetConfig):
    dataset_name: str = DATASET
    split: str = "train"


class SWERebenchV2Taskset(vf.Taskset[SWERebenchV2Task, SWERebenchV2Config]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[SWERebenchV2Task]:
        from datasets import load_dataset

        rows = load_dataset(self.config.dataset_name, split=self.config.split)
        return [
            SWERebenchV2Task(
                idx=i,
                name=row.get("instance_id") or f"swerebench-v2-{i}",
                instruction=row["problem_statement"],
                image=row["image_name"],
                workdir=repo_workdir(row["repo"]),
                resources=vf.Resources(cpu=4, memory=4, disk=10),
                install_config=row["install_config"],
                base_commit=row.get("base_commit") or "",
                test_patch=row.get("test_patch") or "",
                gold_patch=row.get("patch") or "",
                fail_to_pass=list(row.get("FAIL_TO_PASS") or []),
                pass_to_pass=list(row.get("PASS_TO_PASS") or []),
            )
            for i, row in enumerate(rows)
        ]

    async def setup(self, task: SWERebenchV2Task, runtime: vf.Runtime) -> None:
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
        if task.test_patch.strip():
            await self.apply_patch(runtime, task.test_patch, "test_patch")

    @vf.reward(weight=1.0)
    async def solved(self, task: SWERebenchV2Task, runtime: vf.Runtime) -> float:
        output = await self.run_tests(task, runtime)
        return self.calculate_reward(output, task)

    async def run_tests(self, task: SWERebenchV2Task, runtime: vf.Runtime) -> str:
        config = extract_install_config(task.install_config)
        raw_test_cmd = config.get("test_cmd")
        if isinstance(raw_test_cmd, str):
            test_cmds = [raw_test_cmd]
        elif isinstance(raw_test_cmd, list):
            test_cmds = [c for c in raw_test_cmd if isinstance(c, str) and c.strip()]
        else:
            raise vf.ProgramError(
                f"install_config.test_cmd unsupported type: {type(raw_test_cmd)}"
            )
        if not test_cmds:
            raise vf.ProgramError("install_config.test_cmd is empty")

        if task.test_patch.strip() and task.base_commit:
            await self.revert_and_reapply_test_patch(task, runtime)

        await runtime.write("/eval.sh", build_eval_script(test_cmds).encode())
        await runtime.run(["chmod", "+x", "/eval.sh"], ENV)
        result = await runtime.run(
            ["sh", "-c", "bash /eval.sh > /test_output.txt 2>&1"], ENV
        )
        if result.exit_code > 1:
            raise vf.ProgramError(
                f"SWE-rebench-V2 tests errored ({task.name}): exit={result.exit_code}"
            )
        output = await runtime.run(["cat", "/test_output.txt"], ENV)
        return output.stdout or ""

    def calculate_reward(self, test_output: str, task: SWERebenchV2Task) -> float:
        if not test_output:
            return 0.0
        config = extract_install_config(task.install_config)
        parser_name = config.get("log_parser")
        parser = log_parsers.NAME_TO_PARSER.get(parser_name)
        if parser is None:
            parser = getattr(log_parsers, str(parser_name), None)
        if parser is None:
            return 0.0

        region = test_output
        start = "SWEREBENCH_V2_TEST_OUTPUT_START"
        end = "SWEREBENCH_V2_TEST_OUTPUT_END"
        if start in region:
            region = region.split(start, 1)[1]
        if end in region:
            region = region.rsplit(end, 1)[0]
        try:
            status_map = parser(region) or {}
        except Exception:
            return 0.0
        if not status_map:
            return 0.0
        return (
            1.0
            if is_resolved(status_map, task.fail_to_pass, task.pass_to_pass)
            else 0.0
        )

    async def revert_and_reapply_test_patch(
        self, task: SWERebenchV2Task, runtime: vf.Runtime
    ) -> None:
        modified = get_modified_files(task.test_patch)
        new = get_new_files(task.test_patch)
        if modified:
            paths = " ".join(shlex.quote(path) for path in modified)
            await runtime.run(
                [
                    "sh",
                    "-c",
                    f"git checkout {shlex.quote(task.base_commit)} -- {paths}",
                ],
                ENV,
            )
        if new:
            paths = " ".join(shlex.quote(path) for path in new)
            await runtime.run(["sh", "-c", f"rm -f {paths}"], ENV)
        await self.apply_patch(runtime, task.test_patch, "test_patch_reapply")

    async def apply_patch(
        self, runtime: vf.Runtime, patch: str, label: str, reverse: bool = False
    ) -> None:
        path = f"/tmp/{label}.patch"
        await runtime.write(path, patch.encode())
        reverse_flag = "-R " if reverse else ""
        git_cmd = (
            "git apply -v --3way --recount --ignore-space-change "
            f"--whitespace=nowarn {reverse_flag}{path}"
        )
        for cmd in (git_cmd, f"patch --fuzz=5 -p1 {reverse_flag}-i {path}"):
            result = await runtime.run(["sh", "-c", cmd], ENV)
            if result.exit_code == 0:
                return
        raise vf.ProgramError(
            f"{label} apply failed: exit={result.exit_code} "
            f"{result.stderr.strip()[-500:]}"
        )

    async def validate(self, task: SWERebenchV2Task, runtime: vf.Runtime) -> bool:
        await self.apply_gold_patch(task, runtime)
        return await self.solved(task, runtime) == 1.0

    async def apply_gold_patch(
        self, task: SWERebenchV2Task, runtime: vf.Runtime
    ) -> None:
        if not task.gold_patch.strip():
            raise vf.ProgramError(f"empty gold patch for {task.name}")
        await self.apply_patch(runtime, task.gold_patch, "gold")


def load_taskset(config: SWERebenchV2Config) -> SWERebenchV2Taskset:
    return SWERebenchV2Taskset(config)
