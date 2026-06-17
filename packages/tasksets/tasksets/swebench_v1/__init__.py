"""swebench-v1 - SWE-bench Verified as a v1 taskset.

Rows are loaded from ``princeton-nlp/SWE-bench_Verified``. Each task runs in the
Prime-mirrored SWE-bench image, uses upstream ``swebench`` test specs to run
the canonical hidden-test eval script, and scores through the SWE-bench parser
against FAIL_TO_PASS and PASS_TO_PASS.
"""

import logging
from pathlib import Path

import verifiers.v1 as vf

# swebench imports call logging.basicConfig(); clear the eager handler so this
# example package does not unexpectedly change repo logging behavior.
import swebench  # noqa: F401

logging.root.handlers.clear()
logger = logging.getLogger(__name__)

DATASET = "princeton-nlp/SWE-bench_Verified"
REGISTRY = "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"
REPO_PATH = "/testbed"

ENV = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


def make_test_spec_with_retry(row: dict, namespace: str = "swebench"):
    """Build a SWE-bench TestSpec, retrying transient GitHub connection resets."""
    import tenacity as tc
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from swebench.harness.test_spec.test_spec import make_test_spec

    @tc.retry(
        retry=tc.retry_if_exception_type(RequestsConnectionError),
        stop=tc.stop_after_attempt(5),
        wait=tc.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def inner():
        return make_test_spec(row, namespace=namespace)

    return inner()


def get_logs_eval(test_spec, content: str) -> tuple[dict[str, str], bool]:
    """Custom SWE-bench log evaluator used by the composable taskset."""
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        MAP_REPO_VERSION_TO_SPECS,
        RESET_FAILED,
        TESTS_ERROR,
        TESTS_TIMEOUT,
    )
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    repo = test_spec.repo
    version = test_spec.version
    log_parser = MAP_REPO_TO_PARSER[repo]
    test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    bad_codes = [
        code
        for code in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT]
        if code in content
    ]
    if bad_codes:
        return {}, False

    return log_parser(content.split(test_cmd)[-1], test_spec), True


class SWEBenchTask(vf.Task):
    row: dict
    """Original dataset row, kept for upstream SWE-bench test-spec construction."""
    gold_patch: str = ""
    """Gold source patch for model-free validation."""


class SWEBenchConfig(vf.TasksetConfig):
    dataset_name: str = DATASET
    split: str | None = None
    skip_install: bool = True
    """Skip install when the agent changed only pure-Python files."""


class SWEBenchTaskset(vf.Taskset[SWEBenchTask, SWEBenchConfig]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[SWEBenchTask]:
        from datasets import load_dataset

        split = self.config.split
        if split is None:
            split = "test" if "bench" in self.config.dataset_name.lower() else "train"
        rows = load_dataset(self.config.dataset_name, split=split)
        tasks: list[SWEBenchTask] = []
        for i, row in enumerate(rows):
            row_dict = dict(row)
            test_spec = make_test_spec_with_retry(row_dict)
            tasks.append(
                SWEBenchTask(
                    idx=i,
                    name=row_dict.get("instance_id") or f"swebench-{i}",
                    instruction=row_dict["problem_statement"],
                    image=f"{REGISTRY}/{test_spec.instance_image_key}",
                    workdir=REPO_PATH,
                    resources=vf.Resources(cpu=4, memory=4, disk=10),
                    row=row_dict,
                    gold_patch=row_dict.get("patch") or "",
                )
            )
        return tasks

    async def setup(self, task: SWEBenchTask, runtime: vf.Runtime) -> None:
        for target in ("/testbed/.venv", "/root/.venv"):
            result = await runtime.run(
                ["sh", "-c", f"ln -sfn /opt/miniconda3/envs/testbed {target}"],
                ENV,
            )
            if result.exit_code != 0:
                raise vf.ProgramError(
                    f"swebench setup failed ({task.name}): "
                    f"target={target} {result.stderr.strip()[-500:]}"
                )

    @vf.reward(weight=1.0)
    async def solved(self, task: SWEBenchTask, runtime: vf.Runtime) -> float:
        output = await self.run_tests(task, runtime)
        return self.calculate_reward(output, task.row)

    async def run_tests(self, task: SWEBenchTask, runtime: vf.Runtime) -> str:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

        test_spec = make_test_spec_with_retry(task.row)
        eval_script = test_spec.eval_script

        if self.config.skip_install:
            try:
                diff_res = await runtime.run(
                    ["sh", "-c", "git -c core.fileMode=false diff --name-only HEAD"],
                    ENV,
                )
                if diff_res.exit_code != 0:
                    raise RuntimeError(f"diff failed: exit_code={diff_res.exit_code}")
                changed = [p.strip() for p in diff_res.stdout.splitlines() if p.strip()]
                compiled_exts = {
                    ".pyx",
                    ".pxd",
                    ".pxi",
                    ".c",
                    ".cc",
                    ".cpp",
                    ".h",
                    ".hpp",
                    ".f",
                    ".f90",
                }
                build_files = {
                    "setup.py",
                    "setup.cfg",
                    "pyproject.toml",
                    "meson.build",
                    "CMakeLists.txt",
                }
                needs_rebuild = any(
                    Path(path).name in build_files
                    or "/_build_utils/" in path
                    or "/__check_build/" in path
                    or Path(path).suffix in compiled_exts
                    for path in changed
                )
                if not needs_rebuild:
                    specs = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]
                    install_cmd = specs.get("install")
                    if install_cmd:
                        eval_lines = [
                            line
                            for line in test_spec.eval_script_list
                            if line.strip() != install_cmd.strip()
                        ]
                        eval_script = (
                            "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_lines)
                            + "\n"
                        )
            except Exception as e:
                logger.warning("keeping SWE-bench install step: %r", e)

        await runtime.write("/eval.sh", eval_script.encode())
        await runtime.run(["chmod", "+x", "/eval.sh"], ENV)
        result = await runtime.run(
            ["sh", "-c", "/eval.sh > /test_output.txt 2>&1"], ENV
        )
        if result.exit_code > 1:
            raise vf.ProgramError(
                f"swebench tests errored ({task.name}): exit={result.exit_code}"
            )
        output = await runtime.run(["cat", "/test_output.txt"], ENV)
        return output.stdout or ""

    def calculate_reward(self, test_output: str, row: dict) -> float:
        from swebench.harness.constants import (
            FAIL_ONLY_REPOS,
            FAIL_TO_PASS,
            KEY_INSTANCE_ID,
            PASS_TO_PASS,
            EvalType,
            ResolvedStatus,
        )
        from swebench.harness.grading import (
            get_eval_tests_report,
            get_resolution_status,
        )

        test_spec = make_test_spec_with_retry(row)
        eval_status_map, _ = get_logs_eval(test_spec, test_output)
        eval_ref = {
            KEY_INSTANCE_ID: test_spec.instance_id,
            FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: test_spec.PASS_TO_PASS,
        }
        eval_type = (
            EvalType.FAIL_ONLY
            if test_spec.repo in FAIL_ONLY_REPOS
            else EvalType.PASS_AND_FAIL
        )
        report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
        return float(get_resolution_status(report) == ResolvedStatus.FULL.value)

    async def validate(self, task: SWEBenchTask, runtime: vf.Runtime) -> bool:
        await self.apply_gold_patch(task, runtime)
        return await self.solved(task, runtime) == 1.0

    async def apply_gold_patch(self, task: SWEBenchTask, runtime: vf.Runtime) -> None:
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
            f"swebench gold apply failed ({task.name}): "
            f"exit={result.exit_code} {result.stderr.strip()[-500:]}"
        )


def load_taskset(config: SWEBenchConfig) -> SWEBenchTaskset:
    return SWEBenchTaskset(config)
