from __future__ import annotations

import logging
import shlex
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import verifiers as vf
from datasets import load_dataset
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet

logger = logging.getLogger(__name__)

REGISTRY_PREFIX = "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"

ENV_VARS_SWE_LEGO = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


def _process_example(x: dict) -> dict:
    return {
        "question": x["problem_statement"],
        "info": {**x},
        "answer": "",
    }


def _build_eval_script(fail_to_pass: list[str], pass_to_pass: list[str]) -> str:
    """Construct a bash eval script from FAIL_TO_PASS and PASS_TO_PASS test lists."""
    f2p_args = " ".join(shlex.quote(t) for t in fail_to_pass) if fail_to_pass else ""
    p2p_args = " ".join(shlex.quote(t) for t in pass_to_pass) if pass_to_pass else ""

    return dedent(
        f"""\
        #!/bin/bash
        set -uo pipefail

        cd /testbed

        set +u
        if [ -f /opt/miniconda3/bin/activate ]; then
            source /opt/miniconda3/bin/activate
        elif [ -f /opt/conda/bin/activate ]; then
            source /opt/conda/bin/activate
        fi
        conda activate testbed 2>/dev/null || true
        set -u

        FAIL=0

        {f"python -m pytest -x --tb=short {f2p_args} || FAIL=1" if f2p_args else "# No FAIL_TO_PASS tests"}

        {f"python -m pytest -x --tb=short {p2p_args} || FAIL=1" if p2p_args else "# No PASS_TO_PASS tests"}

        if [ "$FAIL" -eq 0 ]; then
            echo "SWELEGO_EXIT_CODE=0"
        else
            echo "SWELEGO_EXIT_CODE=1"
        fi

        exit "$FAIL"
        """
    )


class SWELegoRubric(vf.Rubric):
    """Scores SWE-Lego tasks by checking test exit code."""

    def __init__(self, taskset: "SWELegoTaskSet", **kwargs):
        super().__init__(**kwargs)
        self.taskset = taskset
        self.add_reward_func(self.solved)

    async def solved(self, state, info, **kwargs) -> float:
        if isinstance(state.get("error"), vf.InfraError):
            return 0.0
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            return 0.0
        try:
            test_output = await self.taskset._run_tests(
                sandbox_client, sandbox_id, state, state.get("test_timeout", 900)
            )
            state["test_output"] = test_output
        except Exception as e:
            logger.warning(f"Test execution failed: {e}")
            state["test_output"] = f"ERROR: {e}"
            return 0.0
        return float(self.taskset._calculate_reward(test_output, info))

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            try:
                await sandbox_client.delete(sandbox_id)
            except Exception:
                pass


class SWELegoTaskSet(SandboxTaskSet):
    """TaskSet for SWE-Lego/SWE-Lego-Synthetic-Data (SWE-smith instances).

    Uses public Docker Hub images (chaofantao/autocode07:swesmith.*) —
    no registry credentials required. Test execution is constructed at
    runtime from the dataset's FAIL_TO_PASS / PASS_TO_PASS fields.
    """

    default_workdir = "/testbed"

    def __init__(
        self,
        dataset_name: str = "SWE-Lego/SWE-Lego-Synthetic-Data",
        split: str = "resolved",
        filter_repos: list[str] | None = None,
        ds_num_proc: int | None = 4,
        ds_keep_in_memory: bool = True,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.filter_repos = filter_repos
        self.ds_num_proc = ds_num_proc
        self.ds_keep_in_memory = ds_keep_in_memory
        super().__init__(dataset=self._build_dataset(), name="swe/swelego")

    def _build_dataset(self) -> Any:
        _kw = dict(
            num_proc=self.ds_num_proc,
            keep_in_memory=self.ds_keep_in_memory,
            load_from_cache_file=False,
        )
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            keep_in_memory=self.ds_keep_in_memory,
            num_proc=self.ds_num_proc,
        )
        if self.filter_repos:
            filter_set = frozenset(self.filter_repos)
            dataset = dataset.filter(lambda x: x.get("repo") not in filter_set, **_kw)
        return dataset.map(_process_example, remove_columns=dataset.column_names, **_kw)

    def get_instruction(self, info: dict) -> str:
        return info["problem_statement"]

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        # Images must be mirrored to our registry before use.
        # Original: chaofantao/autocode07:swesmith.x86_64.<repo>
        # Mirrored: {REGISTRY_PREFIX}/chaofantao/autocode07:swesmith.x86_64.<repo>
        return SandboxSpec(image=f"{REGISTRY_PREFIX}/{info['image_name']}")

    def get_workdir(self, info: dict) -> str:
        return "/testbed"

    def get_env_vars(self) -> dict[str, str]:
        return dict(ENV_VARS_SWE_LEGO)

    async def setup(self, state) -> None:
        """Create /root/.venv symlink so agents can find the testbed env."""
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        for venv_src in ("/opt/miniconda3/envs/testbed", "/opt/conda/envs/testbed"):
            result = await sandbox_client.execute_command(
                sandbox_id, f"ln -sf {venv_src} /root/.venv"
            )
            if result.exit_code == 0:
                return
        logger.warning(f"[{sandbox_id}] Could not create /root/.venv symlink")

    async def _run_tests(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: dict,
        test_timeout: int,
    ) -> str:
        info = state["info"]
        fail_to_pass: list[str] = info.get("FAIL_TO_PASS") or []
        pass_to_pass: list[str] = info.get("PASS_TO_PASS") or []

        eval_script = _build_eval_script(fail_to_pass, pass_to_pass)

        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            f.write(eval_script)
            f.flush()
            local_path = f.name

        try:
            await sandbox_client.upload_file(sandbox_id, "/eval.sh", local_path)
        finally:
            Path(local_path).unlink(missing_ok=True)

        await sandbox_client.execute_command(sandbox_id, "chmod +x /eval.sh")
        env_str = " ".join(f"{k}={v}" for k, v in self.get_env_vars().items())
        command = f"export {env_str}; bash /eval.sh > /test_output.txt 2>&1"
        results = await sandbox_client.run_background_job(
            sandbox_id, command, timeout=test_timeout
        )
        if results.exit_code > 1:
            raise RuntimeError(f"Error running tests: exit_code={results.exit_code}")
        results = await sandbox_client.execute_command(
            sandbox_id, "cat /test_output.txt"
        )
        return results.stdout or ""

    def _calculate_reward(self, test_output: str, info: dict) -> float:
        if not test_output:
            return 0.0
        return 1.0 if "SWELEGO_EXIT_CODE=0" in test_output else 0.0

    async def _apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: dict
    ) -> None:
        info = state["info"]
        patch = info.get("patch", "")
        if not patch or not patch.strip():
            raise RuntimeError("No gold patch in info['patch']")

        with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as f:
            f.write(patch)
            f.flush()
            local_path = f.name

        try:
            await sandbox_client.upload_file(sandbox_id, "/tmp/gold.patch", local_path)
        finally:
            Path(local_path).unlink(missing_ok=True)

        result = await sandbox_client.execute_command(
            sandbox_id,
            "git apply --whitespace=fix /tmp/gold.patch",
            working_dir="/testbed",
            timeout=30,
        )
        if result.exit_code != 0:
            result = await sandbox_client.execute_command(
                sandbox_id,
                "patch --fuzz=5 -p1 -i /tmp/gold.patch",
                working_dir="/testbed",
                timeout=30,
            )
            if result.exit_code != 0:
                stderr = (result.stderr or "")[:500]
                raise RuntimeError(
                    f"Gold patch apply failed: exit_code={result.exit_code} stderr={stderr}"
                )

    def get_rubric(self):
        return SWELegoRubric(self)

    async def validate_instance(self, state) -> bool:
        """Apply gold patch, run tests, and check if reward > 0."""
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        try:
            await self._apply_gold_patch(sandbox_client, sandbox_id, state)
            test_output = await self._run_tests(
                sandbox_client, sandbox_id, state, state.get("test_timeout", 900)
            )
            state["test_output"] = test_output
            info = state.get("info") or {}
            return self._calculate_reward(test_output, info) > 0
        except Exception:
            return False
