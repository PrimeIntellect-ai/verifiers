"""SWE-Smith multilingual TaskSet.

SWE-Smith (https://github.com/SWE-bench/SWE-smith) is a synthetic bug-fix
dataset published by the SWE-bench authors. The upstream project ships per-repo
``RepoProfile`` subclasses that define the test command and a language-specific
log parser; the harness runs the whole test suite, parses pass/fail per test,
then compares against ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` from the dataset row.

This taskset wraps that harness for ``ComposableEnv``:

* The dataset is loaded from ``SWE-bench/SWE-smith-{language}`` on HF.
* For each instance we look up the registered profile, run its ``test_cmd``
  inside ``/testbed`` with ``TEST_OUTPUT_START/END`` markers, then feed the
  captured output to ``profile.log_parser`` to get a ``test → status`` map.
* Reward is 1.0 iff every F2P and P2P test is in the PASSED set
  (``ResolvedStatus.FULL``).

C++ is special-cased: upstream currently registers profiles for ~20 of 69 repos
(~10% of rows), so we filter rows whose repo lacks a profile. All other
priority languages (py, go, java, js, ts, rs) have 100% coverage.
"""

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


# Broad PATH covering all known SWE-Smith language toolchains. Individual
# images only need a subset of these prefixes to exist for their own tooling.
ENV_VARS_SWE_SMITH = {
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


def _process_example(x: dict) -> dict:
    return {
        "question": x["problem_statement"],
        "info": {**x},
        "answer": "",
    }


def _get_profile(info: dict):
    """Return the SWE-Smith RepoProfile registered for this instance."""
    from swesmith.profiles import registry

    return registry.get_from_inst(info)


def _build_eval_script(test_cmd: str) -> str:
    """Wrap the profile's test_cmd with SWE-Smith's output markers.

    Mirrors ``swesmith.harness.utils.run_patch_in_container`` so that
    ``read_test_output`` (from upstream grading) can bound the parsed region
    exactly like upstream does.
    """
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START

    return dedent(
        f"""\
        #!/bin/bash
        set -uxo pipefail

        cd /testbed

        : '{TEST_OUTPUT_START}'
        {test_cmd}
        : '{TEST_OUTPUT_END}'
        """
    )


def _parse_status_map(output: str, profile) -> dict[str, str]:
    """Extract pass/fail per test by delegating to upstream's parser chain.

    The profile's ``log_parser`` expects the already-bounded output (between
    the two markers). We reuse upstream's ``read_test_output`` semantics via
    an in-memory temp file so we pick up its marker handling for free.
    """
    from swesmith.harness.grading import read_test_output

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(output)
        tmp = Path(f.name)
    try:
        bounded, found = read_test_output(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    if not found or not bounded:
        return {}
    return profile.log_parser(bounded)


def _is_resolved(
    status_map: dict[str, str], fail_to_pass: list[str], pass_to_pass: list[str]
) -> bool:
    """All F2P and P2P must be PASSED (XFAIL also counts, matching upstream)."""
    from swebench.harness.constants import TestStatus

    passed = {TestStatus.PASSED.value, TestStatus.XFAIL.value}
    tests = list(fail_to_pass) + list(pass_to_pass)
    if not tests:
        return False
    return all(status_map.get(t) in passed for t in tests)


class SWESmithRubric(vf.Rubric):
    """Scores SWE-Smith tasks by parsing test output and comparing to F2P/P2P."""

    def __init__(self, taskset: "SWESmithTaskSet", **kwargs):
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


class SWESmithTaskSet(SandboxTaskSet):
    """Multilingual SWE-Smith TaskSet, parameterized by language."""

    default_workdir = "/testbed"

    def __init__(
        self,
        language: str = "py",
        dataset_name: str | None = None,
        split: str = "train",
        filter_repos: list[str] | None = None,
        filter_fn: str | None = None,
        ds_num_proc: int | None = None,
        ds_keep_in_memory: bool = True,
        timeout_minutes: int | None = None,
    ):
        if language not in LANGUAGE_TO_DATASET:
            raise ValueError(
                f"Unknown SWE-Smith language: {language!r}. "
                f"Available: {sorted(LANGUAGE_TO_DATASET)}"
            )
        self.language = language
        self.dataset_name = dataset_name or LANGUAGE_TO_DATASET[language]
        self.split = split
        self.filter_repos = filter_repos
        self.ds_num_proc = ds_num_proc
        self.ds_keep_in_memory = ds_keep_in_memory
        self.timeout_minutes = timeout_minutes
        super().__init__(
            dataset=self._build_dataset,
            name=f"swe/swesmith-{language}",
            filter_fn=filter_fn,
        )

    def _build_dataset(self) -> Any:
        from swesmith.profiles import registry

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

        # Drop rows whose repo has no registered profile — needed for cpp where
        # upstream currently covers only ~20/69 repos. All other priority langs
        # have full coverage, so this is a no-op for them.
        def _has_profile(x: dict) -> bool:
            try:
                registry.get_from_inst(x)
                return True
            except KeyError:
                return False

        dataset = dataset.filter(_has_profile, **_kw)

        if self.filter_repos:
            filter_set = frozenset(self.filter_repos)
            dataset = dataset.filter(lambda x: x.get("repo") not in filter_set, **_kw)

        return dataset.map(_process_example, remove_columns=dataset.column_names, **_kw)

    def get_instruction(self, info: dict) -> str:
        return info["problem_statement"]

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec(
            image=info["image_name"],
            timeout_minutes=self.timeout_minutes,
        )

    def get_workdir(self, info: dict) -> str:
        return self.default_workdir

    def get_env_vars(self) -> dict[str, str]:
        return dict(ENV_VARS_SWE_SMITH)

    async def setup(self, state) -> None:
        """Prep the sandbox for the agent.

        SWE-Smith images contain the full mirror repo with one branch per
        instance. Each branch has two extra commits on top of ``main``:
        ``HEAD``: "Remove F2P Tests" (bug + tests deleted) and
        ``HEAD~1``: "Bug Patch" (bug with F2P tests visible). Upstream's
        ``run_patch_in_container`` checks out the instance branch then
        ``HEAD~1`` so the agent/eval starts with tests visible; we do the
        same here.

        Python images ship a conda ``testbed`` env; link it to ``/root/.venv``
        so the rlm harness finds Python there. Other languages don't need it.
        Ripgrep is pre-installed to short-circuit rlm's apt fallback (slow on
        debian-based SWE-Smith images).
        """
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        info = state["info"]
        instance_id = info["instance_id"]

        # Images are shallow clones that only track ``main`` by default; fetch
        # the full branch set so we can check out the instance branch.
        fetch = await sandbox_client.execute_command(
            sandbox_id,
            "git fetch origin",
            working_dir=self.default_workdir,
            timeout=120,
        )
        if fetch.exit_code != 0:
            stderr = (fetch.stderr or "")[:500]
            raise RuntimeError(
                f"git fetch failed: exit_code={fetch.exit_code} stderr={stderr}"
            )
        checkout = await sandbox_client.execute_command(
            sandbox_id,
            f"git checkout {instance_id}",
            working_dir=self.default_workdir,
            timeout=30,
        )
        if checkout.exit_code != 0:
            stderr = (checkout.stderr or "")[:500]
            raise RuntimeError(
                f"git checkout {instance_id} failed: exit_code={checkout.exit_code} stderr={stderr}"
            )

        # Land on the "Bug Patch" commit regardless of branch structure:
        # * 3-commit branches (py/go/java/...): HEAD = "Remove F2P Tests" →
        #   step back once so the F2P tests are visible.
        # * 2-commit branches (ts/cpp/php/...): HEAD is already "Bug Patch".
        head_msg = await sandbox_client.execute_command(
            sandbox_id,
            "git log -1 --format=%s",
            working_dir=self.default_workdir,
            timeout=15,
        )
        head_summary = (head_msg.stdout or "").strip()
        if "Remove F2P Tests" in head_summary:
            head1 = await sandbox_client.execute_command(
                sandbox_id,
                "git checkout HEAD~1",
                working_dir=self.default_workdir,
                timeout=30,
            )
            if head1.exit_code != 0:
                stderr = (head1.stderr or "")[:500]
                raise RuntimeError(
                    f"git checkout HEAD~1 failed: exit_code={head1.exit_code} stderr={stderr}"
                )

        if self.language == "py":
            for venv_src in ("/opt/miniconda3/envs/testbed", "/opt/conda/envs/testbed"):
                result = await sandbox_client.execute_command(
                    sandbox_id, f"[ -d {venv_src} ] && ln -sf {venv_src} /root/.venv"
                )
                if result.exit_code == 0:
                    break
            else:
                logger.warning(f"[{sandbox_id}] Could not create /root/.venv symlink")

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
        result = await sandbox_client.execute_command(
            sandbox_id, rg_install, timeout=120
        )
        if result.exit_code != 0:
            logger.warning(
                f"[{sandbox_id}] ripgrep install failed (exit={result.exit_code}); "
                f"rlm install will fall back to apt-get"
            )

    async def _revert_test_files(
        self, sandbox_client: Any, sandbox_id: str, profile, info: dict
    ) -> None:
        """Revert any changes to F2P/P2P test files.

        Mirrors upstream ``run_patch_in_container`` (SWE-Smith
        ``harness/utils.py``): after applying a patch the upstream harness
        does ``git checkout -- <test_files>`` so an agent that edited the
        test file to make it pass can't cheat the reward.

        Python and Go profiles override ``get_test_files``; other languages
        inherit the base implementation which returns ``([], [])``. The
        no-op case is harmless.
        """
        try:
            f2p_files, p2p_files = profile.get_test_files(info)
        except Exception as e:  # noqa: BLE001 — match upstream's best-effort
            logger.warning(f"[{sandbox_id}] get_test_files raised: {e}")
            return
        test_files = [str(p) for p in list(f2p_files) + list(p2p_files)]
        if not test_files:
            return
        await sandbox_client.execute_command(
            sandbox_id,
            "git checkout -- " + " ".join(shlex.quote(f) for f in test_files),
            working_dir=self.default_workdir,
            timeout=30,
        )

    async def _run_tests(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: dict,
        test_timeout: int,
    ) -> str:
        info = state["info"]
        profile = _get_profile(info)
        # Revert F2P/P2P test files before scoring so an agent's edits to
        # the tests don't leak into the reward. No-op when the profile
        # doesn't override ``get_test_files`` (all non-py/non-go langs).
        await self._revert_test_files(sandbox_client, sandbox_id, profile, info)
        test_cmd, _ = profile.get_test_cmd(info, f2p_only=False)
        eval_script = _build_eval_script(test_cmd)

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
        try:
            profile = _get_profile(info)
        except KeyError:
            return 0.0
        status_map = _parse_status_map(test_output, profile)
        if not status_map:
            return 0.0
        f2p = info.get("FAIL_TO_PASS") or []
        p2p = info.get("PASS_TO_PASS") or []
        return 1.0 if _is_resolved(status_map, f2p, p2p) else 0.0

    async def _apply_patch_file(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        patch: str,
        label: str,
        reverse: bool = False,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as f:
            f.write(patch)
            f.flush()
            local_path = f.name

        remote_path = f"/tmp/{label}.patch"
        try:
            await sandbox_client.upload_file(sandbox_id, remote_path, local_path)
        finally:
            Path(local_path).unlink(missing_ok=True)

        reverse_flag = "-R " if reverse else ""
        result = await sandbox_client.execute_command(
            sandbox_id,
            f"git apply --whitespace=fix {reverse_flag}{remote_path}",
            working_dir=self.default_workdir,
            timeout=30,
        )
        if result.exit_code != 0:
            patch_reverse = "-R " if reverse else ""
            result = await sandbox_client.execute_command(
                sandbox_id,
                f"patch --fuzz=5 -p1 {patch_reverse}-i {remote_path}",
                working_dir=self.default_workdir,
                timeout=30,
            )
            if result.exit_code != 0:
                stderr = (result.stderr or "")[:500]
                raise RuntimeError(
                    f"{label} apply failed: exit_code={result.exit_code} stderr={stderr}"
                )

    async def _apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: dict
    ) -> None:
        """Apply the 'gold solution' for SWE-Smith.

        Unlike SWE-bench, SWE-Smith's ``patch`` field is the **bug-introduction
        patch** (the diff in the "Bug Patch" commit). After ``setup()`` checks
        out ``HEAD~1`` — the bug commit — the correct solution is to
        *reverse-apply* the bug patch, which restores the initial (correct)
        code while keeping the F2P tests visible.
        """
        info = state["info"]
        patch = info.get("patch", "")
        if not patch or not patch.strip():
            raise RuntimeError("No gold patch in info['patch']")
        await self._apply_patch_file(
            sandbox_client, sandbox_id, patch, "gold", reverse=True
        )

    def get_rubric(self):
        return SWESmithRubric(self)

    async def validate_instance(self, state) -> bool:
        """Apply gold patch, run tests, check reward > 0."""
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await self._apply_gold_patch(sandbox_client, sandbox_id, state)
        test_output = await self._run_tests(
            sandbox_client, sandbox_id, state, state.get("test_timeout", 900)
        )
        state["test_output"] = test_output
        info = state.get("info") or {}
        return self._calculate_reward(test_output, info) > 0
