from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import verifiers as vf
from datasets import load_dataset
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet

from verifiers.envs.experimental.composable.tasksets.swe._test_patch import (
    revert_and_reapply_test_patch,
)

logger = logging.getLogger(__name__)

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


# Parametrized pytest ids can contain spaces (e.g.
# ``test_foo[TypeMismatch, List var assigned to String]``), so the id
# capture group is non-greedy up to an optional ``\s+-\s+<reason>`` tail
# that FAILED / ERROR / XFAIL emit. Plain `\S+` would truncate such ids
# at the first inner whitespace and silently fail to match F2P/P2P entries.
_OUTCOME_LINE_RE = re.compile(
    r"^(PASSED|FAILED|ERROR|XFAIL|XPASS)\s+(.+?)(?:\s+-\s+.*)?$"
)


def _parse_outcomes(output: str) -> dict[str, str]:
    """Parse pytest ``-rA`` short-summary lines into ``{test_id: outcome}``.

    Pytest with ``-rA`` prints one ``OUTCOME test_id`` line per test in the
    short-summary block (e.g. ``PASSED tests/foo.py::Cls::test``). We only
    need the *last* outcome per test id — if pytest re-runs anything, later
    lines win.

    ``SKIPPED`` is intentionally not parsed here: pytest prints it as
    ``SKIPPED [N] <file>:<line>: <reason>`` without a usable test id, so
    it can't be matched against FAIL_TO_PASS / PASS_TO_PASS regardless.
    A skipped F2P/P2P test correctly scores 0 via "no PASSED entry".
    """
    outcomes: dict[str, str] = {}
    for line in output.splitlines():
        m = _OUTCOME_LINE_RE.match(line)
        if m:
            # Non-greedy capture can leave trailing whitespace — strip before
            # storing so dict lookups against F2P/P2P entries match exactly.
            outcomes[m.group(2).rstrip()] = m.group(1)
    return outcomes


def _build_eval_script(test_cmd: str) -> str:
    """Build a bash wrapper that runs the dataset's per-row ``test_cmd``.

    SWE-Lego-Real-Data ships a canonical pytest invocation per row in
    ``info['test_cmd']`` (points at whole test FILE paths and carries all
    the flags upstream's eval uses: ``LANG=C.UTF-8``, ``-p no:cacheprovider``,
    ``-W ignore::DeprecationWarning``, occasionally ``--cov=pkg`` etc.).

    Running the whole file — instead of cherry-picking FAIL_TO_PASS /
    PASS_TO_PASS IDs — matters because:

    * Many pytest fixtures are module-scoped; isolating a test ID skips
      setup that the test depends on (shows up as spurious P2P failures).
    * Repos with ``[tool.pytest.ini_options] addopts = "--cov-fail-under=N"``
      can't meet threshold when you run a subset → pytest exits non-zero
      even though every test passed. The ``--cov=`` flag in ``test_cmd``
      keeps the threshold check working.
    * Parametrize IDs with whitespace/special chars are sometimes
      unparseable as CLI args; the whole-file run avoids the issue.

    We still use FAIL_TO_PASS and PASS_TO_PASS after the run — just via
    parsing the ``-rA`` outcomes (see ``_parse_outcomes``).
    """
    if not test_cmd or not test_cmd.strip():
        raise ValueError("test_cmd is empty — cannot score this instance.")

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

        set +e
        {test_cmd}
        PYTEST_EXIT=$?
        set -e

        echo ""
        echo "SWELEGO_PYTEST_EXIT=$PYTEST_EXIT"
        exit 0
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
    """TaskSet for SWE-Lego-Real-Data (real GitHub issues, public Docker images).

    Defaults to ``PrimeIntellect/SWE-Lego-Real-Data`` — a filtered fork of
    ``SWE-Lego/SWE-Lego-Real-Data`` that drops rows with truncated pytest
    parametrize test IDs. Images come from ``jierun/sweb.eval.x86_64.*``
    (public on Docker Hub).

    **Evaluation strategy.** Each row ships an ``info['test_cmd']`` — the
    canonical pytest invocation upstream uses (whole test-file paths, not
    specific IDs, plus the flags the repo's ``pyproject.toml`` /
    ``setup.cfg`` expect: ``LANG=C.UTF-8``, ``-p no:cacheprovider``,
    ``-W ignore::DeprecationWarning``, sometimes ``--cov=pkg``). We run
    that verbatim, then parse the ``-rA`` output for FAIL_TO_PASS /
    PASS_TO_PASS outcomes. Scoring on parsed outcomes (not pytest exit
    code) avoids false negatives where a repo-wide coverage threshold
    makes pytest exit non-zero even though every scored test passed.

    ``test_patch`` (adds the failing tests to the repo) is applied in
    ``setup()`` so both gold-patch validation and live agent rollouts
    score against the correct tests.
    """

    default_workdir = "/testbed"

    def __init__(
        self,
        dataset_name: str = "PrimeIntellect/SWE-Lego-Real-Data",
        split: str = "resolved",
        filter_repos: list[str] | None = None,
        filter_fn: str | None = None,
        ds_num_proc: int | None = None,
        ds_keep_in_memory: bool = True,
        timeout_minutes: int = 60,
    ):
        """
        Args:
            filter_fn: Optional Python expression string forwarded to
                :class:`TaskSet` — see its docstring. Applied to
                post-``_process_example`` rows, so predicates see the
                ``{"question", "info", "answer", ...}`` shape (e.g.
                ``"lambda x: x['info']['repo'] == 'googleapis/python-bigquery'"``).
        """
        self.dataset_name = dataset_name
        self.split = split
        self.filter_repos = filter_repos
        self.ds_num_proc = ds_num_proc
        self.ds_keep_in_memory = ds_keep_in_memory
        self.timeout_minutes = timeout_minutes
        super().__init__(
            dataset=self._build_dataset,
            name="swe/swelego",
            filter_fn=filter_fn,
        )

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
        # Some datasets (e.g. Real-Data) have struct columns with variable sub-keys
        # (e.g. install_config.JUPYTER_PLATFORM_DIRS). Arrow can't infer a consistent
        # schema across batches, so pre-serialize them to JSON strings.
        struct_cols = [
            col
            for col, feat in dataset.features.items()
            if hasattr(feat, "__class__")
            and feat.__class__.__name__
            not in (
                "Value",
                "Sequence",
                "ClassLabel",
                "Translation",
                "List",
            )
            and not isinstance(feat, list)
        ]
        if struct_cols:
            dataset = dataset.map(
                lambda x: {
                    col: json.dumps(x[col]) if x[col] is not None else None
                    for col in struct_cols
                },
                num_proc=None,
                load_from_cache_file=False,
            )
        return dataset.map(_process_example, remove_columns=dataset.column_names, **_kw)

    def get_instruction(self, info: dict) -> str:
        return info["problem_statement"]

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec(
            image=info["image_name"],
            timeout_minutes=self.timeout_minutes,
        )

    def get_workdir(self, info: dict) -> str:
        return "/testbed"

    def get_env_vars(self) -> dict[str, str]:
        return dict(ENV_VARS_SWE_LEGO)

    async def setup(self, state) -> None:
        """Prep the sandbox before the agent runs:

        1. Link ``/root/.venv`` to the testbed conda env (checks both common
           paths; ``ln -sf`` returns 0 even when the target is missing so we
           must test ``-d`` explicitly).
        2. Install ripgrep via a direct binary download. SWE-bench eval images
           (``jierun/*``) have slow apt sources (``apt-get update`` ~150s); the
           rlm harness installs rg via apt when missing, so we pre-install the
           static musl binary to skip apt.
        3. Apply ``test_patch`` from the dataset. SWE-bench instances carry
           the failing tests in a separate patch that must be applied before
           pytest can collect them — required for both gold-patch validation
           and live agent scoring.
        """
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]

        linked = False
        for venv_src in ("/opt/miniconda3/envs/testbed", "/opt/conda/envs/testbed"):
            result = await sandbox_client.execute_command(
                sandbox_id, f"[ -d {venv_src} ] && ln -sf {venv_src} /root/.venv"
            )
            if result.exit_code == 0:
                linked = True
                break
        if not linked:
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

        test_patch = (state.get("info") or {}).get("test_patch") or ""
        if test_patch.strip():
            await self._apply_patch_file(
                sandbox_client, sandbox_id, test_patch, "test_patch"
            )

    async def _run_tests(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: dict,
        test_timeout: int,
    ) -> str:
        info = state["info"]
        workdir = self.get_workdir(info)

        # Canonical SWE-bench grading dance: revert any agent edits to
        # files touched by ``test_patch`` (``git checkout HEAD -- <path>``
        # for pre-existing files, ``rm -f <path>`` for newly-added ones),
        # then re-apply ``test_patch`` cleanly. Closes the reward-hack
        # where an agent weakens F2P assertions mid-rollout. Agent source
        # edits are untouched — only test-file bits get canonicalized.
        test_patch: str = info.get("test_patch") or ""
        base_commit: str = info.get("base_commit") or ""
        if test_patch.strip() and base_commit:

            async def _apply(
                sc: Any, sid: str, wd: str, patch: str, label: str
            ) -> None:
                del wd  # _apply_patch_file hard-codes /testbed internally
                await self._apply_patch_file(sc, sid, patch, label)

            await revert_and_reapply_test_patch(
                sandbox_client,
                sandbox_id,
                workdir,
                test_patch,
                base_commit,
                apply_patch=_apply,
            )

        fail_to_pass: list[str] = info.get("FAIL_TO_PASS") or []
        pass_to_pass: list[str] = info.get("PASS_TO_PASS") or []
        test_cmd: str = info.get("test_cmd") or ""

        if not fail_to_pass and not pass_to_pass:
            logger.warning(
                f"[{sandbox_id}] No tests to run: FAIL_TO_PASS and PASS_TO_PASS both empty"
            )
            return "ERROR: no tests to score (FAIL_TO_PASS and PASS_TO_PASS both empty)"

        if not test_cmd.strip():
            raise RuntimeError(
                f"[{sandbox_id}] Dataset row missing 'test_cmd' — cannot score."
            )

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
        """Score 1.0 iff every FAIL_TO_PASS and PASS_TO_PASS test is PASSED.

        Parses pytest ``-rA`` short-summary lines (``_parse_outcomes``)
        rather than trusting pytest's overall exit code — a single repo-wide
        threshold (``--cov-fail-under``) can make pytest exit non-zero even
        when every scored test passed. Scoring the specific F2P/P2P IDs we
        care about is both stricter and more accurate.
        """
        if not test_output:
            return 0.0
        fail_to_pass: list[str] = info.get("FAIL_TO_PASS") or []
        pass_to_pass: list[str] = info.get("PASS_TO_PASS") or []
        if not fail_to_pass and not pass_to_pass:
            return 0.0

        outcomes = _parse_outcomes(test_output)
        if not outcomes:
            # No short-summary lines parsed. Most likely cause: the row's
            # ``test_cmd`` doesn't include a ``-r<letters>`` flag covering
            # the outcomes we need (``-rA`` is the canonical choice).
            # Surface loudly instead of silently scoring 0.
            logger.warning(
                "SWELego _parse_outcomes returned no outcomes — does test_cmd "
                "emit `-rA` short-summary lines? instance=%r test_output head=%r",
                info.get("instance_id"),
                test_output[:200],
            )
            return 0.0
        required = list(fail_to_pass) + list(pass_to_pass)
        return 1.0 if all(outcomes.get(t) == "PASSED" for t in required) else 0.0

    async def _apply_patch_file(
        self, sandbox_client: Any, sandbox_id: str, patch: str, label: str
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

        result = await sandbox_client.execute_command(
            sandbox_id,
            f"git apply --whitespace=fix {remote_path}",
            working_dir="/testbed",
            timeout=30,
        )
        if result.exit_code != 0:
            result = await sandbox_client.execute_command(
                sandbox_id,
                f"patch --fuzz=5 -p1 -i {remote_path}",
                working_dir="/testbed",
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
        """Apply the gold code patch.

        ``test_patch`` is applied separately in ``setup()`` so both the
        gold-patch validation path and the live agent scoring path see the
        F2P tests.
        """
        info = state["info"]
        patch = info.get("patch", "")
        if not patch or not patch.strip():
            raise RuntimeError("No gold patch in info['patch']")
        await self._apply_patch_file(sandbox_client, sandbox_id, patch, "gold")

    def get_rubric(self):
        return SWELegoRubric(self)

    async def validate_instance(self, state) -> bool:
        """Apply gold patch, run tests, and check if reward > 0.

        Exceptions propagate to the caller (``TaskSet.validate``) so that
        ``CommandTimeoutError`` / ``vf.InfraError`` / gold-apply failures
        can be classified correctly by their type instead of being flattened
        into ``test_failed``. Agent rollouts use ``SWELegoRubric.solved``
        (not this method), which keeps its own try/except so a transient
        failure still scores 0 rather than crashing the rollout.
        """
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await self._apply_gold_patch(sandbox_client, sandbox_id, state)
        test_output = await self._run_tests(
            sandbox_client, sandbox_id, state, state.get("test_timeout", 900)
        )
        state["test_output"] = test_output
        info = state.get("info") or {}
        return self._calculate_reward(test_output, info) > 0
