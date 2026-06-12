"""swelego-v1 — SWE-Lego-Real-Data (PrimeIntellect/SWE-Lego-Real-Data) as a v1 taskset.

Each row is a real GitHub issue with a public Docker image (`jierun/sweb.eval.x86_64.*`)
carrying the repo at `/testbed` and a conda `testbed` env. `setup` links that env onto
`/root/.venv`, pre-installs a static ripgrep (the images' apt is ~150s slow and the rlm
harness installs `rg` via apt when missing), and applies the row's `test_patch` so the
agent can read the failing tests from t=0. The `solved` reward canonicalizes the test
files (revert agent edits from `base_commit`, re-apply `test_patch` — the SWE-bench
anti-tamper dance), runs the row's `test_cmd` verbatim (whole test FILES plus the flags
upstream's eval uses — module-scoped fixtures and `--cov-fail-under` thresholds break
when you cherry-pick ids), and scores 1.0 iff every FAIL_TO_PASS and PASS_TO_PASS id is
PASSED in the pytest `-rA` short summary. Scoring parsed outcomes (not pytest's exit
code) avoids false negatives where a repo-wide coverage threshold makes pytest exit
non-zero even though every scored test passed. A v1 port of the v0 ComposableEnv
`SWELegoTaskSet`.
"""

import re
from textwrap import dedent

import verifiers.v1 as vf

DATASET = "PrimeIntellect/SWE-Lego-Real-Data"
# A filtered fork of SWE-Lego/SWE-Lego-Real-Data that drops rows with truncated pytest
# parametrize test ids; `resolved` carries the rows whose gold patch validates.
SPLIT = "resolved"

REPO_PATH = "/testbed"

# The testbed conda env (with the project + pytest installed) and quiet, non-interactive
# tooling — exported for every command the taskset runs in the sandbox.
ENV = {
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

# Symlink the testbed conda env onto /root/.venv (checks both common conda roots;
# `ln -sfn` alone returns 0 even when the target is missing, so test `-d` explicitly).
LINK_VENV = (
    "for d in /opt/miniconda3/envs/testbed /opt/conda/envs/testbed; do "
    '[ -d "$d" ] && ln -sfn "$d" /root/.venv && break; done; true'
)

# Pre-install a static musl ripgrep. The jierun SWE-bench eval images have slow apt
# sources (`apt-get update` ~150s) and the rlm harness apt-installs rg when missing —
# the direct binary download skips that. Failure is non-fatal (the harness falls back).
INSTALL_RG = (
    "command -v rg >/dev/null 2>&1 || { "
    "V=14.1.1; "
    "curl -sSL --max-time 100 "
    "https://github.com/BurntSushi/ripgrep/releases/download/"
    "${V}/ripgrep-${V}-x86_64-unknown-linux-musl.tar.gz "
    "| tar xz -C /tmp "
    "&& install -m 755 /tmp/ripgrep-${V}-x86_64-unknown-linux-musl/rg /usr/local/bin/rg; "
    "}"
)

# Canonical SWE-bench grading dance, step 1: wipe agent edits to every file `$patch`
# (the row's test_patch) touches — `git checkout "$base" -- <path>` for files that
# existed at the base commit, `rm -f` for files the patch added. Reverts from `$base`
# (not HEAD): an agent that ran `git add && git commit` mid-rollout moved HEAD to its
# own — potentially weakened — test version. Step 2 (re-apply test_patch cleanly) runs
# right after, via `_apply_patch`. Agent *source* edits survive untouched.
REVERT_TEST_FILES = r"""
git apply --numstat "$patch" | cut -f3- | while IFS= read -r path; do
  path=${path#\"}; path=${path%\"}
  if git cat-file -e "$base:$path" 2>/dev/null; then
    git checkout "$base" -- "$path" 2>/dev/null || true
  else
    rm -f -- "$path"
  fi
done
"""

# Parametrized pytest ids can contain spaces (e.g.
# ``test_foo[TypeMismatch, List var assigned to String]``), so the id capture group is
# non-greedy up to an optional ``\s+-\s+<reason>`` tail that FAILED / ERROR / XFAIL
# emit. Plain `\S+` would truncate such ids at the first inner whitespace and silently
# fail to match F2P/P2P entries.
OUTCOME_LINE_RE = re.compile(
    r"^(PASSED|FAILED|ERROR|XFAIL|XPASS)\s+(.+?)(?:\s+-\s+.*)?$"
)


def parse_outcomes(output: str) -> dict[str, str]:
    """Parse pytest ``-rA`` short-summary lines into ``{test_id: outcome}``.

    Pytest with ``-rA`` prints one ``OUTCOME test_id`` line per test in the
    short-summary block (e.g. ``PASSED tests/foo.py::Cls::test``); if pytest re-runs
    anything, later lines win. ``SKIPPED`` is intentionally not parsed: pytest prints
    it as ``SKIPPED [N] <file>:<line>: <reason>`` without a usable test id, so it can't
    be matched against FAIL_TO_PASS / PASS_TO_PASS regardless — a skipped F2P/P2P test
    correctly scores 0 via "no PASSED entry".
    """
    outcomes: dict[str, str] = {}
    for line in output.splitlines():
        m = OUTCOME_LINE_RE.match(line)
        if m:
            # Non-greedy capture can leave trailing whitespace — strip before storing
            # so dict lookups against F2P/P2P entries match exactly.
            outcomes[m.group(2).rstrip()] = m.group(1)
    return outcomes


def calculate_reward(test_output: str, required: list[str]) -> float:
    """1.0 iff every required (FAIL_TO_PASS + PASS_TO_PASS) test id is PASSED.

    No parsed outcomes at all usually means the row's ``test_cmd`` doesn't emit
    ``-rA`` short-summary lines — that scores 0, like any missing id.
    """
    outcomes = parse_outcomes(test_output)
    if not outcomes or not required:
        return 0.0
    return 1.0 if all(outcomes.get(t) == "PASSED" for t in required) else 0.0


def build_eval_script(test_cmd: str) -> str:
    """Wrap the row's canonical ``test_cmd`` (whole test-file paths plus the flags the
    repo's config expects: ``LANG=C.UTF-8``, ``-p no:cacheprovider``, sometimes
    ``--cov=pkg``) in conda activation. Exits 0 regardless of the pytest outcome —
    scoring reads the ``-rA`` summary, so a non-zero exit from here means infra, not a
    failing test."""
    return dedent(
        f"""\
        #!/bin/bash
        set -uo pipefail

        cd {REPO_PATH}

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


class SWELegoTask(vf.Task):
    test_cmd: str
    """Canonical pytest invocation for scoring — run verbatim, outcomes parsed via `-rA`."""
    test_patch: str = ""
    """Patch adding the failing tests; applied in `setup`, re-canonicalized before scoring."""
    base_commit: str = ""
    """Commit test files are reverted to before `test_patch` is re-applied at scoring."""
    fail_to_pass: list[str] = []
    pass_to_pass: list[str] = []
    gold_patch: str = ""
    """Gold source patch (for validation/dummy rollouts)."""


class SWELegoTaskset(vf.Taskset[SWELegoTask, vf.TasksetConfig]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[SWELegoTask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET, split=SPLIT)
        return [
            SWELegoTask(
                idx=i,
                name=row.get("instance_id") or f"swelego-{i}",
                instruction=row["problem_statement"],
                image=row["image_name"],
                workdir=REPO_PATH,
                resources=vf.Resources(cpu=4, memory=4, disk=10),
                test_cmd=row.get("test_cmd") or "",
                test_patch=row.get("test_patch") or "",
                base_commit=row.get("base_commit") or "",
                fail_to_pass=list(row.get("FAIL_TO_PASS") or []),
                pass_to_pass=list(row.get("PASS_TO_PASS") or []),
                gold_patch=row.get("patch") or "",
            )
            for i, row in enumerate(rows)
        ]

    async def setup(self, task: SWELegoTask, runtime: vf.Runtime) -> None:
        await runtime.run(["sh", "-c", LINK_VENV], ENV)
        await runtime.run(["sh", "-c", INSTALL_RG], ENV)
        # The failing tests live in a separate test_patch — apply it before the agent
        # runs so it can read them (and so gold-patch validation scores against them).
        if task.test_patch.strip():
            await self._apply_patch(runtime, task.test_patch, "test_patch")

    @vf.reward(weight=1.0)
    async def solved(self, task: SWELegoTask, runtime: vf.Runtime) -> float:
        required = task.fail_to_pass + task.pass_to_pass
        if not required or not task.test_cmd.strip():
            return 0.0
        if task.test_patch.strip() and task.base_commit:
            # Re-write the patch from the task (the agent could have tampered with the
            # copy in /tmp), revert its files to base, and re-apply it cleanly.
            await runtime.write("/tmp/test_patch.patch", task.test_patch.encode())
            await runtime.run(
                ["sh", "-c", REVERT_TEST_FILES],
                {**ENV, "patch": "/tmp/test_patch.patch", "base": task.base_commit},
            )
            try:
                await self._apply_patch(runtime, task.test_patch, "test_patch")
            except vf.ProgramError:
                # The agent broke the tree badly enough that the canonical tests can't
                # come back — that's a failed task, not an infra error.
                return 0.0
        result = await runtime.run(
            ["bash", "-c", build_eval_script(task.test_cmd)], ENV
        )
        return calculate_reward(result.stdout or "", required)

    async def apply_gold_patch(self, task: SWELegoTask, runtime: vf.Runtime) -> None:
        """Apply the gold source patch (for validation/dummy rollouts); `setup` already
        applied `test_patch`, so the F2P tests are in place."""
        if not task.gold_patch.strip():
            raise vf.ProgramError(f"empty gold patch for {task.name}")
        await self._apply_patch(runtime, task.gold_patch, "gold")

    async def _apply_patch(self, runtime: vf.Runtime, patch: str, label: str) -> None:
        # Strict apply first, then a fuzzy `patch` fallback for slightly-stale hunks
        # (mirrors the v0 taskset's two-strategy apply).
        path = f"/tmp/{label}.patch"
        await runtime.write(path, patch.encode())
        for cmd in (
            f"git apply --whitespace=fix {path}",
            f"patch --batch --fuzz=5 -p1 -i {path}",
        ):
            result = await runtime.run(["sh", "-c", cmd], ENV)
            if result.exit_code == 0:
                return
        raise vf.ProgramError(
            f"{label} apply failed: exit={result.exit_code} {result.stderr.strip()[-500:]}"
        )


def load_taskset(config: vf.TasksetConfig) -> SWELegoTaskset:
    return SWELegoTaskset(config)
