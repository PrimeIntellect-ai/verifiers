"""scaleswe-v1 — Scale-SWE (AweAI-Team/Scale-SWE) as a v1 taskset.

Each row ships a per-task Docker image with the repo checked out, `pre_commands` that
reset it to the base commit on a clean `scaleswe` branch, F2P/P2P pytest ids, and an
optional `f2p_patch` / `f2p_script` carrying the failing test. `setup` runs the row's
`pre_commands` in the live runtime before the agent (the runtime's workdir is the row's
repo). The `solved` reward restores the test files to base (the agent only fixes the
source), applies the f2p test, then runs the merged F2P+P2P ids through `score.py` —
which scores 1.0 iff every expected id passed. A v1 port of the v0 ComposableEnv
`ScaleSWETaskSet`.
"""

import json
from pathlib import Path

import verifiers.v1 as vf

DATASET = "AweAI-Team/Scale-SWE"

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
    "CI": "1",
}

# Restore the test files to the base commit so the agent's edits to the source are scored
# against the original tests (and tests it added are dropped). `$base` comes from the env.
RESTORE = r"""
git checkout "$base" -- tests/ test/ Test/ Tests/ 2>/dev/null || true
git ls-tree -r --name-only "$base" 2>/dev/null | while IFS= read -r path; do
  case "$path" in
    test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
      git checkout "$base" -- "$path" 2>/dev/null || true ;;
  esac
done
git ls-files 2>/dev/null | while IFS= read -r path; do
  case "$path" in
    tests/*|test/*|Test/*|Tests/*|test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
      if ! git cat-file -e "$base:$path" 2>/dev/null; then
        rm -f -- "$path"; git rm -q --cached -- "$path" 2>/dev/null || true
      fi ;;
  esac
done
"""

PATCH = "/tmp/scaleswe_f2p.patch"
SCORER = "/tmp/scaleswe_scorer.py"
SCORER_SRC = (Path(__file__).parent / "score.py").read_bytes()


def _ids(raw: str | list[str] | None) -> list[str]:
    """F2P/P2P ids arrive as a list or a JSON-encoded string; normalize to list[str]."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(t).strip() for t in parsed if t]
    return [parsed] if isinstance(parsed, str) and parsed else []


class ScaleSWETask(vf.Task):
    base_commit: str
    """Commit the repo is reset to before the agent runs and tests are scored against."""
    pre_commands: str
    """Shell run in the repo before the agent — resets to `base_commit` on branch `scaleswe`."""
    f2p_patch: str = ""
    """Optional patch adding the fail-to-pass test (applied before scoring)."""
    f2p_script: str = ""
    """Optional test file uploaded as `test_fail_to_pass.py` before scoring."""
    patch: str = ""
    """The gold source patch (reference solution); `validate` applies it to confirm the task
    is solvable. Not used during rollouts."""
    fail_to_pass: list[str] = []
    pass_to_pass: list[str] = []


class ScaleSWETaskset(vf.Taskset[ScaleSWETask, vf.TasksetConfig]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[ScaleSWETask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET, split="train")
        return [
            ScaleSWETask(
                idx=i,
                name=row["instance_id"],
                instruction=row["problem_statement"],
                image=row["image_url"],
                workdir=row["workdir"],
                resources=vf.Resources(cpu=4, memory=4, disk=10),
                base_commit=row.get("parent_commit") or row.get("base_commit") or "",
                pre_commands=(row.get("pre_commands") or "")
                .strip()
                .removesuffix("\\n"),
                f2p_patch=row.get("f2p_patch") or "",
                f2p_script=row.get("f2p_script") or "",
                patch=row.get("patch") or "",
                fail_to_pass=_ids(row.get("FAIL_TO_PASS")),
                pass_to_pass=_ids(row.get("PASS_TO_PASS")),
            )
            for i, row in enumerate(rows)
        ]

    async def setup(self, task: ScaleSWETask, runtime: vf.Runtime) -> None:
        result = await runtime.run(["sh", "-c", task.pre_commands], ENV)
        if result.exit_code != 0:
            raise vf.ProgramError(
                f"scaleswe setup failed ({task.name}): {result.stderr.strip()[-500:]}"
            )

    @vf.reward(weight=1.0)
    async def solved(self, task: ScaleSWETask, runtime: vf.Runtime) -> float:
        test_ids = task.fail_to_pass + task.pass_to_pass
        if not test_ids:
            return 0.0
        await runtime.run(["sh", "-c", RESTORE], {**ENV, "base": task.base_commit})
        if task.f2p_patch.strip():
            await self._apply_patch(runtime, task.f2p_patch)
        if task.f2p_script:
            await runtime.write("test_fail_to_pass.py", task.f2p_script.encode())
        await runtime.write(SCORER, SCORER_SRC)
        result = await runtime.run(["python", SCORER, json.dumps(test_ids)], ENV)
        lines = result.stdout.strip().splitlines()
        return float(lines[-1]) if lines else 0.0

    async def validate(self, task: ScaleSWETask, runtime: vf.Runtime) -> bool:
        """Valid iff the gold solution scores 1.0: apply the reference source patch, then run
        the same `solved` reward the agent is graded by (setup has reset the repo to base)."""
        await self.apply_gold_patch(task, runtime)
        return await self.solved(task, runtime) == 1.0

    async def apply_gold_patch(self, task: ScaleSWETask, runtime: vf.Runtime) -> None:
        """Apply the gold source patch (for validation), raising if no strategy takes."""
        if not task.patch.strip():
            raise vf.ProgramError(f"empty gold patch for {task.name}")
        if not await self._apply_patch(runtime, task.patch):
            raise vf.ProgramError(f"gold apply failed ({task.name})")

    async def _apply_patch(self, runtime: vf.Runtime, patch: str) -> bool:
        # Try strict, then whitespace-tolerant, then a fuzzy `patch`, then a partial
        # `--reject` apply (mirrors the v0 multi-strategy helper). Returns whether a strategy
        # applied cleanly: `solved` applies the test patch best-effort (scoring catches a
        # patch that didn't take), `apply_gold_patch` requires a clean apply.
        await runtime.write(PATCH, patch.encode())
        for cmd in (
            f"git apply --verbose {PATCH}",
            f"git apply --verbose --ignore-space-change --ignore-whitespace {PATCH}",
            f"patch --batch --fuzz=5 -p1 -i {PATCH}",
            f"git apply --verbose --reject --ignore-whitespace {PATCH}",
        ):
            if (await runtime.run(["sh", "-c", cmd], ENV)).exit_code == 0:
                return True
        return False


def load_taskset(config: vf.TasksetConfig) -> ScaleSWETaskset:
    return ScaleSWETaskset(config)
