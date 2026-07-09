"""lean: a reusable Lean 4 theorem-proving taskset base.

Loads a Lean dataset (any HuggingFace dataset with a formal-statement column),
plants a ``sorry`` starter file in a Mathlib sandbox at ``setup``, and rewards a
clean ``lake env lean`` compile — with a reward-hacking guard that pins the
original theorem signature — at scoring time. It exposes **no tools**: the agent
reads, edits, and compiles ``proof.lean`` through whichever shell-capable harness
drives the rollout (``mini-swe-agent`` / ``rlm`` / ``bash``), so a lean task runs
under ANY harness.

This is a *base*: point it at a dataset via ``--taskset.dataset-name`` (+ column
config), or subclass it in a thin per-dataset package that just sets the config
defaults — the way ``swebench-verified-v1`` etc. wrap ``HarborTaskset``. The
per-dataset packages live in research-environments.
"""

from __future__ import annotations

import shlex

from pydantic_config import BaseConfig

from verifiers.v1.decorators import reward
from verifiers.v1.runtimes import Runtime
from verifiers.v1.state import State
from verifiers.v1.task import Task, TaskData, TaskResources
from verifiers.v1.taskset import TaskConfig, Taskset, TasksetConfig
from verifiers.v1.tasksets.lean.scoring import (
    build_starter_file,
    expected_protected_signature,
    parse_compile_output,
    protected_signature_substring_present,
)
from verifiers.v1.trace import Trace

# Default sandbox image: Lean v4.27 + Mathlib v4.27 (PI Research team registry).
# A per-dataset package overrides ``docker_image`` with its version-matched image.
DEFAULT_DOCKER_IMAGE = "team-clyvldofb0000gg1kx39rgzjq/lean-tactic:mathlib-v4.27.0-v3"
LEAN_PROJECT_PATH = "/workspace/mathlib4"
PROOF_FILE_PATH = "/tmp/proof.lean"

DEFAULT_SYSTEM_PROMPT = "You are an expert Lean 4 theorem prover working with Mathlib."


class LeanDatasetConfig(BaseConfig):
    """Which HuggingFace dataset to load and how to read its columns."""

    name: str
    """HuggingFace dataset id (required; each per-dataset package sets it)."""
    split: str = "train"
    subset: str | None = None
    statement_column: str = "formal_statement"
    header_column: str | None = None
    imports_column: str | None = None
    name_column: str | None = None
    proof_column: str | None = None
    """Column holding the gold proof body (used by ``validate``); None = no gold."""
    normalize_mathlib_imports: bool = False


class LeanTaskConfig(TaskConfig):
    """The knobs lean's hooks and reward read (``self.config``): the sandbox layout and
    the compile budget. Everything under ``--taskset.task.*``."""

    lean_project_path: str = LEAN_PROJECT_PATH
    proof_file_path: str = PROOF_FILE_PATH
    compile_timeout: int = 300
    """Per-compile ``timeout`` wrapper (seconds), bounding each ``lake env lean``."""


class LeanConfig(TasksetConfig):
    dataset: LeanDatasetConfig
    docker_image: str = DEFAULT_DOCKER_IMAGE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    task: LeanTaskConfig = LeanTaskConfig()


class LeanData(TaskData):
    formal_statement: str
    header: str = ""
    imports: str = "import Mathlib"
    normalize_mathlib_imports: bool = False
    # Canonical ``theorem ... := by`` text pinned at load; the reward checks it
    # still appears in the final file (the only edit the reward cares about).
    protected_signature: str = ""
    # Gold proof body (replaces ``  sorry``); "" when the dataset ships no gold.
    formal_proof: str = ""


class LeanTask(Task[LeanData, State, LeanTaskConfig]):
    NEEDS_CONTAINER = True
    # The row is `self.data` (LeanData); the sandbox layout + compile budget are
    # config knobs, read off `self.config` (LeanTaskConfig).

    async def _compile(self, runtime: Runtime) -> tuple[bool, str, int]:
        """Run ``lake env lean`` on the proof file; returns (compiled, output, exit_code)."""
        cmd = (
            f"cd {shlex.quote(self.config.lean_project_path)} && "
            f"timeout {self.config.compile_timeout} lake env lean "
            f"{shlex.quote(self.config.proof_file_path)} 2>&1; "
            "echo EXIT_CODE:$?"
        )
        result = await runtime.run(["bash", "-lc", cmd], {})
        return parse_compile_output((result.stdout or "") + (result.stderr or ""))

    async def setup(self, runtime: Runtime) -> None:
        """Plant the ``sorry`` starter file in the sandbox before the agent runs."""
        content = build_starter_file(
            self.data.formal_statement,
            header=self.data.header,
            imports=self.data.imports,
            normalize=self.data.normalize_mathlib_imports,
        )
        await runtime.write(self.config.proof_file_path, content.encode())

    @reward(weight=1.0)
    async def lean_compiled(self, trace: Trace, runtime: Runtime) -> float:
        """1.0 iff the final proof compiles cleanly (exit 0, no ``sorry``) and the
        protected theorem signature is intact; 0.0 otherwise.

        Reads the final ``proof.lean`` back through the live runtime, runs the
        host-side signature guard (with comment/string stripping), then re-runs
        ``lake env lean``. Diagnostics land on ``trace.info`` for inspection.
        """
        if trace.has_error:
            return 0.0

        # Read the final proof back. A read failure here is genuinely exceptional
        # (setup planted the file; the agent edits it in-sandbox), so let it
        # propagate as a scoring error rather than swallow an infra/sandbox failure
        # into a false-negative 0. (The prime runtime collapses every read error
        # into SandboxError, so there's no file-not-found type to narrow to.)
        current = (await runtime.read(self.config.proof_file_path)).decode(
            "utf-8", "replace"
        )

        expected_sig = self.data.protected_signature or expected_protected_signature(
            self.data.formal_statement
        )
        if expected_sig and not protected_signature_substring_present(
            current, expected_sig
        ):
            trace.info["lean_tampered"] = True
            trace.info["compile_output"] = "signature rewritten or hidden in a comment"
            return 0.0
        trace.info["lean_tampered"] = False

        compiled, output, exit_code = await self._compile(runtime)
        trace.info["lean_compiled"] = compiled
        trace.info["compile_exit_code"] = exit_code
        trace.info["compile_output"] = output[-4000:]
        return 1.0 if compiled else 0.0

    async def validate(self, runtime: Runtime) -> bool:
        """Compile the gold proof: substitute it for ``sorry`` and check it type-checks.

        ``False`` is reserved for a row whose gold proof exists but **fails to
        compile** (a genuinely bad/unprovable row). A row with **no** gold proof
        (statement-only datasets, or an empty proof column) returns ``True`` — there
        is nothing to refute, matching the base ``Task.validate`` no-op; flagging
        it ``invalid`` would both swamp the report on statement-only datasets and
        mask the rows whose gold actually fails on a sparse-gold dataset.
        """
        gold = (self.data.formal_proof or "").rstrip()
        if not gold:
            return True
        content = build_starter_file(
            self.data.formal_statement,
            header=self.data.header,
            imports=self.data.imports,
            normalize=self.data.normalize_mathlib_imports,
            proof_body=gold,
        )
        await runtime.write(self.config.proof_file_path, content.encode())
        compiled, _, _ = await self._compile(runtime)
        return compiled


class LeanTaskset(Taskset[LeanTask, LeanConfig]):
    def load(self) -> list[LeanData]:
        from datasets import load_dataset

        config = self.config
        ds = config.dataset
        raw = load_dataset(
            ds.name,
            ds.subset,
            split=ds.split,
            keep_in_memory=True,
            num_proc=8,
        )

        # Fail loud on a misconfigured column name — otherwise a typo is silently
        # treated as a missing/empty field (e.g. a wrong proof_column makes every
        # gold check vacuously fail), or a statement_column typo raises a raw
        # KeyError on the first row instead of this clear error.
        for label, col in (
            ("statement_column", ds.statement_column),
            ("header_column", ds.header_column),
            ("imports_column", ds.imports_column),
            ("name_column", ds.name_column),
            ("proof_column", ds.proof_column),
        ):
            if col is not None and col not in raw.column_names:
                raise ValueError(
                    f"dataset.{label}={col!r} not found in {ds.name!r}; columns={raw.column_names}"
                )

        resources = TaskResources(cpu=4, memory=4, disk=10)
        tasks: list[LeanData] = []
        for index, row in enumerate(raw):
            # Skip degenerate rows with no statement: there's nothing to prove, and
            # an empty statement collapses the pinned signature to just ``:= by``,
            # which every proof contains — so the reward-hacking guard would pass on
            # a rewritten trivial theorem. Drop them rather than ship a free-reward task.
            formal_statement = row[ds.statement_column]
            if not isinstance(formal_statement, str) or not formal_statement.strip():
                continue
            # Unset columns are None, and row.get(None) is None, so the `or`
            # fallbacks cover both an unset column and an empty value.
            header = row.get(ds.header_column) or ""
            imports = row.get(ds.imports_column) or "import Mathlib"
            gold = row.get(ds.proof_column) or ""
            name = row.get(ds.name_column)
            tasks.append(
                LeanData(
                    idx=index,
                    name=str(name) if name else f"task_{index:05d}",
                    prompt=self._build_prompt(formal_statement, header),
                    system_prompt=config.system_prompt,
                    image=config.docker_image,
                    workdir=config.task.lean_project_path,
                    resources=resources,
                    formal_statement=formal_statement,
                    header=header,
                    imports=imports,
                    normalize_mathlib_imports=ds.normalize_mathlib_imports,
                    protected_signature=expected_protected_signature(formal_statement),
                    formal_proof=gold,
                )
            )
        return tasks

    def _build_prompt(self, formal_statement: str, header: str) -> str:
        cfg = self.config
        block = (
            "Prove the following Lean 4 theorem. A starter proof file is at "
            f"`{cfg.task.proof_file_path}` with the theorem statement and a `sorry` "
            "placeholder already in place. Edit it and compile with "
            f"`cd {cfg.task.lean_project_path} && lake env lean {cfg.task.proof_file_path}`.\n\n"
            f"```lean\n{formal_statement}\n```"
        )
        if header:
            block += f"\n\nThe file header (imports/namespaces) is already set up:\n```lean\n{header}\n```"
        block += (
            "\n\nDo NOT modify the theorem statement (the lines from `theorem ...` "
            "through `:= by`) — the grader checks the original statement still "
            "appears and gives zero reward if you rewrote it. Write your proof "
            "tactics in place of `sorry`; the final proof must not contain `sorry` "
            "or `admit`. A clean compile prints nothing and exits 0."
        )
        return block


__all__ = ["LeanTask", "LeanConfig", "LeanTaskset"]
