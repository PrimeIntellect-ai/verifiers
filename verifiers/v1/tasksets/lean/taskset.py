"""Lean 4 theorem-proving tasks backed by Hugging Face datasets.

Each task plants a ``sorry``-based starter file, lets any container-capable
harness edit it, and rewards a clean ``lake env lean`` compile. The original
theorem signature must remain present, which prevents replacing the assigned
statement with an easier theorem. Dataset-specific packages can subclass this
taskset and only supply column mappings.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator

from pydantic_config import BaseConfig

from verifiers.v1.decorators import reward
from verifiers.v1.runtimes import Runtime
from verifiers.v1.state import State
from verifiers.v1.task import Task, TaskConfig, TaskData, TaskResources
from verifiers.v1.taskset import Taskset, TasksetConfig, resolve_system_prompt
from verifiers.v1.tasksets.lean.scoring import (
    build_starter_file,
    expected_protected_signature,
    parse_compile_output,
    protected_signature_substring_present,
)
from verifiers.v1.trace import Trace

# Lean v4.27 with Mathlib v4.27.
DEFAULT_DOCKER_IMAGE = "team-clyvldofb0000gg1kx39rgzjq/lean-tactic:mathlib-v4.27.0-v3"
LEAN_PROJECT_PATH = "/workspace/mathlib4"
PROOF_FILE_PATH = "/tmp/proof.lean"

DEFAULT_SYSTEM_PROMPT = "You are an expert Lean 4 theorem prover working with Mathlib."


class LeanDatasetConfig(BaseConfig):
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
    lean_project_path: str = LEAN_PROJECT_PATH
    proof_file_path: str = PROOF_FILE_PATH
    compile_timeout: int = 300
    """Per-compile ``timeout`` wrapper (seconds), bounding each ``lake env lean``."""


class LeanConfig(TasksetConfig):
    dataset: LeanDatasetConfig
    docker_image: str = DEFAULT_DOCKER_IMAGE
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

    async def _compile(self, runtime: Runtime) -> tuple[bool, str, int]:
        cmd = (
            f"cd {shlex.quote(self.config.lean_project_path)} && "
            f"timeout {self.config.compile_timeout} lake env lean "
            f"{shlex.quote(self.config.proof_file_path)} 2>&1; "
            "echo EXIT_CODE:$?"
        )
        result = await runtime.run(["bash", "-lc", cmd], {})
        return parse_compile_output((result.stdout or "") + (result.stderr or ""))

    async def setup(self, runtime: Runtime) -> None:
        content = build_starter_file(
            self.data.formal_statement,
            header=self.data.header,
            imports=self.data.imports,
            normalize=self.data.normalize_mathlib_imports,
        )
        await runtime.write(self.config.proof_file_path, content.encode())

    @reward(weight=1.0)
    async def lean_compiled(self, trace: Trace, runtime: Runtime) -> float:
        """Require both the assigned signature and a clean Lean compile."""
        if trace.has_error:
            return 0.0

        # Setup created this file, so a read failure is an infrastructure error.
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
        """Compile the gold proof; rows without one have nothing to preflight."""
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
    def load(self) -> Iterator[LeanTask]:
        from datasets import load_dataset

        config = self.config
        ds = config.dataset
        raw = load_dataset(
            ds.name,
            ds.subset,
            split=ds.split,
            num_proc=8,
        )

        # Validate optional columns before empty-value fallbacks hide a typo.
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
        system_prompt = resolve_system_prompt(config) or DEFAULT_SYSTEM_PROMPT
        for index, row in enumerate(raw):
            # An empty statement would reduce the signature guard to `:= by`.
            formal_statement = row[ds.statement_column]
            if not isinstance(formal_statement, str) or not formal_statement.strip():
                continue
            # A disabled optional column is None; row.get(None) yields the same
            # empty fallback as a present-but-empty column.
            header = row.get(ds.header_column) or ""
            imports = row.get(ds.imports_column) or "import Mathlib"
            gold = row.get(ds.proof_column) or ""
            name = row.get(ds.name_column)
            yield LeanTask(
                LeanData(
                    idx=index,
                    name=str(name) if name else f"task_{index:05d}",
                    prompt=self._build_prompt(formal_statement, header),
                    system_prompt=system_prompt,
                    image=config.docker_image,
                    workdir=config.task.lean_project_path,
                    resources=resources,
                    formal_statement=formal_statement,
                    header=header,
                    imports=imports,
                    normalize_mathlib_imports=ds.normalize_mathlib_imports,
                    protected_signature=expected_protected_signature(formal_statement),
                    formal_proof=gold,
                ),
                self.config.task,
            )

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
