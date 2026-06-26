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

from verifiers.v1.decorators import reward
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset, TasksetConfig
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


class LeanTask(Task):
    formal_statement: str
    header: str = ""
    imports: str = "import Mathlib"
    normalize_mathlib_imports: bool = False
    # Canonical ``theorem ... := by`` text pinned at load; the reward checks it
    # still appears in the final file (the only edit the reward cares about).
    protected_signature: str = ""
    # Gold proof body (replaces ``  sorry``); "" when the dataset ships no gold.
    formal_proof: str = ""


class LeanConfig(TasksetConfig):
    # ── dataset selection ────────────────────────────────────────────────────
    dataset_name: str = ""
    """HuggingFace dataset id. Required — a per-dataset package sets this default."""
    dataset_split: str = "train"
    dataset_subset: str | None = None
    statement_column: str | None = None
    """Column holding the formal statement; None auto-detects
    ``formal_statement`` / ``statement`` / ``theorem``."""
    header_column: str | None = None
    imports_column: str | None = None
    name_column: str | None = None
    proof_column: str | None = None
    """Column holding the gold proof body (used by ``validate``); None = no gold."""
    normalize_mathlib_imports: bool = False
    # ── sandbox / compile ─────────────────────────────────────────────────────
    docker_image: str = DEFAULT_DOCKER_IMAGE
    lean_project_path: str = LEAN_PROJECT_PATH
    proof_file_path: str = PROOF_FILE_PATH
    compile_timeout: int = 300
    """Per-compile ``timeout`` wrapper (seconds). Cold-start Mathlib loading on a
    fresh sandbox eats most of a 120s budget, so 300 leaves headroom."""
    setup_timeout: int = 300
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    sandbox_cpu_cores: float = 4
    sandbox_memory_gb: float = 4
    sandbox_disk_size_gb: float = 10
    # ── dataset loading ───────────────────────────────────────────────────────
    ds_num_proc: int | None = 8
    ds_keep_in_memory: bool = True
    max_examples: int = -1


class LeanTaskset(Taskset[LeanTask, LeanConfig]):
    NEEDS_CONTAINER = True

    # ---- task loading -------------------------------------------------------

    def load_tasks(self) -> list[LeanTask]:
        from datasets import load_dataset

        config = self.config
        if not config.dataset_name:
            raise ValueError(
                "LeanConfig.dataset_name is empty. Set --taskset.dataset-name, or use a "
                "per-dataset package (e.g. lean-deepseek-prover-v1) that sets it."
            )
        raw = load_dataset(
            config.dataset_name,
            config.dataset_subset,
            split=config.dataset_split,
            keep_in_memory=config.ds_keep_in_memory,
            num_proc=config.ds_num_proc,
        )

        stmt_col = config.statement_column or next(
            (
                c
                for c in ("formal_statement", "statement", "theorem")
                if c in raw.column_names
            ),
            None,
        )
        if stmt_col is None:
            raise ValueError(
                f"No formal-statement column in {config.dataset_name!r}; set --taskset.statement-column "
                f"(columns={raw.column_names})."
            )
        # Fail loud on a misconfigured column name — otherwise a typo is silently
        # treated as a missing/empty field (e.g. a wrong proof_column makes every
        # gold check vacuously fail), or an explicit statement_column typo raises a
        # raw KeyError on the first row instead of this clear error.
        for label, col in (
            ("statement_column", config.statement_column),
            ("header_column", config.header_column),
            ("imports_column", config.imports_column),
            ("name_column", config.name_column),
            ("proof_column", config.proof_column),
        ):
            if col is not None and col not in raw.column_names:
                raise ValueError(
                    f"{label}={col!r} not found in {config.dataset_name!r}; columns={raw.column_names}"
                )

        resources = TaskResources(
            cpu=config.sandbox_cpu_cores,
            memory=config.sandbox_memory_gb,
            disk=config.sandbox_disk_size_gb,
        )
        # Cap the scoring stage so a slow/hung compile can't wedge the run; the
        # reward wraps ``lake`` in ``timeout`` too — this is the outer backstop.
        timeout = TaskTimeout(
            setup=config.setup_timeout, scoring=config.compile_timeout + 120
        )

        limit = config.max_examples if config.max_examples >= 0 else len(raw)
        tasks: list[LeanTask] = []
        for index, row in enumerate(raw):
            if index >= limit:
                break
            formal_statement = row[stmt_col]
            header = (
                row.get(config.header_column) if config.header_column else ""
            ) or ""
            imports = (
                row.get(config.imports_column) if config.imports_column else ""
            ) or "import Mathlib"
            gold = (row.get(config.proof_column) if config.proof_column else "") or ""
            name = row.get(config.name_column) if config.name_column else None
            tasks.append(
                LeanTask(
                    idx=index,
                    name=str(name) if name else f"task_{index:05d}",
                    prompt=self._build_prompt(formal_statement, header),
                    system_prompt=config.system_prompt,
                    image=config.docker_image,
                    workdir=config.lean_project_path,
                    resources=resources,
                    timeout=timeout,
                    formal_statement=formal_statement,
                    header=header,
                    imports=imports,
                    normalize_mathlib_imports=config.normalize_mathlib_imports,
                    protected_signature=expected_protected_signature(formal_statement),
                    formal_proof=gold,
                )
            )
        return tasks

    def _build_prompt(self, formal_statement: str, header: str) -> str:
        cfg = self.config
        block = (
            "Prove the following Lean 4 theorem. A starter proof file is at "
            f"`{cfg.proof_file_path}` with the theorem statement and a `sorry` "
            "placeholder already in place. Edit it and compile with "
            f"`cd {cfg.lean_project_path} && lake env lean {cfg.proof_file_path}`.\n\n"
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

    # ---- sandbox helpers ----------------------------------------------------

    async def _write_file(self, runtime: Runtime, path: str, content: str) -> None:
        """Write ``content`` to ``path`` in the sandbox.

        Uses ``runtime.write`` (multipart upload that creates parent dirs) rather
        than inlining the bytes on the command line — arbitrary Lean source or a
        large gold proof would otherwise risk the shell argument-length limit.
        """
        await runtime.write(path, content.encode())

    async def _compile(self, runtime: Runtime) -> tuple[bool, str, int]:
        """Run ``lake env lean`` on the proof file; returns (compiled, output, exit_code)."""
        cfg = self.config
        cmd = (
            f"cd {shlex.quote(cfg.lean_project_path)} && "
            f"timeout {cfg.compile_timeout} lake env lean {shlex.quote(cfg.proof_file_path)} 2>&1; "
            "echo EXIT_CODE:$?"
        )
        result = await runtime.run(["bash", "-lc", cmd], {})
        return parse_compile_output((result.stdout or "") + (result.stderr or ""))

    # ---- lifecycle ----------------------------------------------------------

    async def setup(self, task: LeanTask, runtime: Runtime) -> None:
        """Plant the ``sorry`` starter file in the sandbox before the agent runs."""
        content = build_starter_file(
            task.formal_statement,
            header=task.header,
            imports=task.imports,
            normalize=task.normalize_mathlib_imports,
        )
        await self._write_file(runtime, self.config.proof_file_path, content)

    # ---- scoring ------------------------------------------------------------

    @reward(weight=1.0)
    async def lean_compiled(
        self, task: LeanTask, trace: Trace, runtime: Runtime
    ) -> float:
        """1.0 iff the final proof compiles cleanly (exit 0, no ``sorry``) and the
        protected theorem signature is intact; 0.0 otherwise.

        Reads the final ``proof.lean`` back through the live runtime, runs the
        host-side signature guard (with comment/string stripping), then re-runs
        ``lake env lean``. Diagnostics land on ``trace.info`` for inspection.
        """
        if trace.has_error:
            return 0.0

        try:
            current = (await runtime.read(self.config.proof_file_path)).decode(
                "utf-8", "replace"
            )
        except Exception:
            trace.info["compile_output"] = "proof file unreadable (missing?)"
            return 0.0

        expected_sig = task.protected_signature or expected_protected_signature(
            task.formal_statement
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

    # ---- validation (model-free gold check) ---------------------------------

    async def validate(self, task: LeanTask, runtime: Runtime) -> bool:
        """Compile the gold proof: substitute it for ``sorry`` and check it type-checks.

        ``False`` is reserved for a row whose gold proof exists but **fails to
        compile** (a genuinely bad/unprovable row). A row with **no** gold proof
        (statement-only datasets, or an empty proof column) returns ``True`` — there
        is nothing to refute, matching the base ``Taskset.validate`` no-op; flagging
        it ``invalid`` would both swamp the report on statement-only datasets and
        mask the rows whose gold actually fails on a sparse-gold dataset.
        """
        gold = (task.formal_proof or "").rstrip()
        if not gold:
            return True
        content = build_starter_file(
            task.formal_statement,
            header=task.header,
            imports=task.imports,
            normalize=task.normalize_mathlib_imports,
            proof_body=gold,
        )
        await self._write_file(runtime, self.config.proof_file_path, content)
        compiled, _, _ = await self._compile(runtime)
        return compiled


__all__ = ["LeanTask", "LeanConfig", "LeanTaskset"]
