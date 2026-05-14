"""Lean 4 theorem proving tasks.

Extracts the task-specific logic from ``lean_code`` (~1100 lines) into a
reusable ``LeanTaskSet`` (~120 lines).  Each dataset preset becomes its own
``TaskSet``.

Usage::

    from tasksets.lean import LeanTaskSet

    # Each preset is a TaskSet
    prover_v1 = LeanTaskSet("deepseek-prover-v1")   # 2000+ theorems
    minif2f   = LeanTaskSet("minif2f")               # 488 AMC/AIME/IMO
    goedel    = LeanTaskSet("goedel-pset")            # Mathlib formalization

    # Use with any agent
    env = ComposableEnv(task=prover_v1, agent=react_agent)

Reward-hacking guard
--------------------

The starter proof file wraps the theorem signature in marker comments::

    -- lean-guard: begin protected
    theorem foo (x : ℝ) : x = x := by
    -- lean-guard: end protected
      sorry

``LeanRubric`` re-reads the file post-rollout and refuses to award reward if
the protected region was modified, defeating the trivial "rewrite the
statement to ``True := trivial``" cheat.  The marker convention matches
``hallerite/lean-guard`` (the OpenCode plugin) so both layers agree on what
"protected" means.
"""

import re
from dataclasses import dataclass

import verifiers as vf
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet

DEFAULT_DOCKER_IMAGE = "cmkkc4gtv000mapvd5jegz3yz/lean-tactic:mathlib-v4.27.0-v3"
LEAN_PROJECT_PATH = "/workspace/mathlib4"
PROOF_FILE_PATH = "/tmp/proof.lean"

LEAN_GUARD_BEGIN_MARKER = "-- lean-guard: begin protected"
LEAN_GUARD_END_MARKER = "-- lean-guard: end protected"

LEAN_SYSTEM_PROMPT = """\
You are a Lean 4 theorem prover.

A proof file has been placed at /tmp/proof.lean containing the theorem \
statement with `sorry` as placeholder.

Your goal is to replace `sorry` with a valid proof so that the file compiles \
without errors.

WORKFLOW — you MUST follow this loop:
1. Read the proof file: `cat /tmp/proof.lean`
2. Edit the file to replace `sorry` with your proof attempt
3. COMPILE to check: `cd /workspace/mathlib4 && lake env lean /tmp/proof.lean`
4. If there are errors, read them carefully, edit the file, and compile again
5. Repeat steps 2-4 until compilation succeeds with NO errors
6. NEVER stop until the proof compiles successfully

CRITICAL: You must run the compile command after EVERY edit. \
Do NOT declare success without seeing a clean compilation output. \
If you have not compiled, you are not done.

Rules:
- No `sorry` or `admit` in the final proof
- Use Lean 4 / Mathlib syntax
- Each response must contain EXACTLY ONE tool call
- The lines wrapped by `-- lean-guard: begin protected` and \
`-- lean-guard: end protected` are the locked theorem signature. \
DO NOT modify them, the markers, or anything between them. \
Only edit the lines BELOW `-- lean-guard: end protected` (the proof body).\
"""


# ---------------------------------------------------------------------------
# Dataset presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetPreset:
    dataset_name: str
    dataset_split: str = "train"
    dataset_subset: str | None = None
    formal_statement_column: str | None = None
    header_column: str | None = None
    imports_column: str | None = None
    name_column: str | None = None
    normalize_mathlib_imports: bool = False


PRESETS: dict[str, DatasetPreset] = {
    "goedel-pset": DatasetPreset("Goedel-LM/Goedel-Pset-v1"),
    "numina-lean": DatasetPreset("AI-MO/NuminaMath-LEAN", name_column="uuid"),
    "deepseek-prover-v1": DatasetPreset(
        "deepseek-ai/DeepSeek-Prover-V1",
        header_column="header",
        name_column="name",
    ),
    "kimina": DatasetPreset("AI-MO/Kimina-Prover-Promptset", name_column="name"),
    "minif2f": DatasetPreset(
        "cat-searcher/minif2f-lean4",
        dataset_split="test",
        header_column="header",
        name_column="id",
        normalize_mathlib_imports=True,
    ),
    "deepseek-proverbench": DatasetPreset(
        "deepseek-ai/DeepSeek-ProverBench",
        header_column="header",
        name_column="name",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_imports(text: str) -> str:
    """Collapse individual Mathlib imports into a single `import Mathlib`."""
    lines = text.split("\n")
    result = []
    inserted = False
    for line in lines:
        if line.strip().startswith("import Mathlib"):
            if not inserted:
                result.append("import Mathlib")
                inserted = True
        else:
            result.append(line)
    return "\n".join(result)


def _build_preamble(
    imports_str: str,
    header: str,
    normalize: bool,
) -> str:
    if header and header.strip().startswith("import"):
        preamble = header.strip()
        if normalize:
            preamble = _normalize_imports(preamble)
        return preamble
    parts = [imports_str.strip()]
    if header and header.strip():
        parts.append(header.strip())
    preamble = "\n\n".join(parts)
    if normalize:
        preamble = _normalize_imports(preamble)
    return preamble


def _normalize_signature(stmt: str) -> str:
    """Canonicalize a Lean theorem statement to end with `:= by`.

    Strips trailing ``sorry``/``admit`` placeholders and any trailing
    ``by`` / ``:=`` tokens, then re-appends `` := by`` so the result
    is uniformly shaped regardless of the input dataset's convention
    (``:= sorry``, ``:= by sorry``, ``:= by\\n  sorry``, etc.).
    """
    s = stmt.rstrip()
    s = re.sub(r"\s*\b(?:sorry|admit)\b\s*$", "", s)
    s = re.sub(r"\s*\bby\b\s*$", "", s)
    s = re.sub(r"\s*:=\s*$", "", s)
    return s.rstrip() + " := by"


def _split_imports_and_signature(stmt: str) -> tuple[str, str]:
    """Split a self-contained statement into ``(imports_block, signature)``.

    ``formal_statement`` fields occasionally contain their own ``import``
    preamble.  We split at the first ``theorem``/``lemma``/``example``
    keyword so the signature can be wrapped independently of imports.
    """
    decl_match = re.search(
        r"^(?:theorem|lemma|example)\s",
        stmt,
        flags=re.MULTILINE,
    )
    if not decl_match:
        return "", stmt
    return stmt[: decl_match.start()].rstrip(), stmt[decl_match.start() :]


def _wrap_with_lean_guard(signature: str) -> str:
    """Wrap a normalized ``... := by`` signature with lean-guard markers."""
    return f"{LEAN_GUARD_BEGIN_MARKER}\n{signature}\n{LEAN_GUARD_END_MARKER}\n  sorry\n"


def _extract_protected_region(content: str) -> str | None:
    """Return the text from the begin marker through the end-of-end-line.

    Returns ``None`` if either marker is missing or out of order.
    Mirrors the substring extracted by ``hallerite/lean-guard``'s plugin
    so the two implementations agree on what "protected" means.
    """
    begin = content.find(LEAN_GUARD_BEGIN_MARKER)
    if begin == -1:
        return None
    end = content.find(LEAN_GUARD_END_MARKER, begin)
    if end == -1:
        return None
    end_of_end_line = content.find("\n", end)
    if end_of_end_line == -1:
        protected_end = len(content)
    else:
        protected_end = end_of_end_line + 1
    return content[begin:protected_end]


def _build_starter_file(info: dict) -> str:
    """Build the proof file: imports + header + protected signature + sorry."""
    stmt = info.get("formal_statement", "")

    if stmt.strip().startswith("import "):
        imports_block, signature_raw = _split_imports_and_signature(stmt)
        preamble = imports_block
    else:
        imports_str = info.get("imports", "import Mathlib")
        header = info.get("header", "")
        normalize = info.get("_normalize_mathlib_imports", False)
        preamble = _build_preamble(imports_str, header, normalize)
        signature_raw = stmt

    signature = _normalize_signature(signature_raw)
    wrapped = _wrap_with_lean_guard(signature)

    if preamble:
        return preamble.rstrip() + "\n\n" + wrapped
    return wrapped


def _expected_protected_region(info: dict) -> str:
    """Compute the protected region for a task instance from its ``info`` dict.

    Re-runs ``_build_starter_file`` so the rubric and the setup step share
    the exact same wrapping logic — no separate state plumbing needed.
    """
    starter = _build_starter_file(info)
    region = _extract_protected_region(starter)
    return region or ""


# ---------------------------------------------------------------------------
# LeanTaskSet
# ---------------------------------------------------------------------------


class LeanRubric(vf.Rubric):
    """Scores Lean tasks by compiling the proof in the sandbox.

    Before compiling, verifies the lean-guard protected region in
    ``/tmp/proof.lean`` matches the original starter; tampering yields
    ``reward=0`` and sets ``state["lean_tampered"]=True``.
    """

    def __init__(self, taskset: "LeanTaskSet", **kwargs):
        super().__init__(**kwargs)
        self.taskset = taskset
        self.add_reward_func(self.solved)

    async def solved(self, state, **kwargs) -> float:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            return 0.0
        timeout = state.get("test_timeout", 900)

        info = state.get("info") or {}
        expected_region = _expected_protected_region(info)
        if expected_region:
            try:
                cat_result = await sandbox_client.execute_command(
                    sandbox_id,
                    f"cat {self.taskset.proof_file_path}",
                    timeout=10,
                )
                current_content = cat_result.stdout or ""
            except Exception:
                state["lean_tampered"] = True
                state["lean_compiled"] = False
                return 0.0

            actual_region = _extract_protected_region(current_content)
            if actual_region != expected_region:
                state["lean_tampered"] = True
                state["lean_compiled"] = False
                return 0.0
            state["lean_tampered"] = False

        try:
            cmd = f"cd {self.taskset.lean_project_path} && lake env lean {self.taskset.proof_file_path} 2>&1; echo EXIT_CODE:$?"
            result = await sandbox_client.execute_command(
                sandbox_id,
                cmd,
                timeout=min(timeout, self.taskset.compile_timeout),
            )
            output = (result.stdout or "") + (result.stderr or "")

            # Parse exit code
            exit_code = 1
            match = re.search(r"EXIT_CODE:(\d+)", output)
            if match:
                exit_code = int(match.group(1))
                output = output[: match.start()].strip()

            has_sorry = bool(re.search(r"declaration uses 'sorry'", output))
            compiled = exit_code == 0 and not has_sorry

            state["lean_compiled"] = compiled
            state["compile_output"] = output
        except Exception:
            state["lean_compiled"] = False
            return 0.0
        return 1.0 if compiled else 0.0

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            try:
                await sandbox_client.delete(sandbox_id)
            except Exception:
                pass


class LeanTaskSet(SandboxTaskSet):
    default_workdir = "/workspace/mathlib4"
    """Lean 4 theorem proving task.

    Provides: docker image with Lean 4 + Mathlib, sandbox setup (writes
    starter proof file), and compilation-based evaluation.
    """

    def __init__(
        self,
        preset: str = "deepseek-prover-v1",
        dataset_name: str | None = None,
        dataset_split: str | None = None,
        filter_fn: str | None = None,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        lean_project_path: str = LEAN_PROJECT_PATH,
        proof_file_path: str = PROOF_FILE_PATH,
        compile_timeout: int = 120,
        ds_num_proc: int | None = 8,
        ds_keep_in_memory: bool = True,
    ):
        self.ds_num_proc = ds_num_proc
        self.ds_keep_in_memory = ds_keep_in_memory
        self.docker_image = docker_image
        self.lean_project_path = lean_project_path
        self.proof_file_path = proof_file_path
        self.compile_timeout = compile_timeout
        dataset = self._build_dataset(preset, dataset_name, dataset_split)
        super().__init__(dataset=dataset, name=f"lean/{preset}", filter_fn=filter_fn)

    def _build_dataset(
        self, preset: str, dataset_name: str | None, dataset_split: str | None
    ):
        from datasets import load_dataset

        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Available: {list(PRESETS.keys())}"
            )
        p = PRESETS[preset]
        ds_name = dataset_name or p.dataset_name
        ds_split = dataset_split or p.dataset_split
        raw = load_dataset(
            ds_name,
            p.dataset_subset,
            split=ds_split,
            keep_in_memory=self.ds_keep_in_memory,
            num_proc=self.ds_num_proc,
        )

        stmt_col = p.formal_statement_column
        if stmt_col is None:
            for candidate in ["formal_statement", "statement", "theorem"]:
                if candidate in raw.column_names:
                    stmt_col = candidate
                    break
            if stmt_col is None:
                raise ValueError(
                    f"Cannot find formal_statement column in {raw.column_names}"
                )

        def process_row(row):
            info = {
                "formal_statement": row[stmt_col],
                "header": row.get(p.header_column or "__none__", ""),
                "imports": row.get(p.imports_column or "__none__", "import Mathlib"),
                "_normalize_mathlib_imports": p.normalize_mathlib_imports,
            }
            if p.name_column and p.name_column in row:
                info["name"] = row[p.name_column]
            return {"question": row[stmt_col], "info": info, "answer": ""}

        return raw.map(
            process_row,
            remove_columns=raw.column_names,
            num_proc=self.ds_num_proc,
            keep_in_memory=self.ds_keep_in_memory,
            load_from_cache_file=False,
        )

    def get_instruction(self, info: dict) -> str:
        stmt = info.get("formal_statement", "")
        header = info.get("header", "")
        user_content = f"Prove the following Lean 4 theorem. The proof file is at `{self.proof_file_path}`.\n\n```lean\n{stmt}\n```"
        if header:
            user_content += f"\n\nThe file header (imports/namespaces) is already set up:\n```lean\n{header}\n```"
        user_content += (
            "\n\nCompile with your shell tool using:\n"
            f"`cd {self.lean_project_path} && lake env lean {self.proof_file_path}`"
        )
        return user_content

    def get_sandbox_spec(self, info: dict) -> SandboxSpec | None:
        return SandboxSpec(image=self.docker_image)

    def get_workdir(self, info: dict) -> str:
        return self.lean_project_path

    def get_env_vars(self) -> dict[str, str]:
        return {}

    async def setup(self, state) -> None:
        """Write the starter proof file to the sandbox."""
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        info = state.get("info") or {}
        content = _build_starter_file(info)

        # Base64 encode for safe transmission
        import base64

        encoded = base64.b64encode(content.encode()).decode()
        await sandbox_client.execute_command(
            sandbox_id,
            f'echo "{encoded}" | base64 -d > {self.proof_file_path}',
            timeout=10,
        )

    def get_rubric(self):
        return LeanRubric(self)

    async def validate_instance(self, state) -> bool:
        # Lean datasets don't include ground-truth proofs, so we can't validate.
        return True


# ---------------------------------------------------------------------------
# LeanTaskSet factory
# ---------------------------------------------------------------------------
