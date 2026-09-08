"""Tests for the Lean taskset's scoring helpers and axiom audit."""

from types import SimpleNamespace

from verifiers.v1.tasksets.lean.scoring import (
    TRUSTED_AXIOMS,
    declaration_name,
    parse_axioms_output,
    untrusted_axioms,
)

CHEAT_OUTPUT = (
    "'fake_irrational' depends on axioms: "
    "[cheat, propext, Classical.choice, Quot.sound]"
)
CLEAN_OUTPUT = (
    "'real_theorem' depends on axioms: [propext, Classical.choice, Quot.sound]"
)


def test_declaration_name_reads_theorem_and_lemma() -> None:
    assert (
        declaration_name("theorem fake_irrational : Irrational (1 : ℝ) := by")
        == "fake_irrational"
    )
    assert declaration_name("lemma my_lemma (x : ℕ) : x = x := by") == "my_lemma"
    assert declaration_name("import Mathlib\ntheorem foo : True := by") == "foo"


def test_declaration_name_returns_empty_when_unparsable() -> None:
    assert declaration_name("example : True := by") == ""
    assert declaration_name("") == ""


def test_parse_axioms_output_reads_the_axiom_list() -> None:
    assert parse_axioms_output(CHEAT_OUTPUT) == [
        "cheat",
        "propext",
        "Classical.choice",
        "Quot.sound",
    ]


def test_parse_axioms_output_handles_no_axioms() -> None:
    assert parse_axioms_output("'t' does not depend on any axioms") == []
    assert parse_axioms_output("") == []


def test_untrusted_axioms_flags_an_injected_axiom() -> None:
    """Regression test for the ``axiom cheat : False`` reward bypass."""
    assert untrusted_axioms(CHEAT_OUTPUT) == ["cheat"]


def test_untrusted_axioms_accepts_the_mathlib_baseline() -> None:
    assert untrusted_axioms(CLEAN_OUTPUT) == []
    assert set(TRUSTED_AXIOMS) == {"propext", "Classical.choice", "Quot.sound"}


def test_untrusted_axioms_matches_the_last_report() -> None:
    """A printed decoy must not shadow the real audit result.

    Our ``#print axioms`` query is appended last, so anything earlier in the
    transcript came from the rollout -- the same reasoning ``parse_compile_output``
    uses for ``EXIT_CODE`` markers.
    """
    decoy = (
        "'fake_irrational' depends on axioms: "
        "[propext, Classical.choice, Quot.sound]\n" + CHEAT_OUTPUT
    )
    assert untrusted_axioms(decoy) == ["cheat"]


class _StubResult:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""


class _StubRuntime:
    """Serves the proof file and canned ``lake env lean`` transcripts."""

    def __init__(self, axioms_output: str) -> None:
        self.files = {
            "/tmp/proof.lean": (
                b"import Mathlib\naxiom cheat : False\n"
                b"theorem foo : True := by\n  exact cheat.elim\n"
            )
        }
        self.axioms_output = axioms_output

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        return self.files[path]

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def run(self, argv: list[str], env: dict[str, str]) -> _StubResult:
        command = argv[-1]
        if ".axioms.lean" in command:
            return _StubResult(self.axioms_output)
        return _StubResult("EXIT_CODE:0")


def _lean_task():
    from verifiers.v1.tasksets.lean.taskset import LeanData, LeanTask, LeanTaskConfig

    return LeanTask(
        LeanData(
            idx=0,
            prompt="prove it",
            formal_statement="theorem foo : True := by",
            protected_signature="theorem foo : True := by",
        ),
        LeanTaskConfig(),
    )


async def test_lean_compiled_rejects_a_proof_resting_on_an_injected_axiom() -> None:
    """A clean compile must not score when the rollout supplied the axiom."""
    trace = SimpleNamespace(has_error=False, info={})
    runtime = _StubRuntime(CHEAT_OUTPUT.replace("fake_irrational", "foo"))

    assert await _lean_task().lean_compiled(trace, runtime) == 0.0
    assert trace.info["lean_compiled"] is True
    assert trace.info["lean_tampered"] is True
    assert trace.info["lean_untrusted_axioms"] == ["cheat"]


async def test_lean_compiled_still_rewards_a_genuine_proof() -> None:
    trace = SimpleNamespace(has_error=False, info={})
    runtime = _StubRuntime(CLEAN_OUTPUT.replace("real_theorem", "foo"))

    assert await _lean_task().lean_compiled(trace, runtime) == 1.0
    assert trace.info["lean_tampered"] is False
