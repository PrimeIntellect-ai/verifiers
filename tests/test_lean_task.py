"""Tests for ``LeanTaskSet`` lean-guard wrapping and reward enforcement."""

from dataclasses import dataclass

import pytest

from verifiers.envs.experimental.composable.tasksets.lean.lean_task import (
    LEAN_GUARD_BEGIN_MARKER,
    LEAN_GUARD_END_MARKER,
    LeanRubric,
    _build_starter_file,
    _expected_protected_region,
    _extract_protected_region,
    _normalize_signature,
    _wrap_with_lean_guard,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestNormalizeSignature:
    """``_normalize_signature`` must converge every starter shape on ``:= by``."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (
                "theorem foo (x : ℝ) : x = x := sorry",
                "theorem foo (x : ℝ) : x = x := by",
            ),
            (
                "theorem foo (x : ℝ) : x = x := by sorry",
                "theorem foo (x : ℝ) : x = x := by",
            ),
            (
                "theorem foo (x : ℝ) : x = x := by\n  sorry",
                "theorem foo (x : ℝ) : x = x := by",
            ),
            (
                "theorem foo (x : ℝ) : x = x",
                "theorem foo (x : ℝ) : x = x := by",
            ),
            (
                "theorem foo (x : ℝ) : x = x := by",
                "theorem foo (x : ℝ) : x = x := by",
            ),
            (
                "theorem foo (x : ℝ) : x = x :=",
                "theorem foo (x : ℝ) : x = x := by",
            ),
            (
                "theorem foo (x : ℝ) : x = x := admit",
                "theorem foo (x : ℝ) : x = x := by",
            ),
        ],
    )
    def test_canonicalizes_to_by(self, raw: str, expected: str) -> None:
        assert _normalize_signature(raw) == expected

    def test_preserves_multiline_signatures(self) -> None:
        raw = (
            "theorem mathd_algebra_478\n"
            "  (b h v : ℝ)\n"
            "  (h : 0 < b) :\n"
            "  v = 65 := sorry"
        )
        expected = (
            "theorem mathd_algebra_478\n  (b h v : ℝ)\n  (h : 0 < b) :\n  v = 65 := by"
        )
        assert _normalize_signature(raw) == expected

    def test_does_not_strip_internal_by(self) -> None:
        # `by` inside a type ascription should stay; only trailing tokens are stripped.
        raw = "theorem foo : ((by trivial : True) ∧ True) := sorry"
        assert _normalize_signature(raw) == (
            "theorem foo : ((by trivial : True) ∧ True) := by"
        )


class TestWrapWithLeanGuard:
    def test_marker_layout(self) -> None:
        signature = "theorem foo (x : ℝ) : x = x := by"
        wrapped = _wrap_with_lean_guard(signature)
        assert wrapped == (
            "-- lean-guard: begin protected\n"
            "theorem foo (x : ℝ) : x = x := by\n"
            "-- lean-guard: end protected\n"
            "  sorry\n"
        )

    def test_round_trip_via_extract(self) -> None:
        signature = "theorem foo : True := by"
        wrapped = _wrap_with_lean_guard(signature)
        region = _extract_protected_region(wrapped)
        assert region is not None
        assert LEAN_GUARD_BEGIN_MARKER in region
        assert LEAN_GUARD_END_MARKER in region
        assert "  sorry" not in region


class TestExtractProtectedRegion:
    def test_returns_none_when_markers_missing(self) -> None:
        assert _extract_protected_region("theorem foo := sorry") is None

    def test_returns_none_when_only_begin(self) -> None:
        content = LEAN_GUARD_BEGIN_MARKER + "\nfoo\n  sorry\n"
        assert _extract_protected_region(content) is None

    def test_returns_none_when_only_end(self) -> None:
        content = "foo\n" + LEAN_GUARD_END_MARKER + "\n  sorry\n"
        assert _extract_protected_region(content) is None

    def test_extracts_inclusive_of_both_markers(self) -> None:
        content = (
            "import Mathlib\n\n"
            "-- lean-guard: begin protected\n"
            "theorem foo : True := by\n"
            "-- lean-guard: end protected\n"
            "  trivial\n"
        )
        region = _extract_protected_region(content)
        assert region == (
            "-- lean-guard: begin protected\n"
            "theorem foo : True := by\n"
            "-- lean-guard: end protected\n"
        )

    def test_handles_extra_text_after_end_marker_on_line(self) -> None:
        # End marker followed by trailing text on the same line still
        # delimits at the next newline.
        content = (
            "-- lean-guard: begin protected\n"
            "theorem foo : True := by\n"
            "-- lean-guard: end protected (do not edit)\n"
            "  trivial\n"
        )
        region = _extract_protected_region(content)
        assert region is not None
        assert region.endswith("(do not edit)\n")


# ---------------------------------------------------------------------------
# _build_starter_file shape coverage
# ---------------------------------------------------------------------------


@pytest.fixture
def starter_for():
    """Helper that runs ``_build_starter_file`` on a minimal info dict."""

    def _build(stmt: str, header: str = "import Mathlib", **extra) -> str:
        info = {"formal_statement": stmt, "header": header, **extra}
        return _build_starter_file(info)

    return _build


class TestBuildStarterFile:
    """Every supported starter shape must produce a wrapped file."""

    @pytest.mark.parametrize(
        "stmt",
        [
            "theorem foo (x : ℝ) : x = x := sorry",
            "theorem foo (x : ℝ) : x = x := by sorry",
            "theorem foo (x : ℝ) : x = x := by\n  sorry",
            "theorem foo (x : ℝ) : x = x",
            "theorem foo (x : ℝ) : x = x := by",
            "theorem foo (x : ℝ) : x = x :=",
        ],
    )
    def test_all_shapes_get_wrapped(self, starter_for, stmt: str) -> None:
        out = starter_for(stmt)
        assert LEAN_GUARD_BEGIN_MARKER in out
        assert LEAN_GUARD_END_MARKER in out
        # Body always defaults to `  sorry` after the end marker.
        assert out.endswith("  sorry\n")
        # The protected region must contain the canonical signature.
        region = _extract_protected_region(out)
        assert region is not None
        assert "theorem foo (x : ℝ) : x = x := by" in region

    def test_self_contained_imports_are_separated(self, starter_for) -> None:
        stmt = "import Mathlib.Tactic\n\nopen Real\n\ntheorem foo : 1 + 1 = 2 := sorry"
        out = starter_for(stmt, header="")
        # Imports + open stay outside the protected region.
        assert out.startswith("import Mathlib.Tactic\n\nopen Real")
        # The theorem signature is wrapped.
        region = _extract_protected_region(out)
        assert region is not None
        assert "theorem foo : 1 + 1 = 2 := by" in region

    def test_multiline_signature_preserved(self, starter_for) -> None:
        stmt = (
            "theorem mathd_algebra_478\n"
            "  (b h v : ℝ)\n"
            "  (h₀ : 0 < b) :\n"
            "  v = 65 := sorry"
        )
        out = starter_for(stmt, header="import Mathlib")
        region = _extract_protected_region(out) or ""
        assert "theorem mathd_algebra_478" in region
        assert "(h₀ : 0 < b)" in region
        assert "v = 65 := by" in region

    def test_expected_protected_region_round_trips(self, starter_for) -> None:
        info = {
            "formal_statement": "theorem foo : True := sorry",
            "header": "import Mathlib",
        }
        starter = _build_starter_file(info)
        expected = _expected_protected_region(info)
        actual = _extract_protected_region(starter)
        assert expected == actual
        assert expected != ""


# ---------------------------------------------------------------------------
# LeanRubric tamper enforcement
# ---------------------------------------------------------------------------


@dataclass
class _CmdResult:
    stdout: str = ""
    stderr: str = ""


class _FakeSandboxClient:
    """Minimal async sandbox stub that scripts cat / lake responses."""

    def __init__(self, file_contents: str, compile_output: str = "EXIT_CODE:0\n"):
        self._file_contents = file_contents
        self._compile_output = compile_output
        self.calls: list[str] = []

    async def execute_command(self, sandbox_id, cmd, timeout=None):
        self.calls.append(cmd)
        if cmd.startswith("cat "):
            return _CmdResult(stdout=self._file_contents)
        if "lake env lean" in cmd:
            return _CmdResult(stdout=self._compile_output)
        raise AssertionError(f"unexpected command: {cmd!r}")

    async def delete(self, _sandbox_id):
        return None


class _StubTaskSet:
    """Just enough of ``LeanTaskSet`` for ``LeanRubric.solved``."""

    proof_file_path = "/tmp/proof.lean"
    lean_project_path = "/workspace/mathlib4"
    compile_timeout = 120


def _make_rubric() -> LeanRubric:
    return LeanRubric(_StubTaskSet())  # type: ignore[arg-type]


def _make_state(info: dict, file_contents: str, compile_output: str = "EXIT_CODE:0\n"):
    client = _FakeSandboxClient(file_contents, compile_output)
    state = {
        "sandbox_client": client,
        "sandbox_id": "sb-test",
        "info": info,
    }
    return state, client


@pytest.mark.asyncio
class TestLeanRubricTamper:
    """The rubric must zero out reward when the protected region is altered."""

    INFO = {
        "formal_statement": "theorem foo (x : ℝ) : x = x := sorry",
        "header": "import Mathlib",
    }

    def _legitimate_proof(self) -> str:
        starter = _build_starter_file(self.INFO)
        return starter.replace("  sorry\n", "  rfl\n")

    async def test_clean_proof_compiles_to_one(self) -> None:
        rubric = _make_rubric()
        state, _ = _make_state(self.INFO, self._legitimate_proof())
        reward = await rubric.solved(state)
        assert reward == 1.0
        assert state["lean_compiled"] is True
        assert state.get("lean_tampered") is False

    async def test_signature_replaced_zeroes_reward(self) -> None:
        rubric = _make_rubric()
        # Cheat: replace the entire theorem with a trivially-true one.
        cheat = "import Mathlib\n\ntheorem foo : True := trivial\n"
        state, client = _make_state(self.INFO, cheat)
        reward = await rubric.solved(state)
        assert reward == 0.0
        assert state["lean_tampered"] is True
        assert state["lean_compiled"] is False
        # Compile should not have been attempted once tampering is detected.
        assert all("lake env lean" not in c for c in client.calls)

    async def test_markers_removed_zeroes_reward(self) -> None:
        rubric = _make_rubric()
        # Markers stripped, but the signature itself was left intact.
        starter = _build_starter_file(self.INFO)
        no_markers = starter.replace(LEAN_GUARD_BEGIN_MARKER + "\n", "").replace(
            LEAN_GUARD_END_MARKER + "\n", ""
        )
        state, _ = _make_state(self.INFO, no_markers)
        reward = await rubric.solved(state)
        assert reward == 0.0
        assert state["lean_tampered"] is True

    async def test_signature_subtly_altered_zeroes_reward(self) -> None:
        rubric = _make_rubric()
        # Swap the goal `x = x` for the trivially-true `True`.
        starter = _build_starter_file(self.INFO)
        tampered = starter.replace("x = x", "True")
        state, _ = _make_state(self.INFO, tampered)
        reward = await rubric.solved(state)
        assert reward == 0.0
        assert state["lean_tampered"] is True

    async def test_compile_failure_after_clean_signature_returns_zero(self) -> None:
        rubric = _make_rubric()
        starter = _build_starter_file(self.INFO)
        proof = starter.replace("  sorry\n", "  exact rfl\n")
        state, _ = _make_state(
            self.INFO,
            proof,
            compile_output="error: unsolved goals\nEXIT_CODE:1\n",
        )
        reward = await rubric.solved(state)
        assert reward == 0.0
        assert state["lean_compiled"] is False
        # Tampering check passed first: the flag is cleared.
        assert state["lean_tampered"] is False
