"""Unit tests for the v1 lean taskset's pure scoring helpers (signature guard,
starter file, compile-output parsing) in `verifiers.v1.tasksets.lean.scoring`."""

from verifiers.v1.tasksets.lean import (
    LeanConfig,
    LeanDatasetConfig,
    LeanTaskset,
    build_starter_file,
    expected_protected_signature,
    parse_compile_output,
    protected_signature_substring_present,
    strip_lean_comments,
)


def test_needs_container():
    assert LeanTaskset.NEEDS_CONTAINER is True


def test_dataset_required():
    import pytest

    # `dataset` has no default — pydantic rejects an unset one at construction
    # (ValidationError is a ValueError subclass).
    with pytest.raises(ValueError):
        LeanConfig()


def test_validate_true_without_gold():
    # No gold proof => nothing to refute => valid (matches base no-op); returns
    # before touching the runtime, so None is safe. False is reserved for a gold
    # proof that fails to compile.
    import asyncio

    from verifiers.v1.tasksets.lean import LeanTask

    ts = LeanTaskset(LeanConfig(dataset=LeanDatasetConfig(name="x")))
    task = LeanTask(
        idx=0, prompt="p", formal_statement="theorem t : True := by", formal_proof=""
    )
    assert asyncio.run(ts.validate(task, None)) is True


def test_build_starter_file_plants_sorry():
    out = build_starter_file("theorem foo : 1 = 1 := by sorry")
    assert "import Mathlib" in out
    assert out.rstrip().endswith("sorry")
    assert ":= by" in out


def test_build_starter_file_with_gold_body():
    out = build_starter_file("theorem foo : 1 = 1 := by sorry", proof_body="  rfl")
    assert out.rstrip().endswith("rfl")
    assert (
        "\n  sorry\n" not in out
    )  # placeholder body replaced (backticked `sorry` in header is fine)


def test_signature_normalizes_to_by():
    assert (
        expected_protected_signature("theorem foo : 1 = 1 := by sorry")
        == "theorem foo : 1 = 1 := by"
    )
    assert expected_protected_signature("lemma bar : True := by admit").endswith(
        ":= by"
    )


def test_inline_imports_split_from_signature():
    stmt = "import Mathlib\n\ntheorem foo : 2 = 2 := by sorry"
    assert build_starter_file(stmt).count("theorem foo") == 1
    assert expected_protected_signature(stmt) == "theorem foo : 2 = 2 := by"


def test_guard_accepts_reflowed_whitespace():
    sig = "theorem foo : 1 = 1 := by"
    content = "import Mathlib\n\ntheorem    foo  :  1 = 1   :=   by\n  rfl\n"
    assert protected_signature_substring_present(content, sig)


def test_guard_rejects_rewritten_statement():
    sig = "theorem foo : 1 = 1 := by"
    content = "import Mathlib\n\ntheorem foo : True := by\n  trivial\n"
    assert not protected_signature_substring_present(content, sig)


def test_guard_rejects_statement_hidden_in_comment():
    sig = "theorem foo : 1 = 1 := by"
    content = "import Mathlib\n\n-- theorem foo : 1 = 1 := by\ntheorem foo : True := by\n  trivial\n"
    assert not protected_signature_substring_present(content, sig)


def test_guard_rejects_statement_hidden_in_string():
    sig = "theorem foo : 1 = 1 := by"
    content = 'import Mathlib\n\n#eval "theorem foo : 1 = 1 := by"\ntheorem foo : True := by trivial\n'
    assert not protected_signature_substring_present(content, sig)


def test_strip_lean_comments_handles_nested_blocks():
    stripped = strip_lean_comments("a /- outer /- inner -/ still -/ b -- line\nc")
    assert "outer" not in stripped and "inner" not in stripped
    assert "a" in stripped and "b" in stripped and "c" in stripped


def test_parse_compile_clean():
    compiled, _, code = parse_compile_output("EXIT_CODE:0")
    assert compiled and code == 0


def test_parse_compile_sorry_fails():
    compiled, _, code = parse_compile_output(
        "warning: declaration uses 'sorry'\nEXIT_CODE:0"
    )
    assert not compiled and code == 0


def test_parse_compile_uses_last_exit_marker():
    # An injected EXIT_CODE:0 earlier in the output must not mask the real failure.
    compiled, _, code = parse_compile_output(
        "fake EXIT_CODE:0 output\nerror: real failure\nEXIT_CODE:1"
    )
    assert not compiled and code == 1
