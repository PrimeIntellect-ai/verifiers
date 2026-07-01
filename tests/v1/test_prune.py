"""`--prune` drops errored rows from a finished run's results.jsonl, and composes with `--resume`.

Type selection mirrors `--retries.rollout.include`/`exclude`: `--prune-include` keeps the prune to
listed error types, `--prune-exclude` spares them (exclude wins), and an empty include prunes all
errored rows. `--resume <dir> --prune` prunes first, then resumes the same dir.
"""

import json
from pathlib import Path

import pytest

from verifiers.v1.cli.eval import main as eval_main
from verifiers.v1.cli.eval.prune import prune_results, split_prune

ROWS = [
    {"task": {"idx": 0}, "rewards": {"solved": 1.0}},
    {"task": {"idx": 1}, "errors": [{"type": "SandboxError"}]},
    {"task": {"idx": 2}, "errors": [{"type": "ProviderError"}]},
    {"task": {"idx": 3}, "rewards": {"solved": 0.0}},  # clean, reward 0 (not errored)
]


def _write(d: Path) -> Path:
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    return d


def _kept(d: Path) -> list[int]:
    lines = (d / "results.jsonl").read_text().splitlines()
    return [json.loads(line)["task"]["idx"] for line in lines if line.strip()]


def test_prune_drops_all_errored_by_default(tmp_path: Path) -> None:
    prune_results(_write(tmp_path), [], [])
    assert _kept(tmp_path) == [0, 3]


def test_prune_include_limits_to_listed(tmp_path: Path) -> None:
    prune_results(_write(tmp_path), ["SandboxError"], [])
    assert _kept(tmp_path) == [0, 2, 3]  # the ProviderError row survives


def test_prune_exclude_spares_listed(tmp_path: Path) -> None:
    prune_results(_write(tmp_path), [], ["ProviderError"])
    assert _kept(tmp_path) == [
        0,
        2,
        3,
    ]  # every errored row pruned except the excluded one


def test_prune_exclude_wins_over_include(tmp_path: Path) -> None:
    prune_results(
        _write(tmp_path), ["SandboxError", "ProviderError"], ["ProviderError"]
    )
    assert _kept(tmp_path) == [
        0,
        2,
        3,
    ]  # ProviderError excluded even though it's in include


def test_prune_keeps_everything_when_type_absent(tmp_path: Path) -> None:
    prune_results(_write(tmp_path), ["NopeError"], [])
    assert _kept(tmp_path) == [0, 1, 2, 3]


def test_prune_missing_results_dir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        prune_results(tmp_path, [], [])  # no results.jsonl


def test_split_prune_forms() -> None:
    assert split_prune(["--prune", "/d"]) == (True, Path("/d"), [], [], [])
    assert split_prune(["--prune=/d"]) == (True, Path("/d"), [], [], [])
    # bare `--prune` (next token is a flag) leaves the dir to `--resume`
    assert split_prune(["--prune", "--prune-include", "A,B"]) == (
        True,
        None,
        ["A", "B"],
        [],
        [],
    )
    assert split_prune(["--prune=/d", "--prune-exclude=C"]) == (
        True,
        Path("/d"),
        [],
        ["C"],
        [],
    )
    assert split_prune(["foo", "--prune", "bar"]) == (
        True,
        Path("bar"),
        [],
        [],
        ["foo"],
    )
    assert split_prune(["x", "y"]) == (False, None, [], [], ["x", "y"])


def test_split_prune_rejects_empty_values() -> None:
    # empty `--prune=` / `--prune-include=` / `--prune-exclude=` must fail explicitly, not
    # default to cwd or vanish silently
    for argv in (
        ["--prune="],
        ["--prune", "/d", "--prune-include="],
        ["--prune", "/d", "--prune-exclude="],
    ):
        with pytest.raises(SystemExit):
            split_prune(argv)


class _Cfg:
    is_legacy = False


class _Stop(Exception):
    pass


def _patch_resume_and_prune(monkeypatch: pytest.MonkeyPatch, order: list) -> None:
    """Record (and stop after) `load_resume_config` / `prune_results` calls in `order`."""

    def _load(d):
        order.append(("load", str(d)))
        return _Cfg()

    def _prune(d, inc, exc):
        order.append(("prune", str(d), inc, exc))
        raise _Stop()

    monkeypatch.setattr(eval_main, "load_resume_config", _load)
    monkeypatch.setattr(eval_main, "prune_results", _prune)


def test_main_prune_only_does_not_load_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list = []
    _patch_resume_and_prune(monkeypatch, order)
    with pytest.raises(_Stop):
        eval_main.main(["--prune", "/d"])
    assert order == [("prune", "/d", [], [])]  # no resume-config load


def test_main_loads_resume_before_pruning(monkeypatch: pytest.MonkeyPatch) -> None:
    # `--resume --prune` must load the config (legacy guard) before rewriting anything
    order: list = []
    _patch_resume_and_prune(monkeypatch, order)
    with pytest.raises(_Stop):
        eval_main.main(["--resume", "/d", "--prune", "--prune-include", "SandboxError"])
    assert order == [("load", "/d"), ("prune", "/d", ["SandboxError"], [])]


def test_main_accepts_equivalent_resume_prune_dirs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `out` and `./out` resolve to the same dir, so the combined form isn't a mismatch
    order: list = []
    _patch_resume_and_prune(monkeypatch, order)
    with pytest.raises(_Stop):
        eval_main.main(["--resume", "out", "--prune", "./out"])
    assert [step[0] for step in order] == ["load", "prune"]


def test_main_rejects_legacy_resume_before_pruning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a legacy (v0) `--resume --prune` errors without rewriting results.jsonl
    pruned: list = []

    class _Legacy:
        is_legacy = True

    monkeypatch.setattr(eval_main, "load_resume_config", lambda d: _Legacy())
    monkeypatch.setattr(
        eval_main, "prune_results", lambda d, inc, exc: pruned.append(str(d))
    )
    with pytest.raises(SystemExit):
        eval_main.main(["--resume", "/legacy", "--prune"])
    assert pruned == []


def test_main_rejects_bad_prune_combos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_main, "prune_results", lambda d, inc, exc: None)
    monkeypatch.setattr(eval_main, "load_resume_config", lambda d: None)
    with pytest.raises(
        SystemExit
    ):  # bare --prune, no dir and no --resume to supply one
        eval_main.main(["--prune"])
    with pytest.raises(SystemExit):  # --prune-include without --prune
        eval_main.main(["--resume", "/d", "--prune-include", "X"])
    with pytest.raises(SystemExit):  # --prune and --resume disagree on the dir
        eval_main.main(["--resume", "/a", "--prune", "/b"])
