"""Resume planning over a saved results.jsonl."""

import json

from verifiers.v1.cli import resume


def _write_results(path, rows):
    # Mirror append_trace: one JSON record per "\n", non-ASCII left literal.
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def test_plan_keeps_traces_containing_unicode_line_separators(tmp_path):
    # A trace's content can carry Unicode line separators (here U+0085 NEL) that str
    # .splitlines() would split a record on — shredding a valid record into fragments that
    # fail to parse. Records are "\n"-delimited, so a NEL-bearing trace must read as one row.
    rows = [
        {"task": {"idx": 0}, "reward": 1.0, "out": "before\x85after"},
        {"task": {"idx": 1}, "reward": 1.0, "out": "ok"},
    ]
    _write_results(tmp_path / "results.jsonl", rows)

    keep, owed = resume.plan(
        tmp_path, selected_idxs=[0, 1], num_rollouts=1, group=False
    )

    assert sorted(r["task"]["idx"] for r in keep) == [0, 1]
    assert owed == {}


def test_plan_redoes_errored_and_missing_rollouts(tmp_path):
    rows = [
        {"task": {"idx": 0}, "errors": []},  # good -> kept
        {"task": {"idx": 0}, "errors": ["boom"]},  # errored -> dropped + owed
    ]
    _write_results(tmp_path / "results.jsonl", rows)

    keep, owed = resume.plan(
        tmp_path, selected_idxs=[0, 1], num_rollouts=2, group=False
    )

    assert len(keep) == 1 and keep[0]["task"]["idx"] == 0
    assert owed == {0: 1, 1: 2}  # idx 0 owes the dropped one; idx 1 never ran
