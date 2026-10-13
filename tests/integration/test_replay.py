"""Offline re-scoring: eval -> replay -> replay-the-replay agree."""

import json
from pathlib import Path

import pytest

from verifiers.v1.cli.output import saved_config_path
from verifiers.v1.cli.replay import run_replay
from verifiers.v1.configs.cli.replay import ReplayConfig


async def test_replay_round_trip(run_v1, tmp_path):
    """Offline re-scoring must preserve the saved task's wire form: replay reads traces as
    `Trace[WireTaskData, ...]`, so its own output dumps through that schema — the
    taskset-specific fields (reverse-text's `answer`) ride `model_extra` and must survive
    into the replay's `traces.jsonl`, or the next replay's typed rebuild fails and the
    trace-only `@reward` silently stops running (the wire-narrowing regression). Trace-only
    rewards are deterministic given the transcript, so all three generations must agree."""
    run_dir = tmp_path / "run"
    (source,) = await run_v1(
        "reverse-text",
        harness="null",
        runtime={"type": "subprocess"},
        output_dir=run_dir,
        max_turns=2,
    )
    assert source.ok
    assert "lcs" in source.rewards

    # Replay must not mix the source run's judge transcript into newly computed
    # judge calls. Seed a saved call directly because this task scores without a
    # judge; cleanup happens before task scoring and therefore does not inspect it.
    stream = run_dir / "traces.jsonl"
    record = json.loads(stream.read_text())
    record["traces"][0]["info"]["judge_calls"] = [
        {
            "name": "source-judge",
            "request": {"model": "source-model", "messages": []},
            "response": {
                "message": {"role": "assistant", "content": "stale"},
                "parsed": None,
                "usage": None,
            },
        }
    ]
    stream.write_text(json.dumps(record) + "\n")

    async def replay(source_dir: Path, out: Path):
        # The CLI's layering, minus the argv plumbing: the saved run's config is the base
        # (`ReplayConfig` ignores its eval-only keys), the source's output_dir is dropped.
        data = json.loads(saved_config_path(source_dir).read_text())
        data.pop("output_dir", None)
        config = ReplayConfig(**{**data, "rich": False})
        (trace,) = await run_replay(config, source_dir, out)
        return trace

    first = await replay(run_dir, tmp_path / "replay1")
    second = await replay(tmp_path / "replay1", tmp_path / "replay2")
    for replayed in (first, second):
        assert replayed.ok
        # The typed rebuild ran (not the base-Task fallback): the trace-only reward re-ran
        # and recomputed the same value.
        assert replayed.rewards.keys() == source.rewards.keys()
        assert replayed.reward == pytest.approx(source.reward)
        assert "judge_calls" not in replayed.info
    # The wire task keeps its taskset-specific fields in the replay's own output.
    raw = (tmp_path / "replay2" / "traces.jsonl").read_text()
    assert '"answer"' in raw
