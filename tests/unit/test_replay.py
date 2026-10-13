"""Offline re-scoring: eval -> replay -> replay-the-replay agree, on a saved run built
here rather than a live eval."""

import json
from pathlib import Path

import pytest
from reverse_text.taskset import ReverseTextData, ReverseTextTask

import verifiers.v1 as vf
from verifiers.v1.cli.output import CONFIG_DIR, RESOLVED_DIR, saved_config_path
from verifiers.v1.cli.replay import run_replay
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.cli.replay import ReplayConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage
from verifiers.v1.utils.trace_store import write_episode


async def saved_run(run_dir: Path) -> vf.Trace:
    """A one-trace reverse-text run on disk, as `run_eval` leaves it: the resolved
    config under `configs/resolved/` and a scored trace in `traces.jsonl`."""
    config = EvalConfig(env={"taskset": {"id": "reverse-text"}}, rich=None, push=False)
    resolved = run_dir / CONFIG_DIR / RESOLVED_DIR
    resolved.mkdir(parents=True)
    (resolved / "eval.json").write_text(json.dumps(config.model_dump(mode="json")))
    data = ReverseTextData(idx=0, prompt="abc", answer="cba")
    task = ReverseTextTask(data)
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="ReverseTextTask", data=data, key=task.key),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="abc")),
            MessageNode(
                parent=0,
                message=AssistantMessage(content="<reversed_text>cba</reversed_text>"),
                sampled=True,
            ),
        ],
    )
    trace.stop("agent_completed")
    trace.ok = True
    await task.score(trace)
    write_episode(run_dir, vf.Episode(task=trace.task, traces=[trace], ok=True))
    return trace


async def test_replay_round_trip(tmp_path):
    """Offline re-scoring must preserve the saved task's wire form: replay reads traces as
    `Trace[WireTaskData, ...]`, so its own output dumps through that schema — the
    taskset-specific fields (reverse-text's `answer`) ride `model_extra` and must survive
    into the replay's `traces.jsonl`, or the next replay's typed rebuild fails and the
    trace-only `@reward` silently stops running (the wire-narrowing regression). Trace-only
    rewards are deterministic given the transcript, so all three generations must agree."""
    run_dir = tmp_path / "run"
    source = await saved_run(run_dir)
    assert source.rewards["lcs"].score == 1.0

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
