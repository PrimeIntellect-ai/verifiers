"""The Episode wire atom: round-trip, old-shape sniffing, resume both shapes."""

import asyncio
import json

import verifiers.v1 as vf
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.output import (
    TRACES_FILE,
    read_episodes,
    sniff_episode,
    write_episode,
)
from verifiers.v1.push import trace_to_sample
from verifiers.v1.trace import AgentInfo, Episode, Trace, TraceTask, WireTrace


def _trace(idx: int = 0, role: str | None = None, error: bool = False) -> Trace:
    trace = Trace(
        task=TraceTask(type="Task", data=vf.TaskData(idx=idx, prompt="hi")),
        agent=AgentInfo(model="stub", name=role) if role is not None else None,
    )
    if error:
        trace.capture_error(ValueError("boom"))
    return trace


def test_record_round_trip(tmp_path):
    episode = Episode(
        env="demo-v1", task=_trace(3).task, traces=[_trace(3, role="solver")]
    )
    write_episode(tmp_path, episode)
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.id == episode.id and loaded.env == "demo-v1"
    assert loaded.traces[0].agent_name == "solver" and loaded.traces[0].trainable
    assert loaded.ok


def test_views_reconstitute_roles_from_stamps():
    """`views` groups traces by role stamp in mint order (a fanned-out seat is its
    list); unstamped traces (the single-agent shape) show no views."""
    solves = [_trace(0, role="solver"), _trace(0, role="solver")]
    episode = Episode(
        env="demo-v1",
        task=_trace(0).task,
        traces=[_trace(0, role="proposer"), *solves],
    )
    assert list(episode.views) == ["proposer", "solver"]
    assert episode.views["solver"] == solves
    assert Episode.of(_trace(1), env="demo-v1").views == {}


def test_pre_record_lines_sniff_and_load(tmp_path):
    # A file written before the episode atom: one bare trace per line.
    trace = _trace(1)
    row = trace.model_dump(mode="json", exclude_none=True)
    assert not sniff_episode(row)
    (tmp_path / TRACES_FILE).write_text(json.dumps(row) + "\n")
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.traces[0].id == trace.id  # wrapped as a single-trace episode


def test_resume_keeps_good_records_and_owes_the_rest(tmp_path):
    good = Episode.of(_trace(0), env="demo-v1")
    bad = Episode.of(_trace(1, error=True), env="demo-v1")
    write_episode(tmp_path, good)
    write_episode(tmp_path, bad)
    episodes, owed = resume.load(tmp_path, [0, 1], num_rollouts=1)
    assert [r.id for r in episodes] == [good.id]
    assert owed == {1: 1}  # the errored rollout is redone as a unit
    # The file was rewritten to just the kept rows, verbatim.
    lines = (tmp_path / TRACES_FILE).read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["id"] == good.id


def test_resume_whole_task_redoes_partial_units(tmp_path):
    """`whole_task` (the legacy group-scored path): a partially-kept task is redone
    as a unit — its kept rows are dropped from memory AND the rewritten file, and the
    full count is owed again; a fully-kept task stays kept."""
    partial = Episode.of(_trace(0), env="demo-v1")
    write_episode(tmp_path, partial)
    for _ in range(2):
        write_episode(tmp_path, Episode.of(_trace(1), env="demo-v1"))
    episodes, owed = resume.load(tmp_path, [0, 1], num_rollouts=2, whole_task=True)
    assert owed == {0: 2}  # the partial unit owes the WHOLE group
    assert all(e.task.data.idx == 1 for e in episodes) and len(episodes) == 2
    lines = (tmp_path / TRACES_FILE).read_text().strip().splitlines()
    assert len(lines) == 2  # idx 0's kept row left the file too


def test_resume_reads_pre_record_files(tmp_path):
    rows = [
        _trace(0).model_dump(mode="json", exclude_none=True),
        _trace(0, error=True).model_dump(mode="json", exclude_none=True),
    ]
    (tmp_path / TRACES_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    episodes, owed = resume.load(tmp_path, [0], num_rollouts=2)
    assert len(episodes) == 1 and episodes[0].traces[0].id == rows[0]["id"]
    assert owed == {0: 1}


def test_push_sample_carries_record_grouping():
    """The sample's grouping columns come off the trace itself — it carries its
    own episode standing on `agent`."""
    trace = _trace(5, role="judge")
    trace.agent.trainable = False
    trace.agent.episode = "rec123"
    sample = trace_to_sample(trace, 1)
    assert sample["episode_id"] == "rec123"
    assert sample["agent"] == "judge" and sample["trainable"] is False


def test_push_samples_share_rollout_number_per_episode():
    """Siblings/seats of one episode are the same attempt at the task: they share
    its rollout_number instead of counting 1..n (which clashed with
    rollouts_per_example when -r was 1)."""
    from verifiers.v1.push import _build_samples

    a1, a2 = _trace(0, role="solver"), _trace(0, role="solver")  # episode A: fan-out
    b1, b2 = _trace(0, role="solver"), _trace(0, role="judge")  # episode B: judged
    for trace, episode_id in ((a1, "A"), (a2, "A"), (b1, "B"), (b2, "B")):
        trace.agent.episode = episode_id
    samples = _build_samples([a1, a2, b1, b2])
    assert [s["rollout_number"] for s in samples] == [1, 1, 2, 2]
    # Unstamped traces (a pre-episode file) keep counting one attempt each.
    assert [s["rollout_number"] for s in _build_samples([_trace(0), _trace(0)])] == [
        1,
        2,
    ]


def test_push_error_rate_counts_episodes():
    """avg_error mirrors the dashboard's err: episode-level failures count even when
    their traces are clean (score() raised) or absent (rollout() died pre-trace) —
    both invisible to the flattened trace list."""
    from verifiers.v1.push import _EpisodeIndex, _run_metrics

    clean = _trace(0)
    index = _EpisodeIndex(
        # A: clean trace, score() failed; B: zero-trace rollout() failure; C: ok.
        ok={"A": False, "B": False, "C": True},
        idx={"A": 0, "B": 1, "C": 2},
    )
    assert _run_metrics([clean], index)["avg_error"] == 2 / 3
    # Without the file, the per-trace fallback (the old rule).
    bare = _EpisodeIndex(ok={}, idx={})
    assert _run_metrics([clean], bare)["avg_error"] == 0.0


def test_episode_index_sees_zero_trace_failures(tmp_path, monkeypatch):
    """The index reads outcomes off traces.jsonl — including an env-rollout that
    failed before minting any trace, which the in-memory trace list can't carry."""
    import verifiers.v1.cli.output as output
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.push import _episode_index
    from verifiers.v1.trace import Error

    good = Episode.of(_trace(0), env="demo-v1")
    hook_failed = Episode(
        env="demo-v1",
        task=_trace(1).task,
        errors=[Error(type="TaskError", message="rollout() died")],
    )
    trace_failed = Episode.of(_trace(2, error=True), env="demo-v1")
    for episode in (good, hook_failed, trace_failed):
        write_episode(tmp_path, episode)
    monkeypatch.setattr(output, "output_path", lambda config: tmp_path)
    index = _episode_index(EvalConfig())
    assert index.ok == {good.id: True, hook_failed.id: False, trace_failed.id: False}
    assert index.idx == {good.id: 0, hook_failed.id: 1, trace_failed.id: 2}


def test_append_trace_wraps_a_record(tmp_path):
    from verifiers.v1.cli.output import append_trace

    asyncio.run(append_trace(tmp_path, _trace(2), asyncio.Lock(), env="demo-v1"))
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.env == "demo-v1" and len(loaded.traces) == 1


def test_resume_env_complete_override(tmp_path):
    """`Environment.complete` is the resume verdict: an env that tolerates errored
    participants keeps the rollouts it accepted; the default stays strict."""
    strict_dir, tolerant_dir = tmp_path / "strict", tmp_path / "tolerant"
    strict_dir.mkdir(), tolerant_dir.mkdir()
    episode = Episode.of(_trace(0, error=True), env="demo-v1")
    write_episode(strict_dir, episode)
    write_episode(tolerant_dir, episode)
    _, owed = resume.load(strict_dir, [0], num_rollouts=1)
    assert owed == {0: 1}  # default: an errored trace means redo
    episodes, owed = resume.load(
        tolerant_dir, [0], num_rollouts=1, complete=lambda r: not r.errors
    )
    assert [r.id for r in episodes] == [episode.id] and owed == {}


def test_legacy_bridge_run_rollout_wraps_a_record():
    """The v0 bridge speaks the episode protocol: `run_rollout` answers with
    `episode`. (The response type renamed its field from `trace`, and pydantic drops
    an unknown kwarg silently — a bridged rollout must survive the rename.)"""
    from verifiers.v1 import legacy
    from verifiers.v1.clients.config import EvalClientConfig
    from verifiers.v1.serve.types import RunRolloutRequest
    from verifiers.v1.trace import WireEpisode
    from verifiers.v1.types import SamplingConfig

    server = legacy.LegacyEnvServer.__new__(legacy.LegacyEnvServer)
    server.taskset_id = "echo-v0"

    async def run_v0(task_idx, client, model, sampling):
        return {"prompt": [{"role": "user", "content": "hi"}], "reward": 1.0}

    server._run_v0 = run_v0
    req = RunRolloutRequest(
        task_idx=0, client=EvalClientConfig(), model="m", sampling=SamplingConfig()
    )
    resp = asyncio.run(server._run_rollout(req))
    episode = WireEpisode.model_validate(resp.model_dump()["episode"])
    assert episode.env == "echo-v0" and episode.ok
    assert episode.traces[0].reward == 1.0


def test_prompt_less_record_round_trips(tmp_path):
    """The wire drops `None`s, so a prompt-less task's row must still read back
    (user-opened tasksets write them)."""
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)))
    write_episode(tmp_path, Episode.of(trace, env="demo-v1"))
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.traces[0].task.data.prompt is None


def test_resume_treats_torn_lines_as_owed(tmp_path):
    """A run killed mid-write leaves a torn final line; resume owes that rollout
    again instead of crashing (the episode is the resume unit)."""
    write_episode(tmp_path, Episode.of(_trace(0), env="demo-v1"))
    path = tmp_path / TRACES_FILE
    with path.open("ab") as f:
        f.write(b'{"task": {"data": {"idx"')  # the torn write
    episodes, owed = resume.load(tmp_path, [0], num_rollouts=2)
    assert len(episodes) == 1 and owed == {0: 1}
    # The rewrite dropped the torn line: a second resume parses cleanly.
    episodes, owed = resume.load(tmp_path, [0], num_rollouts=2)
    assert len(episodes) == 1 and owed == {0: 1}
