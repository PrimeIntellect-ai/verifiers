"""The episode envelope: flat self-contained traces, episodic durability.

The trace schema is flat — each trace carries its own `episode` stamp — while
`traces.jsonl` writes one `EpisodeRecord` per line, so an episode persists whole
or not at all: a torn line is the whole episode owed on resume, and a failure
before any trace minted still leaves its errors on disk.
"""

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
from verifiers.v1.trace import (
    AgentInfo,
    EpisodeInfo,
    EpisodeRecord,
    Error,
    Trace,
    TraceTask,
    WireTrace,
)


def _trace(idx: int = 0, name: str = "agent", error: bool = False) -> Trace:
    trace = Trace(
        task=TraceTask(type="Task", data=vf.TaskData(idx=idx, prompt="hi")),
        agent=AgentInfo(model="stub", name=name),
    )
    if error:
        trace.capture_error(ValueError("boom"))
    return trace


def _record(*traces: Trace, env: str = "demo-v1", errors: list | None = None):
    info = EpisodeInfo(env=env, errors=errors or [])
    for trace in traces:
        trace.episode = info
    return EpisodeRecord(episode=info, traces=list(traces))


def test_record_round_trip(tmp_path):
    record = _record(_trace(3, name="solver"))
    write_episode(tmp_path, record)
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.episode.id == record.episode.id and loaded.episode.env == "demo-v1"
    assert loaded.traces[0].agent_name == "solver" and loaded.traces[0].trainable
    # The trace is self-contained: its own stamp carries the same episode id.
    assert loaded.traces[0].episode.id == record.episode.id
    assert loaded.ok


def test_flat_traces_reconstitute_episodes():
    """A flat bag of traces regroups by its per-trace stamps — no side lookup."""
    a = _record(_trace(0, name="proposer"), _trace(0, name="solver"))
    b = _record(_trace(1))
    flat = [*a.traces, *b.traces]
    groups: dict[str, list[Trace]] = {}
    for trace in flat:
        groups.setdefault(trace.episode.id, []).append(trace)
    assert set(groups) == {a.episode.id, b.episode.id}
    assert [t.agent_name for t in groups[a.episode.id]] == ["proposer", "solver"]


def test_zero_trace_failure_persists_its_errors(tmp_path):
    """An env failure before any trace minted still leaves a durable record: the
    envelope carries the episode errors, resume owes the rollout again, and the
    failure is never mistaken for an empty success."""
    failed = _record(errors=[Error(type="EnvError", message="rollout() died")])
    write_episode(tmp_path, failed)
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.traces == [] and loaded.episode.error.message == "rollout() died"
    assert not loaded.ok
    episodes, owed = resume.load(tmp_path, [0], num_rollouts=1)
    assert episodes == [] and owed == {0: 1}


def test_pre_record_lines_sniff_and_load(tmp_path):
    # A file written before the episode atom: one bare trace per line.
    trace = _trace(1)
    row = trace.model_dump(mode="json", exclude_none=True)
    assert not sniff_episode(row)
    (tmp_path / TRACES_FILE).write_text(json.dumps(row) + "\n")
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.traces[0].id == trace.id  # wrapped as a single-trace record


def test_resume_keeps_good_records_and_owes_the_rest(tmp_path):
    good = _record(_trace(0))
    bad = _record(_trace(1, error=True))
    write_episode(tmp_path, good)
    write_episode(tmp_path, bad)
    episodes, owed = resume.load(tmp_path, [0, 1], num_rollouts=1)
    assert [r.episode.id for r in episodes] == [good.episode.id]
    assert owed == {1: 1}  # the errored rollout is redone as a unit
    # The file was rewritten to just the kept rows, verbatim.
    lines = (tmp_path / TRACES_FILE).read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["episode"]["id"] == good.episode.id


def test_resume_whole_task_redoes_partial_units(tmp_path):
    """`whole_task` (the legacy group-scored path): a partially-kept task is redone
    as a unit — its kept rows are dropped from memory AND the rewritten file, and the
    full count is owed again; a fully-kept task stays kept."""
    write_episode(tmp_path, _record(_trace(0)))
    for _ in range(2):
        write_episode(tmp_path, _record(_trace(1)))
    episodes, owed = resume.load(tmp_path, [0, 1], num_rollouts=2, whole_task=True)
    assert owed == {0: 2}  # the partial unit owes the WHOLE group
    assert all(e.traces[0].task.data.idx == 1 for e in episodes) and len(episodes) == 2
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
    """The sample's grouping columns come off the trace itself — its own episode
    stamp and agent info."""
    trace = _trace(5, name="judge")
    trace.agent.trainable = False
    trace.episode = EpisodeInfo(id="rec123")
    sample = trace_to_sample(trace, 1)
    assert sample["episode_id"] == "rec123"
    assert sample["agent"] == "judge" and sample["trainable"] is False


def test_push_samples_share_rollout_number_per_episode():
    """Siblings/agents of one episode are the same attempt at the task: they share
    its rollout_number instead of counting 1..n (which clashed with
    rollouts_per_example when -r was 1)."""
    from verifiers.v1.push import _build_samples

    a1, a2 = _trace(0, name="solver"), _trace(0, name="solver")  # A: fan-out
    b1, b2 = _trace(0, name="solver"), _trace(0, name="judge")  # B: judged
    for trace, episode_id in ((a1, "A"), (a2, "A"), (b1, "B"), (b2, "B")):
        trace.episode = EpisodeInfo(id=episode_id)
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
    both invisible to a flattened trace list, both visible on the records."""
    from verifiers.v1.push import _run_metrics

    score_failed = _record(
        _trace(0), errors=[Error(type="EnvError", message="score() died")]
    )
    zero_trace = _record(errors=[Error(type="EnvError", message="rollout() died")])
    ok = _record(_trace(2))
    episodes = [score_failed, zero_trace, ok]
    traces = [t for e in episodes for t in e.traces]
    assert _run_metrics(episodes, traces)["avg_error"] == 2 / 3


def test_append_trace_wraps_a_record(tmp_path):
    from verifiers.v1.cli.output import append_trace

    asyncio.run(append_trace(tmp_path, _trace(2), asyncio.Lock(), env="demo-v1"))
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.episode.env == "demo-v1" and len(loaded.traces) == 1


def test_resume_env_complete_override(tmp_path):
    """`Environment.complete` is the resume verdict: an env that tolerates errored
    participants keeps the rollouts it accepted; the default stays strict."""
    strict_dir, tolerant_dir = tmp_path / "strict", tmp_path / "tolerant"
    strict_dir.mkdir(), tolerant_dir.mkdir()
    record = _record(_trace(0, error=True))
    write_episode(strict_dir, record)
    write_episode(tolerant_dir, record)
    _, owed = resume.load(strict_dir, [0], num_rollouts=1)
    assert owed == {0: 1}  # default: an errored trace means redo
    episodes, owed = resume.load(
        tolerant_dir, [0], num_rollouts=1, complete=lambda r: not r.episode.errors
    )
    assert [r.episode.id for r in episodes] == [record.episode.id] and owed == {}


def test_legacy_bridge_answers_a_record():
    """The v0 bridge speaks protocol 2: `run` answers with `episode` (flat traces
    plus the shared stamp)."""
    from verifiers.v1 import legacy
    from verifiers.v1.clients.config import EvalClientConfig
    from verifiers.v1.serve.types import RunRequest
    from verifiers.v1.trace import WireEpisodeRecord
    from verifiers.v1.types import SamplingConfig

    server = legacy.LegacyEnvServer.__new__(legacy.LegacyEnvServer)
    server.taskset_id = "echo-v0"

    async def run_v0(task_idx, client, model, sampling):
        return {"prompt": [{"role": "user", "content": "hi"}], "reward": 1.0}

    server._run_v0 = run_v0
    req = RunRequest(
        task_idx=0, client=EvalClientConfig(), model="m", sampling=SamplingConfig()
    )
    resp = asyncio.run(server._run(req))
    record = WireEpisodeRecord.model_validate(resp.model_dump()["episode"])
    assert record.episode.env == "echo-v0" and record.ok
    assert record.traces[0].reward == 1.0


def test_prompt_less_record_round_trips(tmp_path):
    """The wire drops `None`s, so a prompt-less task's row must still read back
    (user-opened tasksets write them)."""
    trace = Trace(task=TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)))
    write_episode(tmp_path, EpisodeRecord.of(trace, env="demo-v1"))
    (loaded,) = read_episodes(tmp_path, WireTrace)
    assert loaded.traces[0].task.data.prompt is None


def test_resume_treats_torn_lines_as_owed(tmp_path):
    """A run killed mid-write leaves a torn final line; resume owes that WHOLE
    episode again instead of keeping a valid-looking prefix of it — the line is
    the atomicity unit."""
    write_episode(tmp_path, _record(_trace(0)))
    path = tmp_path / TRACES_FILE
    with path.open("ab") as f:
        f.write(b'{"episode": {"id": "torn"}, "traces": [{"task"')  # the torn write
    episodes, owed = resume.load(tmp_path, [0], num_rollouts=2)
    assert len(episodes) == 1 and owed == {0: 1}
    # The rewrite dropped the torn line: a second resume parses cleanly.
    episodes, owed = resume.load(tmp_path, [0], num_rollouts=2)
    assert len(episodes) == 1 and owed == {0: 1}
