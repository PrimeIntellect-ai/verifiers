"""The RolloutRecord wire atom: round-trip, old-shape sniffing, resume both shapes."""

import asyncio
import json

import verifiers.v1 as vf
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.output import (
    TRACES_FILE,
    read_records,
    sniff_record,
    write_record,
)
from verifiers.v1.push import trace_to_sample
from verifiers.v1.trace import RolloutRecord, Trace, TraceTask, WireTrace


def _trace(idx: int = 0, role: str | None = None, error: bool = False) -> Trace:
    trace = Trace(
        task=TraceTask(type="Task", data=vf.TaskData(idx=idx, prompt="hi")),
        role=role,
    )
    if error:
        trace.capture_error(ValueError("boom"))
    return trace


def test_record_round_trip(tmp_path):
    record = RolloutRecord(
        env="demo-v1", task=_trace(3).task, traces=[_trace(3, role="solver")]
    )
    write_record(tmp_path, record)
    (loaded,) = read_records(tmp_path, WireTrace)
    assert loaded.id == record.id and loaded.env == "demo-v1"
    assert loaded.traces[0].role == "solver" and loaded.traces[0].trainable
    assert loaded.ok


def test_pre_record_lines_sniff_and_load(tmp_path):
    # A file written before the record atom: one bare trace per line.
    trace = _trace(1)
    row = trace.model_dump(mode="json", exclude_none=True)
    assert not sniff_record(row)
    (tmp_path / TRACES_FILE).write_text(json.dumps(row) + "\n")
    (loaded,) = read_records(tmp_path, WireTrace)
    assert loaded.traces[0].id == trace.id  # wrapped as a single-trace record


def test_resume_keeps_good_records_and_owes_the_rest(tmp_path):
    good = RolloutRecord.of(_trace(0), env="demo-v1")
    bad = RolloutRecord.of(_trace(1, error=True), env="demo-v1")
    write_record(tmp_path, good)
    write_record(tmp_path, bad)
    records, owed = resume.load(tmp_path, [0, 1], num_rollouts=1)
    assert [r.id for r in records] == [good.id]
    assert owed == {1: 1}  # the errored rollout is redone as a unit
    # The file was rewritten to just the kept rows, verbatim.
    lines = (tmp_path / TRACES_FILE).read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["id"] == good.id


def test_resume_reads_pre_record_files(tmp_path):
    rows = [
        _trace(0).model_dump(mode="json", exclude_none=True),
        _trace(0, error=True).model_dump(mode="json", exclude_none=True),
    ]
    (tmp_path / TRACES_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    records, owed = resume.load(tmp_path, [0], num_rollouts=2)
    assert len(records) == 1 and records[0].traces[0].id == rows[0]["id"]
    assert owed == {0: 1}


def test_push_sample_carries_record_grouping():
    trace = _trace(5, role="judge")
    trace.trainable = False
    sample = trace_to_sample(trace, 1, record_id="rec123")
    assert sample["record_id"] == "rec123"
    assert sample["role"] == "judge" and sample["trainable"] is False


def test_append_trace_wraps_a_record(tmp_path):
    from verifiers.v1.cli.output import append_trace

    asyncio.run(append_trace(tmp_path, _trace(2), asyncio.Lock(), env="demo-v1"))
    (loaded,) = read_records(tmp_path, WireTrace)
    assert loaded.env == "demo-v1" and len(loaded.traces) == 1


def test_resume_env_complete_override(tmp_path):
    """`Environment.complete` is the resume verdict: an env that tolerates errored
    participants keeps the rollouts it accepted; the default stays strict."""
    strict_dir, tolerant_dir = tmp_path / "strict", tmp_path / "tolerant"
    strict_dir.mkdir(), tolerant_dir.mkdir()
    record = RolloutRecord.of(_trace(0, error=True), env="demo-v1")
    write_record(strict_dir, record)
    write_record(tolerant_dir, record)
    _, owed = resume.load(strict_dir, [0], num_rollouts=1)
    assert owed == {0: 1}  # default: an errored trace means redo
    records, owed = resume.load(
        tolerant_dir, [0], num_rollouts=1, complete=lambda r: not r.errors
    )
    assert [r.id for r in records] == [record.id] and owed == {}


def test_legacy_bridge_run_rollout_wraps_a_record():
    """The v0 bridge speaks protocol 2: `run_rollout` answers with `record`. (The
    response type renamed its field from `trace`, and pydantic drops an unknown
    kwarg silently — a bridged rollout must survive the rename.)"""
    from verifiers.v1 import legacy
    from verifiers.v1.clients.config import EvalClientConfig
    from verifiers.v1.serve.types import RunRolloutRequest
    from verifiers.v1.trace import WireRecord
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
    record = WireRecord.model_validate(resp.model_dump()["record"])
    assert record.env == "echo-v0" and record.ok
    assert record.traces[0].reward == 1.0
