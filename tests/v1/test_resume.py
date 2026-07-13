"""Resume keeps good flat traces and re-runs missing or errored rollouts."""

import json
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.output import TRACES_FILE
from verifiers.v1.graph import MessageNode
from verifiers.v1.topology import AgentGraph, graph_complete
from verifiers.v1.trace import Error, Trace, TraceTask
from verifiers.v1.types import AssistantMessage, UserMessage


def make_trace(idx: int, errored: bool = False) -> Trace:
    task = vf.Task(vf.TaskData(idx=idx, prompt="hi"))
    trace = Trace(task=TraceTask(type=type(task).__name__, data=task.data))
    trace.nodes.append(
        MessageNode(parent=None, message=UserMessage(content=str(task.data.prompt)))
    )
    trace.nodes.append(
        MessageNode(parent=0, message=AssistantMessage(content="ok"), sampled=True)
    )
    if errored:
        trace.errors.append(Error(type="HarnessError", message="boom"))
    trace.stop("error" if errored else "agent_completed")
    return trace


def write_traces(path: Path, traces: list[Trace]) -> None:
    lines = [t.model_dump_json(exclude_none=True) for t in traces]
    (path / TRACES_FILE).write_text("\n".join(lines) + "\n")


def test_resume_keeps_clean_traces_and_redoes_errors(tmp_path):
    write_traces(
        tmp_path,
        [make_trace(0), make_trace(1, errored=True), make_trace(2)],
    )
    finished, owed = resume.load(tmp_path, [0, 1, 2], num_rollouts=1)
    assert [t.task.data.idx for t in finished] == [0, 2]
    assert owed == {1: 1}
    kept = [
        json.loads(line)
        for line in (tmp_path / TRACES_FILE).read_text().splitlines()
        if line.strip()
    ]
    assert [row["task"]["data"]["idx"] for row in kept] == [0, 2]


def test_owed_counts_missing_rollouts(tmp_path):
    write_traces(tmp_path, [make_trace(0), make_trace(0)])
    finished, owed = resume.load(tmp_path, [0, 1], num_rollouts=2)
    assert len(finished) == 2
    assert owed == {1: 2}


def test_topology_complete_defaults_and_overrides():
    """`Topology.complete` is still the instance-validity hook for topologies; eval
    resume no longer consumes it (flat traces), but the contract remains for callers."""
    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    clean = AgentGraph(
        topology="single-agent",
        task=TraceTask(type=type(task).__name__, data=task.data),
    )
    clean.add(make_trace(0))
    errored = AgentGraph(
        topology="single-agent",
        task=TraceTask(type=type(task).__name__, data=task.data),
    )
    errored.add(make_trace(1, errored=True))
    assert graph_complete(clean) and not graph_complete(errored)

    topology = vf.SingleAgentTopology(vf.SingleAgentTopologyConfig())
    assert topology.complete(clean) and not topology.complete(errored)

    class Tolerant(vf.Topology[vf.TopologyConfig]):
        def complete(self, graph: AgentGraph) -> bool:
            return graph.error is None

    assert Tolerant(vf.TopologyConfig()).complete(errored)
