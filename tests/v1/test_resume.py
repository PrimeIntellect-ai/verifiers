"""Resume keeps the graph records the topology calls complete and re-runs the rest.

The keep/re-run verdict is `Topology.complete` — conservative by default (any error,
instance-level or nested, means redo; exact for the single-agent lowering, where the one
trace IS the invocation), overridable by topologies whose `go` tolerates child failures.
"""

import json
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.cli.eval import resume
from verifiers.v1.cli.output import TRACES_FILE
from verifiers.v1.graph import MessageNode
from verifiers.v1.topology import AgentGraph, graph_complete
from verifiers.v1.trace import Error, Trace, TraceTask
from verifiers.v1.types import AssistantMessage, UserMessage


def make_trace(task: vf.Task, errored: bool = False) -> Trace:
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


def make_graph(
    idx: int, trace_errored: bool = False, go_crashed: bool = False
) -> AgentGraph:
    task = vf.Task(vf.TaskData(idx=idx, prompt="hi"))
    graph = AgentGraph(
        topology="single-agent",
        task=TraceTask(type=type(task).__name__, data=task.data),
    )
    graph.add(make_trace(task, errored=trace_errored))
    if go_crashed:
        graph.error = Error(type="TopologyError", message="go crashed")
    return graph


def write_records(path: Path, graphs: list[AgentGraph]) -> None:
    lines = [json.dumps(g.to_record()) for g in graphs]
    (path / TRACES_FILE).write_text("\n".join(lines) + "\n")


def test_default_rule_redoes_any_error(tmp_path):
    """The conservative default: a clean graph is kept; a nested trace error or an
    instance-level error means the invocation is owed again — the single-agent lowering's
    old flat-trace resume behavior, graph-shaped."""
    write_records(
        tmp_path,
        [
            make_graph(0),
            make_graph(1, trace_errored=True),
            make_graph(2, go_crashed=True),
        ],
    )
    finished, owed = resume.load(tmp_path, [0, 1, 2], num_rollouts=1)
    assert [g.task.data.idx for g in finished] == [0]
    assert owed == {1: 1, 2: 1}
    # The kept lines were rewritten verbatim; the dropped records are gone from disk.
    kept = [
        json.loads(line)
        for line in (tmp_path / TRACES_FILE).read_text().splitlines()
        if line.strip()
    ]
    assert [row["task"]["data"]["idx"] for row in kept] == [0]


def test_tolerant_topology_keeps_scored_instances(tmp_path):
    """A topology whose `go` tolerates child failures overrides `complete` (typically
    `graph.error is None`): its trace-errored-but-finished instances are kept, not redone."""
    write_records(
        tmp_path,
        [
            make_graph(0),
            make_graph(1, trace_errored=True),
            make_graph(2, go_crashed=True),
        ],
    )
    finished, owed = resume.load(
        tmp_path, [0, 1, 2], num_rollouts=1, complete=lambda g: g.error is None
    )
    assert [g.task.data.idx for g in finished] == [0, 1]
    assert owed == {2: 1}


def test_owed_counts_missing_invocations(tmp_path):
    """`num_rollouts` is the per-seed target: kept records count toward it, the rest is
    owed — including seeds with no record at all."""
    write_records(tmp_path, [make_graph(0), make_graph(0)])
    finished, owed = resume.load(tmp_path, [0, 1], num_rollouts=2)
    assert len(finished) == 2
    assert owed == {1: 2}  # a fully-covered seed owes nothing


def test_topology_complete_defaults_and_overrides():
    """`Topology.complete` defaults to the conservative module-level rule; a subclass
    override is what resume consumes."""
    clean, errored = make_graph(0), make_graph(1, trace_errored=True)
    assert graph_complete(clean) and not graph_complete(errored)

    topology = vf.SingleAgentTopology(vf.SingleAgentTopologyConfig())
    assert topology.complete(clean) and not topology.complete(errored)

    class Tolerant(vf.Topology[vf.TopologyConfig]):
        def complete(self, graph: AgentGraph) -> bool:
            return graph.error is None

    assert Tolerant(vf.TopologyConfig()).complete(errored)
