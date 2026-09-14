"""Compile a `Flow` class into an immutable `Graph` and check it before anything runs.

Checks are named and individually skippable (`Flow.skip_checks`); every failure is
collected and raised together as one `FlowError`.
"""

from __future__ import annotations

import hashlib
import json
import typing
from collections.abc import Callable
from dataclasses import dataclass, replace

from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.flow.flow import Flow, FlowConfig
from verifiers.v1.flow.nodes import ExpandNode, FnNode, Node, RunNode


class FlowError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass(frozen=True)
class Graph:
    name: str
    entry: str
    nodes: dict[str, Node]
    config_type: type[FlowConfig]

    @property
    def preds(self) -> dict[str, set[str]]:
        preds: dict[str, set[str]] = {name: set() for name in self.nodes}
        for node in self.nodes.values():
            for target in node.successors():
                preds.setdefault(target, set()).add(node.name)
        return preds

    @property
    def held(self) -> set[str]:
        """Nodes whose runtime must stay alive after they finish (someone inherits it)."""
        return {node.inherits for node in self.nodes.values() if node.inherits}

    def reachable(self, start: str, *, without: str | None = None) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            name = stack.pop()
            if name in seen or name == without or name not in self.nodes:
                continue
            seen.add(name)
            stack.extend(self.nodes[name].successors())
        return seen

    def structural_hash(self) -> str:
        shape = {
            name: {
                "kind": node.kind,
                "edges": node.edges(),
                "runtime": str(
                    node.inherits and f"inherit:{node.inherits}" or node.runtime
                ),
            }
            for name, node in self.nodes.items()
        }
        return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()[
            :16
        ]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "entry": self.entry,
            "hash": self.structural_hash(),
            "nodes": {
                name: {
                    "kind": node.kind,
                    "edges": node.edges(),
                    "runtime": str(
                        node.inherits and f"inherit:{node.inherits}" or node.runtime
                    ),
                }
                for name, node in self.nodes.items()
            },
        }


def compile_flow(cls: type[Flow]) -> Graph:
    nodes = {
        name: replace(value, name=name)
        for name, value in vars(cls).items()
        if isinstance(value, Node)
    }
    if not nodes:
        raise FlowError([f"{cls.__name__}: declares no nodes"])
    entry = cls.entry or next(iter(nodes))
    graph = Graph(
        name=cls.__name__, entry=entry, nodes=nodes, config_type=cls.config_type()
    )
    errors = [
        error
        for name, check in CHECKS.items()
        if name not in cls.skip_checks
        for error in check(graph)
    ]
    if errors:
        raise FlowError([f"{cls.__name__}: {error}" for error in errors])
    return graph


Check = Callable[[Graph], list[str]]


def targets_exist(graph: Graph) -> list[str]:
    errors = []
    if graph.entry not in graph.nodes:
        errors.append(f"entry {graph.entry!r} is not a node")
    for node in graph.nodes.values():
        for label, names in node.edges().items():
            for name in names:
                if name not in graph.nodes:
                    errors.append(f"{node.name}.{label} targets unknown node {name!r}")
        if node.then is not None and node.outcomes:
            errors.append(f"{node.name}: declare `then` or `outcomes`, not both")
    return errors


def reachable(graph: Graph) -> list[str]:
    seen = graph.reachable(graph.entry)
    return [
        f"{name} is unreachable from {graph.entry!r}"
        for name in graph.nodes
        if name not in seen
    ]


def bounded_cycles(graph: Graph) -> list[str]:
    """Every cycle must contain an outcome edge and pass through a node that
    declares `max_visits`; nodes without one are bounded by those they cycle through."""
    errors = []
    for component in _cyclic_sccs(graph, graph.nodes):
        outcome_edge = any(
            label.startswith(("outcome:", "on_"))
            for n in component
            for label, names in graph.nodes[n].edges().items()
            if set(names) & component
        )
        if not outcome_edge:
            errors.append(f"cycle {sorted(component)} has no outcome edge")
    unbounded = {
        name: node for name, node in graph.nodes.items() if node.max_visits is None
    }
    for component in _cyclic_sccs(graph, unbounded):
        errors.append(
            f"cycle {sorted(component)} passes through no node with max_visits"
        )
    return errors


def runtime_refs(graph: Graph) -> list[str]:
    """A run node needs a runtime to provision from; an inherited runtime must come
    from an agent or run node that lies on every path to the inheriting node."""
    errors = []
    for node in graph.nodes.values():
        source = node.inherits
        if source is None:
            if isinstance(node, RunNode) and isinstance(node.runtime, str):
                errors.append(
                    f"{node.name}: a run node needs a runtime config or inherit:<node>"
                )
            continue
        if source not in graph.nodes:
            errors.append(f"{node.name}: runtime {node.runtime!r} names unknown node")
            continue
        if node.name not in graph.reachable(
            graph.entry
        ) or node.name in graph.reachable(graph.entry, without=source):
            errors.append(
                f"{node.name}: {node.runtime!r} but {source!r} is not on every path to it"
            )
        if graph.nodes[source].kind not in ("agent", "run"):
            errors.append(
                f"{node.name}: cannot inherit the runtime of a {graph.nodes[source].kind} node"
            )
    return errors


def expand_shape(graph: Graph) -> list[str]:
    return [
        f"{node.name}: expand routes with `then`, not `outcomes`"
        for node in graph.nodes.values()
        if isinstance(node, ExpandNode) and node.outcomes
    ]


def fn_outcomes(graph: Graph) -> list[str]:
    errors = []
    for node in graph.nodes.values():
        if not isinstance(node, FnNode) or not node.outcomes:
            continue
        hint = typing.get_type_hints(node.func).get("return")
        if typing.get_origin(hint) is not typing.Literal:
            continue
        declared = set(node.outcomes) - {"*"}
        literal = set(typing.get_args(hint))
        if extra := declared - literal:
            errors.append(
                f"{node.name}: outcomes {sorted(extra)} not in the return Literal"
            )
        if "*" not in node.outcomes and (missing := literal - declared):
            errors.append(
                f"{node.name}: return values {sorted(missing)} have no outcome edge"
            )
    return errors


def seats_exist(graph: Graph) -> list[str]:
    fields = graph.config_type.model_fields
    errors = []
    for node in graph.nodes.values():
        seat = getattr(node, "seat", None)
        if seat is None:
            continue
        if seat not in fields or fields[seat].annotation is not AgentConfig:
            errors.append(
                f"{node.name}: seat {seat!r} is not an AgentConfig field on {graph.config_type.__name__}"
            )
    return errors


def join_arity(graph: Graph) -> list[str]:
    preds = graph.preds
    return [
        f"{node.name}: join at_least({node.join.k}) but only {len(preds[node.name])} predecessors"
        for node in graph.nodes.values()
        if node.join.kind == "at_least"
        and not isinstance(node, ExpandNode)
        and isinstance(node.join.k, int)
        and node.join.k > len(preds[node.name])
    ]


CHECKS: dict[str, Check] = {
    "targets_exist": targets_exist,
    "reachable": reachable,
    "bounded_cycles": bounded_cycles,
    "runtime_refs": runtime_refs,
    "expand_shape": expand_shape,
    "fn_outcomes": fn_outcomes,
    "seats_exist": seats_exist,
    "join_arity": join_arity,
}


def _cyclic_sccs(graph: Graph, nodes: dict[str, Node]) -> list[set[str]]:
    """The strongly connected components of the subgraph induced by `nodes` that
    contain a cycle (more than one node, or a self-loop)."""
    return [
        component
        for component in _sccs(graph, nodes)
        if len(component) > 1
        or any(n in graph.nodes[n].successors() for n in component)
    ]


def _sccs(graph: Graph, nodes: dict[str, Node]) -> list[set[str]]:
    """Tarjan's strongly connected components over the subgraph induced by `nodes`."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[set[str]] = []
    counter = 0

    def visit(v: str) -> None:
        nonlocal counter
        index[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.nodes[v].successors():
            if w not in nodes:
                continue
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.add(w)
                if w == v:
                    break
            out.append(component)

    for name in nodes:
        if name not in index:
            visit(name)
    return out
