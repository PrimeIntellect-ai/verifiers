import random

import numpy as np

from verifiers.v1.graph import MessageNode
from verifiers.v1.serve.delta import dump, pack, unpack
from verifiers.v1.trace import Branch
from verifiers.v1.types import AssistantMessage, UserMessage


def assistant(**fields) -> MessageNode:
    return MessageNode(message=AssistantMessage(content="a"), sampled=True, **fields)


def test_validated_streams_are_typed_arrays():
    node = assistant(
        token_ids=[1, 2, 3],
        mask=[False, True, True],
        is_content=[False, True, True],
        logprobs=[-0.1, -0.2],
        advantages=[0.5, 0.5],
        loss_weights={"rl": [0.0, 1.0, 1.0]},
    )
    dtypes = {
        "token_ids": np.int32,
        "mask": np.bool_,
        "is_content": np.bool_,
        "logprobs": np.float32,
        "advantages": np.float32,
    }
    for name, dtype in dtypes.items():
        value = getattr(node, name)
        assert isinstance(value, np.ndarray) and value.dtype == dtype, name
    assert node.loss_weights["rl"].dtype == np.float32
    # Defaults are typed arrays too.
    bare = assistant()
    assert bare.token_ids.dtype == np.int32 and bare.logprobs.dtype == np.float32


def test_token_arrays_wire_round_trip():
    node = assistant(
        token_ids=[1, 2, 130_000],
        mask=[False, True, True],
        logprobs=[-0.125, -0.25],
        loss_weights={"rl": [0.0, 1.0, 1.0]},
    )
    wire = unpack(pack(dump(node, context={"token_arrays": True})))
    assert wire["token_ids"]["__nd__"] and wire["loss_weights"]["rl"]["__nd__"]
    back = MessageNode.model_validate(wire)
    assert back.token_ids.tolist() == [1, 2, 130_000]
    assert back.mask.tolist() == [False, True, True]
    assert back.logprobs.tolist() == [-0.125, -0.25]
    assert back.loss_weights["rl"].tolist() == [0.0, 1.0, 1.0]
    # Without the context the wire keeps lists, which validate the same way.
    plain = unpack(pack(dump(node)))
    assert plain["token_ids"] == [1, 2, 130_000]
    assert MessageNode.model_validate(plain).token_ids.dtype == np.int32


def test_json_records_stay_lists():
    node = assistant(token_ids=[1, 2], mask=[False, True], logprobs=[-0.123456])
    record = node.model_dump(mode="json")
    assert record["token_ids"] == [1, 2] and record["mask"] == [False, True]
    assert record["logprobs"] == [-0.1235]
    assert MessageNode.model_validate(record).token_ids.dtype == np.int32


def spread_lists(nodes: list[MessageNode], values) -> list[float]:
    """`Branch.spread` as it was on lists, the reference the array version must match."""
    out: list[float] = []
    for node in nodes:
        mask, node_values = list(node.mask), values(node)
        node_values = [] if node_values is None else [float(v) for v in node_values]
        sampled = sum(mask) if node_values else 0
        if not sampled or all(mask[-sampled:]):
            out += [0.0] * (len(mask) - sampled) + node_values[:sampled]
            out += [0.0] * max(0, sampled - len(node_values))
            continue
        li = 0
        for is_sampled in mask:
            if is_sampled:
                out.append(node_values[li] if li < len(node_values) else 0.0)
                li += 1
            else:
                out.append(0.0)
    return out


def test_spread_matches_the_list_implementation():
    rng = random.Random(0)
    for _ in range(300):
        nodes = []
        for _ in range(rng.randint(0, 6)):
            n = rng.randint(0, 12)
            if rng.random() < 0.5:  # canonical: scaffold then sampled suffix
                k = rng.randint(0, n)
                mask = [False] * (n - k) + [True] * k
            else:
                mask = [rng.random() < 0.5 for _ in range(n)]
            values = [rng.uniform(-5, 0) for _ in range(rng.randint(0, n + 2))]
            node_cls = (
                assistant
                if rng.random() < 0.5
                else (lambda **f: MessageNode(message=UserMessage(content="u"), **f))
            )
            node = node_cls(token_ids=list(range(n)), mask=mask, logprobs=values)
            if rng.random() < 0.5:  # env-server nodes hold lists
                node = MessageNode.model_construct(
                    **{**node.__dict__, "mask": mask, "logprobs": values}
                )
            nodes.append(node)
        branch = Branch(index=0, nodes=nodes)
        expected = np.asarray(
            spread_lists(nodes, lambda node: node.logprobs), np.float32
        )
        got = branch.logprobs
        assert got.dtype == np.float32 and np.array_equal(got, expected)
        assert branch.token_ids.dtype == np.int32
        assert len(branch.token_ids) == len(branch.sampled_mask) == len(got)


def test_committed_nodes_own_typed_arrays():
    """The env server builds nodes unvalidated: each still holds typed arrays, owns them (a view
    would retain the turn's full-context prompt) and keeps no copy of identical renderer ids."""
    import verifiers.v1 as vf
    from verifiers.v1 import graph
    from verifiers.v1.types import Response, TurnTokens

    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    graph.prepare_turn(trace, [UserMessage(content="u")]).commit(
        Response(
            id="a",
            created=0,
            model="t",
            message=AssistantMessage(content="a"),
            finish_reason="stop",
            tokens=TurnTokens(
                prompt_ids=np.arange(10, 14, dtype=np.int32),
                completion_ids=np.asarray([20, 21], dtype=np.int32),
                completion_logprobs=np.asarray([-0.5, -0.25], dtype=np.float32),
                message_spans=[(0, 3)],
                is_content=np.asarray([False, True, True, False]),
            ),
        )
    )
    user, reply = trace.nodes
    assert user.token_ids.tolist() == [10, 11, 12]
    assert reply.token_ids.tolist() == [13, 20, 21]
    assert reply.mask.tolist() == [False, True, True]
    assert reply.logprobs.tolist() == [-0.5, -0.25]
    for node in (user, reply):
        assert node.token_ids.dtype == np.int32 and node.token_ids.base is None
        assert node.mask.dtype == node.is_content.dtype == np.bool_
        assert node.logprobs.dtype == np.float32
        assert node.renderer_token_ids is None
