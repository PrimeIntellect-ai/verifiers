from types import SimpleNamespace

from verifiers.v1.prefix import Prefix, PrefixCall, PrefixReplay


def test_take_replays_repeated_completions():
    """Recorded calls with identical completions are each served once, in order."""
    completions = [[1, 2], [5], [5], [7]]
    prefix = Prefix(
        calls=[
            PrefixCall(
                completion_ids=c, observation_hash="", parent=i - 1 if i else None
            )
            for i, c in enumerate(completions)
        ]
    )
    replay = PrefixReplay(prefix)
    nodes: list[SimpleNamespace] = []
    served = []
    for step in range(len(completions) + 1):
        nodes.append(SimpleNamespace(token_ids=[100 + step], sampled=False, replayed=0))
        turn = SimpleNamespace(
            trace=SimpleNamespace(nodes=nodes), prefix_node_ids=list(range(len(nodes)))
        )
        prompt_ids = [t for node in nodes for t in node.token_ids]
        call = replay.take(prompt_ids, turn)
        if call is None:
            break
        served.append(prefix.calls.index(call))
        nodes.append(
            SimpleNamespace(
                token_ids=call.completion_ids,
                sampled=True,
                replayed=len(call.completion_ids),
            )
        )
    assert served == [0, 1, 2, 3]
