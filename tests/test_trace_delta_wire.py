"""Wire delta-encoding for `Trace.to_wire()` (see `verifiers/v1/trace.py`).

Consecutive turns in a branch restate the whole conversation, so `prompt_ids` (and
cumulative multimodal `mm_items`) are stored as per-turn deltas on the wire and restored
by `Trace`'s `mode="before"` validator. These tests pin the round-trip and the size win.
"""

import copy

import verifiers.v1 as vf


def _turn(prompt_ids, completion_ids, mm=None):
    return vf.Turn(
        prompt=[vf.UserMessage(content="x")],
        response=vf.Response(
            id="r",
            created=0,
            model="m",
            message=vf.AssistantMessage(content="y"),
            finish_reason="stop",
        ),
        tokens=vf.TurnTokens(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=[0.0] * len(completion_ids),
            multi_modal_data=mm,
        ),
    )


def _trace(turns):
    return vf.Trace[vf.Task](task=vf.Task(idx=0, instruction="go"), trajectory=turns)


def _linear_turns(n_turns):
    """A linear conversation: each turn's prompt is the prior (prompt+completion) plus
    new context tokens — i.e. the cumulative form the renderer emits."""
    full, turns = [], []
    for t in range(n_turns):
        full = full + list(range(t * 10, t * 10 + 10))
        completion = list(range(900 + t * 5, 900 + t * 5 + 5))
        turns.append(_turn(list(full), completion))
        full = full + completion
    return turns


def test_prompt_ids_delta_is_smaller_and_round_trips():
    turns = _linear_turns(4)
    wire = _trace(turns).to_wire()

    full = sum(len(t.tokens.prompt_ids) for t in turns)
    delta = sum(len(t["tokens"]["prompt_ids"]) for t in wire["trajectory"])
    assert delta < full  # quadratic full -> linear delta
    assert [t["tokens"]["pk"] for t in wire["trajectory"]] == [0, 15, 30, 45]

    back = vf.Trace[vf.Task].model_validate(copy.deepcopy(wire))
    for b, o in zip(back.trajectory, turns):
        assert list(b.tokens.prompt_ids) == list(o.tokens.prompt_ids)
        assert list(b.tokens.completion_ids) == list(o.tokens.completion_ids)
    # markers are wire-only — they must not survive validation onto the strict model
    assert all(
        "pk" not in b.tokens.model_dump() and "mk" not in b.tokens.model_dump()
        for b in back.trajectory
    )


def test_fork_turn_stores_full_prompt():
    """A turn whose prompt doesn't extend the running sequence (a branch/fork) has no
    shared prefix, so it stores the full tail (pk == 0) and still round-trips."""
    turns = _linear_turns(2) + [_turn([7, 7, 7], [1, 1])]
    wire = _trace(turns).to_wire()
    assert wire["trajectory"][-1]["tokens"]["pk"] == 0

    back = vf.Trace[vf.Task].model_validate(copy.deepcopy(wire))
    assert [list(t.tokens.prompt_ids) for t in back.trajectory] == [
        list(t.tokens.prompt_ids) for t in turns
    ]


def test_multimodal_items_delta_round_trips_with_repeats():
    """Cumulative mm_items collapse to the new image(s) per turn; a repeated image (same
    hash, here red appearing twice) keeps a distinct slot, matched by position."""

    def wt():
        return vf.WireTensor(dtype="float32", shape=[1], data="AA==")

    def mm(hashes):
        return vf.MMData(
            mm_items={"image": [{"pixel_values": wt()} for _ in hashes]},
            mm_hashes={"image": list(hashes)},
        )

    turns = [
        _turn([0, 1], [9], mm(["hR"])),
        _turn([0, 1, 9, 2, 3], [8], mm(["hR", "hG"])),
        _turn([0, 1, 9, 2, 3, 8, 4, 5], [7], mm(["hR", "hG", "hR"])),
    ]
    wire = _trace(turns).to_wire()
    assert [t["tokens"]["mk"]["image"] for t in wire["trajectory"]] == [0, 1, 2]
    assert [
        t["tokens"]["multi_modal_data"]["mm_hashes"]["image"]
        for t in wire["trajectory"]
    ] == [
        ["hR"],
        ["hG"],
        ["hR"],
    ]

    back = vf.Trace[vf.Task].model_validate(copy.deepcopy(wire))
    for b, o in zip(back.trajectory, turns):
        assert (
            b.tokens.multi_modal_data.mm_hashes["image"]
            == o.tokens.multi_modal_data.mm_hashes["image"]
        )
        assert len(b.tokens.multi_modal_data.mm_items["image"]) == len(
            o.tokens.multi_modal_data.mm_items["image"]
        )


def test_decode_without_markers_is_noop():
    """Expanding a trace dict that carries no delta markers (full prompt_ids, e.g. a
    non-wire input) leaves it untouched — so validation never corrupts plain traces."""
    from verifiers.v1.trace import _delta_decode_turns

    turns = [
        {"tokens": {"prompt_ids": [1, 2, 3], "completion_ids": [4]}},
        {"tokens": {"prompt_ids": [1, 2, 3, 4, 5], "completion_ids": [6]}},
    ]
    snapshot = copy.deepcopy(turns)
    _delta_decode_turns(turns)
    assert turns == snapshot
