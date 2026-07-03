"""Unit tests for the replay taskset base's pure logic: message-graph surgery and the
buffer's scan/pick/read plumbing. Derivation packages (which bind anchors + prompts)
are tested where they live."""

import json
from pathlib import Path

import pytest
from verifiers.v1.tasksets.replay import (
    ReplayBuffer,
    build_children,
    compaction_forks,
    continue_seed,
    final_leaf,
    main_tree,
    recheck_seed,
    tool_call_anchors,
    unwrap_source_task,
    usable,
)

# ------------------------------------------------------------------ surgery


def _node(parent: int | None, sampled: bool, message: dict) -> dict:
    return {"parent": parent, "sampled": sampled, "message": message}


@pytest.fixture
def forest() -> list[dict]:
    """A synthetic message forest exercising every structure surgery must tell apart.

    Main tree (root 0): a compaction fork off the system root (node 6), a retried-assistant
    twin fork (nodes 2/3), a duplicated tool-result fork (nodes 4/5), and a final assistant
    truncated mid-tool-call (node 9, empty content). Subagent tree (root 10): its own
    compaction-shaped fork (node 12) and the forest's highest leaf index (node 13).
    """
    return [
        _node(None, False, {"role": "system", "content": "You are an agent."}),  # 0
        _node(0, False, {"role": "user", "content": "Original task: sort the list."}),  # 1
        _node(1, True, {"role": "assistant", "content": "first attempt"}),  # 2
        _node(1, True, {"role": "assistant", "content": "retried attempt"}),  # 3: retry twin fork
        _node(3, False, {"role": "tool", "content": "tool output"}),  # 4
        _node(3, False, {"role": "tool", "content": "tool output"}),  # 5: duplicated tool result
        _node(0, False, {"role": "user", "content": "Summary: the agent was sorting a list."}),  # 6: compaction
        _node(
            6,
            True,
            {
                "role": "assistant",
                "content": "resuming work",
                "tool_calls": [{"id": "c1", "name": "search", "arguments": '{"q": "sort"}'}],
            },
        ),  # 7
        _node(7, False, {"role": "tool", "tool_call_id": "c1", "content": "x" * 300}),  # 8
        _node(
            8,
            True,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c2", "name": "search", "arguments": '{"q": "again"}'}],
            },
        ),  # 9: final main-tree leaf, truncated mid-tool-call
        _node(None, False, {"role": "user", "content": "subagent task"}),  # 10: subagent root
        _node(10, True, {"role": "assistant", "content": "subagent answer"}),  # 11
        _node(10, False, {"role": "user", "content": "subagent summary"}),  # 12: compaction-shaped, outside main
        _node(12, True, {"role": "assistant", "content": "subagent resumed"}),  # 13: global max leaf
    ]


def test_main_tree_excludes_subagent_roots(forest):
    children, roots = build_children(forest)
    assert roots == [0, 10]
    assert main_tree(children) == set(range(10))


def test_compaction_forks_finds_only_the_compaction_child(forest):
    children, _ = build_children(forest)
    tree = main_tree(children)
    # Not node 3 (assistant retry), not node 5 (duplicated tool result), not node 12
    # (compaction-shaped but in the subagent tree).
    assert compaction_forks(forest, children, tree) == [6]


def test_final_leaf_stays_in_the_main_tree(forest):
    children, _ = build_children(forest)
    tree = main_tree(children)
    assert final_leaf(children, tree) == 9  # not 13, the subagent's higher-index leaf


def test_continue_seed_is_root_to_anchor(forest):
    assert continue_seed(forest, 6) == [forest[0]["message"], forest[6]["message"]]
    # A tool-call anchor seeds down to (and including) the tool result.
    assert [m["role"] for m in continue_seed(forest, 8)] == ["system", "user", "assistant", "tool"]


def test_tool_call_anchors_require_resolved_calls():
    call = {"name": "t", "arguments": "{}"}
    nodes = [
        _node(None, False, {"role": "system", "content": "s"}),  # 0
        _node(0, False, {"role": "user", "content": "task"}),  # 1
        _node(1, True, {"role": "assistant", "content": "", "tool_calls": [call | {"id": "a"}, call | {"id": "b"}]}),  # 2
        _node(2, False, {"role": "tool", "tool_call_id": "a", "content": "ra"}),  # 3: "b" still pending
        _node(3, False, {"role": "tool", "tool_call_id": "b", "content": "rb"}),  # 4: all resolved -> anchor
        _node(4, True, {"role": "assistant", "content": "", "tool_calls": [call | {"id": "c"}]}),  # 5
        _node(5, False, {"role": "tool", "tool_call_id": "c", "content": "rc"}),  # 6: anchor
        _node(None, False, {"role": "user", "content": "subagent"}),  # 7: subagent root
        _node(7, True, {"role": "assistant", "content": "", "tool_calls": [call | {"id": "z"}]}),  # 8
        _node(8, False, {"role": "tool", "tool_call_id": "z", "content": "rz"}),  # 9: outside main tree
    ]
    children, _ = build_children(nodes)
    tree = main_tree(children)
    assert tool_call_anchors(nodes, children, tree) == [4, 6]
    seed = continue_seed(nodes, 6)
    assert [m["role"] for m in seed] == ["system", "user", "assistant", "tool", "tool", "assistant", "tool"]


def test_recheck_seed_drops_truncated_assistant_and_appends_instruction(forest):
    children, _ = build_children(forest)
    tree = main_tree(children)
    messages = recheck_seed(forest, children, tree, "Check your work.")
    # Final branch is [0, 6, 7, 8, 9]; node 9 (empty content, pending tool_calls) is dropped.
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
    assert messages[-1] == {"role": "user", "content": "Check your work."}
    assert messages[2]["tool_calls"]  # only the trailing assistant is stripped
    assert forest[9]["message"]["tool_calls"]  # the saved nodes are not mutated


def test_recheck_seed_strips_tool_calls_but_keeps_nonempty_assistant():
    nodes = [
        _node(None, False, {"role": "system", "content": "sys"}),
        _node(
            0,
            True,
            {
                "role": "assistant",
                "content": "final answer",
                "tool_calls": [{"id": "c1", "name": "search", "arguments": "{}"}],
            },
        ),
    ]
    children, _ = build_children(nodes)
    messages = recheck_seed(nodes, children, main_tree(children), "Check your work.")
    assert [m["role"] for m in messages] == ["system", "assistant", "user"]
    assert messages[1]["content"] == "final answer"
    assert messages[1]["tool_calls"] is None
    assert nodes[1]["message"]["tool_calls"]  # original untouched


def test_usable_screens_bad_records(forest):
    assert usable({"nodes": forest, "errors": None})
    assert not usable({"nodes": forest, "errors": ["timeout"]})
    assert not usable({"nodes": [], "errors": None})
    unsampled = [_node(None, False, {"role": "system", "content": "sys"})]
    assert not usable({"nodes": unsampled, "errors": None})


def test_unwrap_source_task_resolves_chains():
    original = {"prompt": "sort the list", "image": "sandbox:1"}
    depth_2 = {
        "source_id": "b",
        "source_task": {"source_id": "a", "source_task": original},
    }
    assert unwrap_source_task(depth_2) == original
    assert unwrap_source_task(original) == original  # non-derived tasks pass through


# ------------------------------------------------------------------ buffer


def _record(record_id: str, reward: float) -> dict:
    return {
        "id": record_id,
        "nodes": [
            _node(None, False, {"role": "system", "content": "sys"}),
            _node(0, False, {"role": "user", "content": f"task {record_id}"}),
            _node(1, True, {"role": "assistant", "content": f"answer {record_id}"}),
        ],
        "errors": None,
        "rewards": {"correct": reward},
        "stop_condition": "agent_completed",
        "task": {},
        "info": {},
    }


@pytest.fixture
def buffer_dir(tmp_path: Path) -> tuple[Path, dict[str, dict]]:
    """Two step dirs: step_1 is barrier-complete (train_rollouts.bin present), step_2 is
    not. step_2 also carries an errored record that must never become a candidate."""
    records = {
        "aaa": _record("aaa", 1.0),
        "bbb": _record("bbb", 1.0),
        "ccc": _record("ccc", 0.0),
        "ddd": _record("ddd", 1.0),
    }
    errored = _record("eee", 0.0) | {"errors": ["timeout"]}
    step_1 = tmp_path / "step_1"
    step_1.mkdir()
    (step_1 / "train_rollouts.jsonl").write_text("".join(json.dumps(records[i]) + "\n" for i in ("aaa", "bbb", "ccc")))
    (step_1 / "train_rollouts.bin").touch()
    step_2 = tmp_path / "step_2"
    step_2.mkdir()
    (step_2 / "train_rollouts.jsonl").write_text(json.dumps(records["ddd"]) + "\n" + json.dumps(errored) + "\n")
    return tmp_path, records


def _make_buffer(path: Path, online: bool = False, **overrides) -> ReplayBuffer:
    kwargs = dict(
        buffer_dir=str(path),
        anchors=lambda record: [None],  # one task per rollout, like recheck
        online=online,
        source_envs=None,
        allow_container=False,
    )
    kwargs.update(overrides)
    return ReplayBuffer(**kwargs)


def test_online_scan_requires_barrier(buffer_dir):
    path, _ = buffer_dir
    candidates = _make_buffer(path, online=True).scan()
    assert {c.step for c in candidates} == {1}  # step_2 has no train_rollouts.bin yet


def test_offline_scan_skips_barrier_and_errored_records(buffer_dir):
    path, _ = buffer_dir
    candidates = _make_buffer(path).scan()
    assert [c.source_id for c in candidates] == ["ddd", "aaa", "bbb", "ccc"]  # newest step first, no "eee"


def test_rescan_accumulates_new_steps(buffer_dir):
    path, _ = buffer_dir
    buffer = _make_buffer(path, online=True)
    assert [c.source_id for c in buffer.scan()] == ["aaa", "bbb", "ccc"]
    step_3 = path / "step_3"
    step_3.mkdir()
    step_3.joinpath("train_rollouts.jsonl").write_text(json.dumps(_record("fff", 0.0)) + "\n")
    step_3.joinpath("train_rollouts.bin").touch()
    assert [c.source_id for c in buffer.scan()] == ["fff", "aaa", "bbb", "ccc"]  # newest first


def test_anchor_enumerator_mints_one_candidate_per_anchor(buffer_dir):
    path, _ = buffer_dir
    buffer = _make_buffer(path, anchors=lambda record: [1, 2])
    assert [(c.source_id, c.anchor_node) for c in buffer.scan() if c.step == 1] == [
        ("aaa", 1),
        ("aaa", 2),
        ("bbb", 1),
        ("bbb", 2),
        ("ccc", 1),
        ("ccc", 2),
    ]
    assert _make_buffer(path, anchors=lambda record: []).scan() == []


def test_pick_is_deterministic_and_wraps(buffer_dir):
    path, _ = buffer_dir
    buffer = _make_buffer(path)
    buffer.scan()
    assert buffer.pick(2) == buffer.pick(2)
    assert buffer.pick(2) == buffer.pick(2 + len(buffer))


def test_read_record_round_trips(buffer_dir):
    path, records = buffer_dir
    buffer = _make_buffer(path)
    for candidate in buffer.scan():
        assert buffer.read_record(candidate) == records[candidate.source_id]


def test_replay_derived_records_skipped_by_default_but_listable(buffer_dir):
    """A "self" buffer sees the replay envs' own saved rollouts. By default replaying a
    replay is a feedback loop and is screened out; explicitly listing the replay env in
    source_envs is the deliberate opt-in for chained derivations."""
    path, _ = buffer_dir
    nested = _record("zzz", 1.0)
    nested["task"] = {"source_id": "yyy", "source_task": {"prompt": "original"}, "prompt": None}
    nested["info"] = {"prime_rl": {"env_name": "replay-recheck"}}
    step_1 = path / "step_1"
    with open(step_1 / "train_rollouts.jsonl", "a") as f:
        f.write(json.dumps(nested) + "\n")
    by_default = _make_buffer(path).scan()
    assert "zzz" not in {c.source_id for c in by_default}
    opted_in = _make_buffer(path, source_envs=["replay-recheck"]).scan()
    assert {c.source_id for c in opted_in} == {"zzz"}
