"""Topology tests: config narrowing, agent loading, and instance semantics (stubbed rollouts).

The unit tests stub `TopologyRunner.rollout` with a canned-reply rollout, so the whole
imperative surface — forward trace→task arrows, fan-out, parent links, declared instance
judgement, `go` failure capture — runs without a model or a runtime. The `e2e`-marked test
runs the `echo-chain-v1` fixture topology live (skipped without a key), mirroring `test_e2e`.
"""

import json

import pytest
import verifiers.v1 as vf
from verifiers.v1.cli.output import output_path
from verifiers.v1.cli.resolve import narrow_config
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.topologies.agentic_judge import TRACE_PATH, AgenticJudgeTask
from verifiers.v1.topologies.llm_judge import JudgeTask
from verifiers.v1.topology import AgentGraph, TopologyRunner
from verifiers.v1.trace import Trace, WireTrace
from verifiers.v1.types import AssistantMessage, UserMessage


def echoing_trace(task: vf.Task, reply: str) -> Trace:
    """A minimal completed trace: one user turn, one sampled assistant reply."""
    trace = Trace(task=task)
    trace.nodes.append(
        MessageNode(parent=None, message=UserMessage(content=str(task.prompt)))
    )
    trace.nodes.append(
        MessageNode(parent=0, message=AssistantMessage(content=reply), sampled=True)
    )
    trace.stop("agent_completed")
    return trace


class EchoingRollout:
    """Stands in for `Rollout`: replies with the echo task's own answer and scores like the
    echo task would, so a topology's `go` runs its real logic."""

    def __init__(self, task: vf.Task) -> None:
        self.task = task

    async def run(self) -> Trace:
        trace = echoing_trace(self.task, getattr(self.task, "answer", "ok"))
        trace.record_reward("echoed", 1.0)
        return trace


@pytest.fixture
def echo_chain_env(monkeypatch) -> TopologyRunner:
    """A loaded echo-chain topology whose episodes run against `EchoingRollout` (no model,
    no runtime, no interception)."""
    config = EvalConfig(topology={"id": "echo-chain-v1"}, rich=False)
    env = TopologyRunner(config.topology, config)
    monkeypatch.setattr(
        TopologyRunner, "rollout", lambda self, agent, task: EchoingRollout(task)
    )
    return env


def test_llm_judge_config_narrows_typed_agents():
    """`--topology.id llm-judge` narrows to `LLMJudgeConfig`; the seed factory narrows by
    id, and the judge keeps its pinned defaults (direct harness, non-trainable) unless
    overridden."""
    config = EvalConfig(
        topology={"id": "llm-judge", "taskset": {"id": "echo-v1"}}, rich=False
    )
    assert type(config.topology).__name__ == "LLMJudgeConfig"
    assert type(config.topology.taskset).__name__ == "EchoConfig"
    assert config.topology.solver.harness.id == "default"
    assert config.topology.judge.harness.id == "direct"
    assert config.topology.judge.trainable is False
    assert config.topology.solver.trainable is True


def test_pinned_harness_survives_sibling_and_partial_overrides():
    """The pin contract, all three previously-broken paths: (a) a sibling override
    (`--topology.solver.model X`) must not silently replace a pinned harness with the
    default agent; (b) an id-less partial harness override must deep-merge into the pin,
    not reset it; (c) a base-typed `HarnessConfig(id=...)` pin is detected by value."""
    import echo_chain_v1
    from proposer_solver_v1.topology import ProposerSolverConfig

    # (a) sibling field override keeps the subclass pin
    config = ProposerSolverConfig.model_validate({"solver": {"model": "org/strong"}})
    assert config.solver.model == "org/strong"
    assert config.solver.harness.id == "direct"
    # (b) partial harness override tunes the pin (llm-judge judge: direct + docker runtime)
    eval_config = EvalConfig(
        topology={
            "id": "llm-judge",
            "taskset": {"id": "echo-v1"},
            "judge": {"harness": {"runtime": {"type": "docker"}}},
        },
        rich=False,
    )
    judge = eval_config.topology.judge
    assert judge.harness.id == "direct"
    assert judge.harness.runtime.type == "docker"
    # (c) a base-typed pin (null) is a value pin — bare construction keeps it
    assert echo_chain_v1.NullAgentConfig().harness.id == "null"
    assert (
        echo_chain_v1.NullAgentConfig.model_validate({"model": "org/x"}).harness.id
        == "null"
    )


def test_pinned_agent_harness_still_swaps_by_id():
    """A subclass pins per-agent defaults via the field default *instance*, not the
    annotation — so an explicit `--topology.judge.harness.id <id>` (on `agentic-judge`,
    whose judge harness is configurable) narrows to the swapped harness's own config type
    while the other pins survive the swap."""
    config = EvalConfig(
        topology={
            "id": "agentic-judge",
            "taskset": {"id": "echo-v1"},
            "judge": {"harness": {"id": "null"}},
        },
        rich=False,
    )
    assert type(config.topology.judge.harness).__name__ == "NullHarnessConfig"
    assert config.topology.judge.trainable is False  # other pins survive the swap


def test_llm_judge_harness_is_locked():
    """`llm-judge` fixes its judge to the in-process `direct` chat loop: an explicit
    harness swap is refused with a pointer at `agentic-judge` (whose judge harness IS
    configurable); the judge's routing stays per-agent config."""
    with pytest.raises(ValueError, match="agentic-judge"):
        EvalConfig(
            topology={
                "id": "llm-judge",
                "taskset": {"id": "echo-v1"},
                "judge": {"harness": {"id": "default"}},
            },
            rich=False,
        )
    config = EvalConfig(  # routing overrides don't touch the lock
        topology={
            "id": "llm-judge",
            "taskset": {"id": "echo-v1"},
            "judge": {"model": "org/strong-judge"},
        },
        rich=False,
    )
    assert config.topology.judge.harness.id == "direct"
    assert config.topology.judge.model == "org/strong-judge"


def test_agent_discovery_and_seed_factory():
    config = EvalConfig(topology={"id": "echo-chain-v1"}, rich=False)
    topology = vf.load_topology(config.topology)
    assert list(topology.agents) == ["first", "second"]  # declaration order
    tasks = topology.load_tasks()  # from the pinned echo-v1 seed factory
    assert tasks and tasks[0].answer == "hello world"


def test_topology_without_seeds_is_refused():
    config = EvalConfig(topology={"id": "llm-judge"}, rich=False)
    with pytest.raises(ValueError, match="has no seed tasks"):
        vf.load_topology(config.topology).load_tasks()


def test_seed_slot_is_exclusive_with_load_tasks_override():
    """The seed contract is XOR: proposer-solver overrides `load_tasks` (self-seeding), so
    passing `--topology.taskset.id` — which would be silently ignored — is refused at load."""
    config = EvalConfig(
        topology={"id": "proposer-solver-v1", "taskset": {"id": "echo-v1"}}, rich=False
    )
    with pytest.raises(ValueError, match="constructs its own seeds"):
        TopologyRunner(config.topology, config)


def test_unknown_agent_is_refused(echo_chain_env):
    with pytest.raises(ValueError, match="unknown agent 'thrid'"):
        echo_chain_env.agent("thrid")


def test_topology_reward_scopes_validated_at_load():
    """Declared judgement fails loudly when the topology loads: a typo'd agent scope and
    a missing scope are both refused before anything runs."""
    import echo_chain_v1

    class TypoScope(echo_chain_v1.EchoChainTopology):
        @vf.reward(agent="thrid")
        async def oops(self, trace):
            return 0.0

    with pytest.raises(ValueError, match="unknown agent 'thrid'"):
        _ = TypoScope(echo_chain_v1.EchoChainConfig()).agents

    class NoScope(echo_chain_v1.EchoChainTopology):
        @vf.metric
        async def oops(self, trace):
            return 0.0

    with pytest.raises(ValueError, match="declares no agent scope"):
        _ = NoScope(echo_chain_v1.EchoChainConfig()).agents


async def test_run_instance_links_and_defers(echo_chain_env):
    """One stubbed instance: completion-ordered traces, parent links, agent names, and the
    declared backward-arrow reward landing on the upstream trace at instance end."""
    seed = echo_chain_env.topology.load_tasks()[0]
    graph = await echo_chain_env.run_instance(seed)
    assert graph.error is None
    assert graph.topology == "echo-chain-v1"
    first, second = graph.traces
    assert (first.agent, second.agent) == ("first", "second")
    assert first.parents == [] and second.parents == [first.id]
    assert graph.roots() == [first]
    assert graph.children(first) == [second]
    assert graph.by_agent("second") == [second]
    assert first.trainable and second.trainable
    # forward arrow: the derived task echoes the same phrase
    assert second.task.answer == seed.answer
    # backward arrow: relay reward recorded on the upstream trace after the child finished
    assert first.rewards == {"echoed": 1.0, "relay": 1.0}
    assert second.rewards == {"echoed": 1.0}


async def test_explicit_agents_record_into_topology_graph(monkeypatch):
    """The new `run.agent(name).run(task)` surface still records the same graph links and
    trainability stamps as the compatibility `run.rollout(name, task)` wrapper."""
    import echo_chain_v1

    config = EvalConfig(topology={"id": "echo-chain-v1"}, rich=False)
    env = TopologyRunner(config.topology, config)

    async def explicit_go(self, task, run):
        first = await run.agent("first").run(task)
        derived = type(task)(
            idx=task.idx,
            prompt=f"echo {task.answer}",
            answer=task.answer,
        )
        await run.agent("second").run(derived, parents=[first])

    async def fake_agent_run(
        self, task, *, parents=(), runtime=None, ctx=None, services=None, retry=None
    ):
        trace = echoing_trace(task, getattr(task, "answer", "ok"))
        trace.record_reward("echoed", 1.0)
        self.stamp(
            trace,
            parents=parents,
            runtime=None,
            ctx=ctx or self.ctx,
            borrowed=runtime is not None,
        )
        return trace

    monkeypatch.setattr(echo_chain_v1.EchoChainTopology, "go", explicit_go)
    monkeypatch.setattr(vf.Agent, "run", fake_agent_run)
    seed = env.topology.load_tasks()[0]
    async with env.serving(vf.ModelContext(model="org/model", client=object())):
        graph = await env.run_instance(seed)

    first, second = graph.traces
    assert (first.agent, second.agent) == ("first", "second")
    assert first.parents == [] and second.parents == [first.id]
    assert first.trainable and second.trainable
    assert first.info["agent"]["model"] == "org/model"
    assert graph.children(first, agent="second") == [second]


async def test_declared_judgement_scores_the_instance(monkeypatch):
    """`Topology.score` runs the declared @metric/@reward methods once per matching trace
    after `go` returns — metrics before rewards (`difficulty` reads the `solve_rate`
    metric), and over every trace in scope even when the instance ended early (the stub
    replies with no QUESTION/ANSWER, so `go` stops after the proposer). The proposer's own
    task-declared reward (`well_formed`) is absent here because the stub skips episode
    scoring — task rewards run inside real rollouts."""
    config = EvalConfig(topology={"id": "proposer-solver-v1"}, rich=False)
    env = TopologyRunner(config.topology, config)
    monkeypatch.setattr(
        TopologyRunner, "rollout", lambda self, agent, task: EchoingRollout(task)
    )
    graph = await env.run_instance(env.topology.load_tasks()[0])
    (proposer,) = graph.traces  # malformed proposal → no solver fan-out
    assert proposer.metrics == {"solve_rate": 0.0}
    assert proposer.rewards == {"echoed": 1.0, "difficulty": 0.0}


async def test_go_failure_is_captured_on_graph(echo_chain_env, monkeypatch):
    """A crash in topology-authored code is classified `TopologyError` and recorded on the
    graph — completed episodes stay, nothing raises (a bad instance is data)."""
    cls = type(echo_chain_env.topology)

    async def exploding_go(self, task, run):
        await run.rollout("first", task)
        raise RuntimeError("boom")

    monkeypatch.setattr(cls, "go", exploding_go)
    graph = await echo_chain_env.run_instance(echo_chain_env.topology.load_tasks()[0])
    assert graph.error is not None
    assert graph.error.type == "TopologyError"
    assert "boom" in graph.error.message
    assert len(graph.traces) == 1  # the episode that completed before the crash


def test_eval_config_rejects_topology_with_taskset():
    with pytest.raises(ValueError, match="drop `--taskset.id`"):
        EvalConfig(
            topology={"id": "echo-chain-v1"}, taskset={"id": "echo-v1"}, rich=False
        )


def test_eval_config_rejects_topology_with_server():
    with pytest.raises(ValueError, match="env-server"):
        EvalConfig(topology={"id": "echo-chain-v1"}, server=True, rich=False)


def test_eval_config_rejects_topology_with_harness():
    """A user-supplied `--harness.*` under a topology would be silently ignored (agents
    bind their own) — refused up front, while the framework-manufactured default (absent
    from the input data) never trips the guard."""
    with pytest.raises(ValueError, match="ignored under a topology"):
        EvalConfig(topology={"id": "echo-chain-v1"}, harness={"id": "rlm"}, rich=False)
    EvalConfig(topology={"id": "echo-chain-v1"}, rich=False)  # no harness key: fine


def test_narrow_config_resolves_topology_id():
    narrowed = narrow_config(EvalConfig, ["--topology.id", "llm-judge"])
    assert type(narrowed.model_fields["topology"].default).__name__ == "LLMJudgeConfig"


def test_output_path_names_topology_runs():
    config = EvalConfig(topology={"id": "echo-chain-v1"}, model="org/model", rich=False)
    assert output_path(config).parent.name == "echo-chain-v1--org--model"


def test_judge_task_construction_and_verdict_parsing():
    """`JudgeTask.for_attempt` peels the judge's inputs off the finished episode: the seed
    task's framing, its ground truth (an `answer` field, when the task carries one), and
    the solver's final message (last message of the final branch, not the transcript)."""

    class AnsweredTask(vf.Task):
        answer: str

    task = AnsweredTask(idx=0, prompt="what is 2+2?", answer="4")
    solved = echoing_trace(task, "It is 4.")
    grading = JudgeTask.for_attempt(task, solved)
    assert "what is 2+2?" in grading.prompt and "It is 4." in grading.prompt
    assert "<reference_answer>\n4\n</reference_answer>" in grading.prompt
    # a task without ground truth just omits the reference section
    bare = JudgeTask.for_attempt(vf.Task(idx=0, prompt="write a poem"), solved)
    assert "<reference_answer>" not in bare.prompt
    for reply, expected in [
        ("The attempt is correct.\nSCORE: 10", 1.0),
        ("Partially right.\nscore: 7/10", 0.7),
        ("Flawless.\n**SCORE: 10**", 1.0),  # markdown-bolded verdicts still parse
        ("SCORE: 999", 1.0),  # clamped to the 0-10 scale
        ("SCORE: not-a-number", None),
        ("I refuse to commit to a verdict.", None),
    ]:
        assert JudgeTask.parse_score(echoing_trace(task, reply)) == expected


async def test_agentic_judge_task_uploads_the_trace():
    """`AgenticJudgeTask` carries the solver's entire serialized trace as data — its
    `setup` hook writes it to `TRACE_PATH` in the judge's runtime, and the assignment
    prompt points there."""
    task = vf.Task(idx=0, prompt="what is 2+2?")
    solved = echoing_trace(task, "It is 4.")
    solved.record_reward("correct", 1.0)
    grading = AgenticJudgeTask.for_trace(task, solved)
    assert TRACE_PATH in grading.prompt
    payload = json.loads(grading.trace_json)
    assert payload["id"] == solved.id and payload["rewards"] == {"correct": 1.0}

    written: dict[str, bytes] = {}

    class StubRuntime:
        async def write(self, path: str, data: bytes) -> None:
            written[path] = data

    await grading.setup(echoing_trace(grading, ""), StubRuntime())
    assert json.loads(written[TRACE_PATH]) == payload


async def test_writer_editors_fan_in_and_shared_verdict(monkeypatch):
    """The rounds + fan-in example, stubbed: editors fan out over the draft, the revision
    is linked under the draft AND every editor trace, and one (stubbed) judge call puts the
    same `improvement` reward on every trace of the instance."""
    from writer_editors_v1.topology import ImprovementJudge

    config = EvalConfig(
        topology={"id": "writer-editors-v1", "num_editors": 2}, rich=False
    )
    env = TopologyRunner(config.topology, config)
    monkeypatch.setattr(
        TopologyRunner, "rollout", lambda self, agent, task: EchoingRollout(task)
    )

    async def stub_evaluate(self, *, trace=None, **fields):
        assert {"brief", "first", "final"} <= fields.keys()
        return vf.JudgeResponse(text="SCORE: 8", parsed=0.8)

    monkeypatch.setattr(ImprovementJudge, "evaluate", stub_evaluate)
    graph = await env.run_instance(env.topology.load_tasks()[0])
    assert graph.error is None
    draft, edit_a, edit_b, revision = graph.traces
    assert [t.agent for t in graph.traces] == ["writer", "editor", "editor", "writer"]
    assert edit_a.parents == edit_b.parents == [draft.id]
    assert revision.parents == [draft.id, edit_a.id, edit_b.id]  # the fan-in
    assert all(t.rewards["improvement"] == 0.8 for t in graph.traces)


def test_instance_record_roundtrips():
    """The graph is the serialized instance artifact: `to_record` nests the traces, and
    `load` rebuilds it without the originating packages (WireTrace-typed traces, links
    intact) — what `results.jsonl` consumers and the trainer wire rely on."""
    graph = AgentGraph(topology="llm-judge")
    trace = echoing_trace(vf.Task(idx=0, prompt="hi"), "hi")
    trace.agent = "judge"
    trace.parents = ["abc123"]
    trace.trainable = False
    trace.record_reward("relay", 1.0)
    graph.add(trace)
    loaded = AgentGraph.load(json.loads(json.dumps(graph.to_record())))
    assert loaded.id == graph.id and loaded.topology == "llm-judge"
    (t,) = loaded.traces
    assert isinstance(t, WireTrace)
    assert (t.agent, t.parents, t.trainable) == ("judge", ["abc123"], False)
    assert t.reward == 1.0


@pytest.mark.e2e
@pytest.mark.subprocess
@pytest.mark.null
async def test_echo_chain_live(tmp_path):
    """The echo-chain fixture topology end to end via `run_topology_eval` (live model, null
    harness in-subprocess): two linked reward-1.0 traces per instance, persisted as one
    instance record."""
    from verifiers.v1.cli.eval.runner import run_topology_eval

    config = EvalConfig(
        topology={"id": "echo-chain-v1"},
        num_tasks=1,
        max_turns=2,
        sampling={"max_tokens": 2048, "temperature": 0},
        timeout={"rollout": 180, "scoring": 60},
        retries={"rollout": {"max_retries": 2, "include": ["ProviderError"]}},
        rich=False,
        output_dir=tmp_path,
    )
    env = TopologyRunner(config.topology, config)
    traces = await run_topology_eval(env, config)
    first, second = traces
    assert first.errors == [] and second.errors == []
    assert second.parents == [first.id]
    assert first.rewards == {"echoed": 1.0, "relay": 1.0}
    assert second.reward == 1.0
    (line,) = (tmp_path / "results.jsonl").read_text().splitlines()
    graph = AgentGraph.load(json.loads(line))
    assert [t.agent for t in graph.traces] == ["first", "second"]
    assert graph.error is None
