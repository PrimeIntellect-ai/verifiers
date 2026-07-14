"""Topology tests: config narrowing, agent loading, and instance semantics.

The unit tests stub `vf.Agent.run` with a canned reply, so the explicit-agent topology
surface — forward trace→task arrows, fan-out, parent links, declared instance judgement,
`go` failure capture — runs without a model or a runtime. The `e2e`-marked test runs the
`echo-chain-v1` fixture topology live (skipped without a key), mirroring `test_e2e`.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1.cli.output import output_path
from verifiers.v1.cli.resolve import narrow_config
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.graph import MessageNode
from verifiers.v1.serve.types import RunResponse
from verifiers.v1.topologies.agentic_judge import TRACE_PATH, AgenticJudgeTask
from verifiers.v1.topologies.llm_judge import JudgeTask
from verifiers.v1.topology import AgentGraph, TopologyRunner
from verifiers.v1.trace import Trace, TraceTask, WireTrace
from verifiers.v1.types import AssistantMessage, UserMessage


def echoing_trace(task: vf.Task, reply: str) -> Trace:
    """A minimal completed trace: one user turn, one sampled assistant reply."""
    trace = Trace(task=TraceTask(type=type(task).__name__, data=task.data))
    trace.nodes.append(
        MessageNode(parent=None, message=UserMessage(content=str(task.data.prompt)))
    )
    trace.nodes.append(
        MessageNode(parent=0, message=AssistantMessage(content=reply), sampled=True)
    )
    trace.stop("agent_completed")
    return trace


@pytest.fixture
def stub_agent_run(monkeypatch) -> None:
    """Run topology instances through real `TopologyRunner.serving`, but replace the
    executable agent's rollout work with a tiny deterministic trace."""

    async def fake_agent_run(self, task, *, parents=(), runtime=None, retry=None):
        trace = echoing_trace(task, getattr(task.data, "answer", "ok"))
        trace.record_reward("echoed", 1.0)
        self.stamp(
            trace, parents=parents, runtime=runtime, borrowed=runtime is not None
        )
        return trace

    monkeypatch.setattr(vf.Agent, "run", fake_agent_run)


async def run_stubbed_instance(env: TopologyRunner, task: vf.Task) -> AgentGraph:
    ctx = vf.ModelContext(
        model="org/model", client=object(), sampling=vf.SamplingConfig()
    )
    async with env.serving():
        return await env.run_instance(task, ctx)


@pytest.fixture
def echo_chain_env(stub_agent_run) -> TopologyRunner:
    """A loaded echo-chain topology whose agents return canned traces."""
    config = EvalConfig(topology={"id": "echo-chain-v1"}, rich=False)
    return TopologyRunner(config.topology, config)


@pytest.fixture
def proposer_seed(monkeypatch) -> vf.Task:
    """A local seed with the AIME data shape, avoiding a research-plugin dependency in unit tests."""
    taskset = vf.load_taskset(vf.taskset_config_type("echo-v1")(id="echo-v1"))
    seed = taskset.load()[0]
    from proposer_solver_v1.topology import ProposerSolverTopology

    monkeypatch.setattr(ProposerSolverTopology, "load_tasks", lambda self: [seed])
    return seed


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
    from proposer_solver_v1.topology import ProposerSolverConfig

    # (a) sibling field override keeps the subclass pin
    config = ProposerSolverConfig.model_validate({"solver": {"model": "org/strong"}})
    assert config.proposer.harness.id == "default"
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
    assert vf.NullAgentConfig().harness.id == "null"
    assert vf.NullAgentConfig.model_validate({"model": "org/x"}).harness.id == "null"


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
    assert tasks and tasks[0].data.answer == "hello world"


def test_topology_without_seeds_is_refused():
    config = EvalConfig(topology={"id": "llm-judge"}, rich=False)
    with pytest.raises(ValueError, match="has no seed tasks"):
        vf.load_topology(config.topology).load_tasks()


def test_proposer_solver_fixes_aime26_seed_taskset(monkeypatch):
    from proposer_solver_v1.topology import AIME_TASKSET_ID

    config = EvalConfig(topology={"id": "proposer-solver-v1"}, rich=False)
    topology = vf.load_topology(config.topology)
    requested: list[str] = []

    def config_type(taskset_id):
        requested.append(taskset_id)
        return vf.TasksetConfig

    class StubTaskset:
        def load(self):
            return ["seed"]

    monkeypatch.setattr(vf, "taskset_config_type", config_type)
    monkeypatch.setattr(vf, "load_taskset", lambda config: StubTaskset())
    assert topology.load_tasks() == ["seed"]
    assert requested == [AIME_TASKSET_ID]


def test_proposer_solver_refuses_taskset_override():
    """Its AIME source is fixed in `load_tasks`, so the generic taskset slot is invalid."""
    config = EvalConfig(
        topology={"id": "proposer-solver-v1", "taskset": {"id": "echo-v1"}}, rich=False
    )
    with pytest.raises(ValueError, match="constructs its own seeds"):
        TopologyRunner(config.topology, config)


def test_unknown_agent_is_refused(echo_chain_env):
    with pytest.raises(ValueError, match="unknown agent 'thrid'"):
        echo_chain_env.agent("thrid")


async def test_run_instance_requires_serving(echo_chain_env):
    ctx = vf.ModelContext(
        model="org/model", client=object(), sampling=vf.SamplingConfig()
    )
    with pytest.raises(RuntimeError, match="inside TopologyRunner.serving"):
        await echo_chain_env.run_instance(echo_chain_env.topology.load_tasks()[0], ctx)


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
    graph = await run_stubbed_instance(echo_chain_env, seed)
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
    assert second.task.data.answer == seed.data.answer
    # backward arrow: relay reward recorded on the upstream trace after the child finished
    assert first.rewards == {"echoed": 1.0, "relay": 1.0}
    assert second.rewards == {"echoed": 1.0}


async def test_declared_judgement_scores_the_instance(stub_agent_run, proposer_seed):
    """`Topology.score` runs the declared @metric/@reward methods once per matching trace
    after `go` returns — metrics before rewards (`difficulty` reads the `solve_rate`
    metric), and over every trace in scope even when the instance ended early (the stub
    never calls the submission tool, so `go` stops after the proposer). The proposer's
    task-declared parseability reward is absent because the stub skips episode scoring."""
    config = EvalConfig(topology={"id": "proposer-solver-v1"}, rich=False)
    env = TopologyRunner(config.topology, config)
    graph = await run_stubbed_instance(env, env.topology.load_tasks()[0])
    (proposer,) = graph.traces  # malformed proposal → no solver fan-out
    assert proposer.metrics == {"solve_rate": 0.0}
    assert proposer.rewards == {"echoed": 1.0, "difficulty": 0.0}


async def test_proposer_solver_fan_out_and_difficulty(monkeypatch, proposer_seed):
    """The happy path: the proposer commits a submission, `go` fans `num_solvers` solver
    episodes out over the derived AIME task (all parented under the proposer), and the
    declared judgement averages the solvers' `correct` into `solve_rate` → `difficulty`
    (peaked at 50%, so a 2/4 split scores 1.0)."""
    verdicts = iter([1.0, 0.0, 1.0, 0.0])

    async def fake_agent_run(self, task, *, parents=(), runtime=None, retry=None):
        if self.name == "proposer":
            trace = echoing_trace(task, "submitted")
            trace.info["submission"] = {
                "question": "What is 1?",
                "answer": "1",
            }
        else:
            trace = echoing_trace(task, "\\boxed{1}")
            trace.record_reward("correct", next(verdicts))
        self.stamp(
            trace, parents=parents, runtime=runtime, borrowed=runtime is not None
        )
        return trace

    monkeypatch.setattr(vf.Agent, "run", fake_agent_run)
    config = EvalConfig(topology={"id": "proposer-solver-v1"}, rich=False)
    env = TopologyRunner(config.topology, config)
    graph = await run_stubbed_instance(env, env.topology.load_tasks()[0])
    assert graph.error is None
    proposer, *solvers = graph.traces
    assert [t.agent for t in solvers] == ["solver"] * 4  # num_solvers default
    assert all(t.parents == [proposer.id] for t in solvers)
    assert all(
        t.task.data.prompt == solvers[0].task.data.prompt for t in solvers
    )  # one minted task
    assert all(t.task.data.answer == "1" for t in solvers)
    assert proposer.metrics == {"solve_rate": 0.5}
    assert proposer.rewards == {"difficulty": 1.0}


async def test_provisioned_runtime_is_borrowed_across_runs(echo_chain_env, monkeypatch):
    """`run.agent(...).provision()` hands `go` a live box whose lifetime the provisioner
    owns; runs passed `runtime=box` are stamped borrowed and never tear it down — and a
    bare-id parent (`parents=[trace.id]`) links the same as a `Trace`."""

    class FakeRuntime:
        config = vf.SubprocessConfig()
        descriptor = "fake"
        started = stopped = False

        async def start(self):
            self.started = True

        async def stop(self):
            self.stopped = True

    import verifiers.v1.agent as agent_module

    fake = FakeRuntime()
    monkeypatch.setattr(agent_module, "make_runtime", lambda config: fake)
    env = echo_chain_env
    seed = env.topology.load_tasks()[0]
    ctx = vf.ModelContext(
        model="org/model", client=object(), sampling=vf.SamplingConfig()
    )
    async with env.serving():
        run = vf.TopologyRun(env, seed, env._agents_for(ctx))
        agent = run.agent("first")
        async with agent.provision(seed) as box:
            assert box is fake and fake.started and not fake.stopped
            first = await agent.run(seed, runtime=box)
            second = await agent.run(seed, runtime=box, parents=[first.id])  # str arm
        assert fake.stopped  # provisioner owns teardown, on exit
    assert first.info["agent"]["runtime"]["borrowed"] is True
    assert second.info["agent"]["runtime"]["borrowed"] is True
    assert second.parents == [first.id]
    assert run.graph.traces == [first, second]


async def test_shared_runtime_topology_hands_off_through_one_box(monkeypatch):
    """The shared-runtime example, stubbed: the writer's borrowed run hands its note to a
    reader run in the SAME provisioned box (both traces stamped borrowed), and the
    deferred writer reward mirrors the reader's verification."""

    async def fake_agent_run(self, task, *, parents=(), runtime=None, retry=None):
        trace = echoing_trace(task, "shared runtime handoff ready.")
        if self.name == "writer":
            trace.info["shared_runtime"] = {"wrote": "note", "path": "shared/note.txt"}
        else:
            trace.record_reward("read_shared_note", 1.0)
        self.stamp(
            trace, parents=parents, runtime=runtime, borrowed=runtime is not None
        )
        return trace

    monkeypatch.setattr(vf.Agent, "run", fake_agent_run)
    config = EvalConfig(topology={"id": "shared-runtime-v1"}, rich=False)
    env = TopologyRunner(config.topology, config)
    graph = await run_stubbed_instance(env, env.topology.load_tasks()[0])
    assert graph.error is None
    written, read = graph.traces
    assert (written.agent, read.agent) == ("writer", "reader")
    assert read.parents == [written.id]
    assert read.task.data.expected == "note"  # the forward arrow carried the handoff
    # both runs were placed into the one provisioned box
    assert written.info["agent"]["runtime"]["borrowed"] is True
    assert read.info["agent"]["runtime"]["borrowed"] is True
    assert written.rewards == {"handoff_succeeded": 1.0}  # mirrored off the reader


async def test_go_failure_is_captured_on_graph(echo_chain_env, monkeypatch):
    """A crash in topology-authored code is classified `TopologyError` and recorded on the
    graph — completed episodes stay, nothing raises (a bad instance is data)."""
    cls = type(echo_chain_env.topology)

    async def exploding_go(self, task, run):
        await run.agent("first").run(task)
        raise RuntimeError("boom")

    monkeypatch.setattr(cls, "go", exploding_go)
    graph = await run_stubbed_instance(
        echo_chain_env, echo_chain_env.topology.load_tasks()[0]
    )
    assert graph.error is not None
    assert graph.error.type == "TopologyError"
    assert "boom" in graph.error.message
    assert len(graph.traces) == 1  # the episode that completed before the crash


def test_eval_config_rejects_topology_with_taskset():
    with pytest.raises(ValueError, match="drop `--taskset.id`"):
        EvalConfig(
            topology={"id": "echo-chain-v1"}, taskset={"id": "echo-v1"}, rich=False
        )


def test_eval_config_allows_topology_with_server():
    """`--server` is the same ZMQ pool path prime-rl trains through; topologies use it."""
    config = EvalConfig(topology={"id": "echo-chain-v1"}, server=True, rich=False)
    assert config.server is True
    assert config.push is False  # push still forced off for explicit topologies


def test_eval_config_rejects_topology_with_resume():
    with pytest.raises(ValueError, match="does not support `--resume`"):
        EvalConfig(topology={"id": "echo-chain-v1"}, resume=Path("."), rich=False)


def test_eval_config_forces_no_push_for_topology():
    config = EvalConfig(topology={"id": "echo-chain-v1"}, rich=False)
    assert config.push is False
    assert config.topology.id == "echo-chain-v1"


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
    """`JudgeTask.from_trace` (main's `Task.from_trace` convention) peels the judge's
    inputs off the finished episode: the seed
    task's framing, its ground truth (an `answer` field, when the task carries one), and
    the solver's final message (last message of the final branch, not the transcript)."""

    class AnsweredData(vf.TaskData):
        answer: str

    class AnsweredTask(vf.Task[AnsweredData]):
        pass

    task = AnsweredTask(AnsweredData(idx=0, prompt="what is 2+2?", answer="4"))
    solved = echoing_trace(task, "It is 4.")
    grading = JudgeTask.from_trace(solved)
    assert "what is 2+2?" in grading.data.prompt and "It is 4." in grading.data.prompt
    assert "<reference_answer>\n4\n</reference_answer>" in grading.data.prompt
    # a task without ground truth just omits the reference section
    unanswered = echoing_trace(vf.Task(vf.TaskData(idx=0, prompt="write a poem")), "ok")
    bare = JudgeTask.from_trace(unanswered)
    assert "<reference_answer>" not in bare.data.prompt
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
    task = vf.Task(vf.TaskData(idx=0, prompt="what is 2+2?"))
    solved = echoing_trace(task, "It is 4.")
    solved.record_reward("correct", 1.0)
    grading = AgenticJudgeTask.from_trace(solved)
    assert TRACE_PATH in grading.data.prompt
    payload = json.loads(grading.data.trace_json)
    assert payload["id"] == solved.id and payload["rewards"] == {"correct": 1.0}

    written: dict[str, bytes] = {}

    class StubRuntime:
        async def write(self, path: str, data: bytes) -> None:
            written[path] = data

    await grading.setup(echoing_trace(grading, ""), StubRuntime())
    assert json.loads(written[TRACE_PATH]) == payload


async def test_writer_editors_fan_in_and_shared_verdict(monkeypatch):
    """The rounds + fan-in example, stubbed: editors fan out over the draft, the revision
    is linked under the draft AND every editor trace, and the deterministic first→final
    band score puts the same `improvement` reward on every trace of the instance."""
    from writer_editors_v1.topology import CritiqueTask, DraftTask, ReviseTask

    config = EvalConfig(
        topology={"id": "writer-editors-v1", "num_editors": 2}, rich=False
    )
    env = TopologyRunner(config.topology, config)

    async def fake_agent_run(self, task, *, parents=(), runtime=None, retry=None):
        # Short first draft, in-band revision — improvement = 1.0 - short/120 > 0.
        if isinstance(task, DraftTask):
            reply = "short draft"
        elif isinstance(task, ReviseTask):
            reply = " ".join(["word"] * 150)
        elif isinstance(task, CritiqueTask):
            reply = "tighten the opening sentence"
        else:
            reply = "ok"
        trace = echoing_trace(task, reply)
        self.stamp(
            trace, parents=parents, runtime=runtime, borrowed=runtime is not None
        )
        return trace

    monkeypatch.setattr(vf.Agent, "run", fake_agent_run)
    graph = await run_stubbed_instance(env, env.topology.load_tasks()[0])
    assert graph.error is None
    draft, edit_a, edit_b, revision = graph.traces
    assert [t.agent for t in graph.traces] == ["writer", "editor", "editor", "writer"]
    assert edit_a.parents == edit_b.parents == [draft.id]
    assert revision.parents == [draft.id, edit_a.id, edit_b.id]  # the fan-in
    score = draft.rewards["improvement"]
    assert score > 0.0
    assert all(t.rewards["improvement"] == score for t in graph.traces)


def test_instance_record_roundtrips():
    """The graph is the serialized instance artifact: `to_record` nests the traces, and
    `load` rebuilds it without the originating packages (WireTrace-typed traces, links
    intact) — what `traces.jsonl` consumers and the trainer wire rely on."""
    task = vf.Task(vf.TaskData(idx=0, prompt="hi"))
    graph = AgentGraph(
        topology="llm-judge",
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
    )
    trace = echoing_trace(task, "hi")
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


def test_instance_wire_preserves_training_tensors():
    """Server serialization keeps tensors that JSON persistence intentionally drops."""
    task = vf.TraceTask(type="Task", data=vf.WireTaskData(idx=0, prompt="hi"))
    node = MessageNode(
        message=AssistantMessage(content="hi"),
        token_ids=[1],
        mask=[True],
        sampled=True,
        routed_experts=np.ones((1, 2, 1), dtype=np.uint8),
    )
    graph = AgentGraph(task=task, traces=[Trace(task=task, nodes=[node])])

    wire = RunResponse(graph=graph).model_dump(mode="python")
    loaded = RunResponse.model_validate(wire).graph

    assert loaded is not None
    assert loaded.traces[0].nodes[0].routed_experts.shape == (1, 2, 1)


@pytest.mark.e2e
@pytest.mark.subprocess
@pytest.mark.null
async def test_echo_chain_live(tmp_path):
    """The echo-chain fixture topology end to end via `run_eval` (live model, null
    harness in-subprocess): two linked reward-1.0 traces per instance, persisted flat."""
    from verifiers.v1.cli.eval.runner import run_eval

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
    traces = await run_eval(config)
    first, second = traces
    assert first.errors == [] and second.errors == []
    assert second.parents == [first.id]
    assert first.rewards == {"echoed": 1.0, "relay": 1.0}
    assert second.reward == 1.0
    lines = [
        json.loads(line)
        for line in (tmp_path / "traces.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["agent"] for row in lines] == ["first", "second"]
    assert lines[1]["parents"] == [first.id]
