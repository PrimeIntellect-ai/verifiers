"""Pluggable judges: plugin resolution, base-`TasksetConfig.judges` narrowing, the built-in
`binary` / `rubric` judges, and `Taskset.score` running plugged judges after the decorated
rewards. Judge model calls are faked at `Judge.complete` — no network."""

import json

import pytest

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.judge import Judge, JudgeResponse
from verifiers.v1.loaders import judge_class, judge_config_type, load_judge
from verifiers.v1.types import AssistantMessage, UserMessage

RUBRIC_TOML = """
[[criteria]]
name = "mentions_paris"
text = "The response mentions Paris."
weight = 3.0

[[criteria]]
name = "is_polite"
text = "The response is polite."
"""


class QATask(vf.Task):
    answer: str = ""


def make_trace(reply: str = "It is Paris.") -> vf.Trace:
    return vf.Trace(
        task=QATask(idx=0, prompt="Capital of France?", answer="Paris"),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content="Capital of France?"),
                sampled=False,
            ),
            MessageNode(
                parent=0, message=AssistantMessage(content=reply), sampled=True
            ),
        ],
    )


@pytest.fixture
def fake_judge_model(monkeypatch):
    """Fake the judge's model call: reply "yes" iff "Paris" appears in the response block of
    the rendered prompt, recording each prompt for assertions."""
    prompts: list[str] = []

    async def fake_complete(
        self, messages, *, trace=None, schema=None, parse=None, **sampling
    ):
        prompts.append(messages)
        response = JudgeResponse(
            text="yes" if "Paris" in messages.split("Response:")[-1] else "no"
        )
        if parse is not None:
            response.parsed = parse(response)
        if trace is not None:
            trace.record_judge(response)
        return response

    monkeypatch.setattr(Judge, "complete", fake_complete)
    return prompts


# --- plugin resolution + config narrowing --------------------------------------------------


def test_judge_plugin_resolution():
    assert judge_class("binary") is vf.BinaryJudge
    assert judge_class("rubric") is vf.RubricJudge
    assert judge_config_type("binary") is vf.BinaryJudgeConfig
    assert judge_config_type("rubric") is vf.RubricJudgeConfig
    assert isinstance(load_judge(vf.BinaryJudgeConfig()), vf.BinaryJudge)
    # built-ins pin their id, so a code-level default entry needs no explicit id
    assert vf.BinaryJudgeConfig().id == "binary"
    assert vf.RubricJudgeConfig(path="x.toml").id == "rubric"


def test_judge_config_cls_defaults():
    # A judge written without the config generic falls back to the base JudgeConfig; a
    # parameterized one default-constructs its own config type.
    class PlainJudge(vf.Judge[bool]):
        prompt = "{x}"

    assert type(PlainJudge().config) is vf.JudgeConfig
    assert type(vf.BinaryJudge().config) is vf.BinaryJudgeConfig


def test_taskset_config_narrows_judges(tmp_path):
    # `judges` entries narrow to the config type their id resolves to (like taskset/harness
    # narrowing in EnvConfig), so judge-specific fields validate against the real config —
    # and survive model_dump (SerializeAsAny), which the env-server wire depends on.
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(RUBRIC_TOML)
    cfg = vf.TasksetConfig.model_validate(
        {
            "judges": [
                {"id": "binary", "answer_field": "gold", "weight": 0.5},
                {"id": "rubric", "path": str(rubric), "name": "quality"},
            ]
        }
    )
    binary, rubric_cfg = cfg.judges
    assert isinstance(binary, vf.BinaryJudgeConfig) and binary.answer_field == "gold"
    assert isinstance(rubric_cfg, vf.RubricJudgeConfig) and rubric_cfg.name == "quality"
    assert cfg.model_dump()["judges"][0]["answer_field"] == "gold"
    # a round-trip through the dump re-narrows to the same types
    again = vf.TasksetConfig.model_validate(cfg.model_dump())
    assert isinstance(again.judges[0], vf.BinaryJudgeConfig)


def test_judges_entry_requires_id():
    with pytest.raises(ValueError, match="needs an `id`"):
        vf.TasksetConfig.model_validate({"judges": [{"weight": 1.0}]})


def test_rubric_config_requires_path():
    # `path` is a required Path field: a plugged rubric judge without one fails at config time.
    with pytest.raises(ValueError, match="path"):
        vf.TasksetConfig.model_validate({"judges": [{"id": "rubric"}]})


def test_judges_reject_shared_reward_keys():
    # Ids may repeat (same plugin, two configs) — what must be unique is the derived reward
    # key (`name`, else the id's package name), checked at config time.
    with pytest.raises(ValueError, match="share a reward key"):
        vf.TasksetConfig.model_validate(
            {"judges": [{"id": "binary"}, {"id": "binary"}]}
        )
    cfg = vf.TasksetConfig.model_validate(
        {
            "judges": [
                {"id": "binary", "name": "strict"},
                {"id": "binary", "name": "lenient"},
            ]
        }
    )
    assert [judge.name for judge in cfg.judges] == ["strict", "lenient"]


def test_reward_name_fallback():
    # name > id's package name > snake-cased class name (code-level judge with neither).
    class MyQualityJudge(vf.Judge[float]):
        prompt = "{x}"

    assert vf.BinaryJudge().reward_name == "binary"  # class-name fallback (no id set)
    assert (
        vf.BinaryJudge(vf.BinaryJudgeConfig(id="org/my-judge@1.0.0")).reward_name
        == "my-judge"
    )
    assert vf.BinaryJudge(vf.BinaryJudgeConfig(name="gold")).reward_name == "gold"
    assert MyQualityJudge().reward_name == "my_quality"


async def test_base_judge_score_raises():
    with pytest.raises(NotImplementedError, match="implements no `score`"):
        await vf.Judge().score(task=QATask(idx=0, prompt="q"), trace=make_trace())


# --- binary --------------------------------------------------------------------------------


def test_binary_parse():
    judge = vf.BinaryJudge()
    assert judge.parse(JudgeResponse(text="yes")) == 1.0
    assert judge.parse(JudgeResponse(text="Final answer: NO")) == 0.0
    assert judge.parse(JudgeResponse(text="gibberish")) == 0.0


async def test_binary_score(fake_judge_model):
    trace = make_trace()
    verdict = await vf.BinaryJudge(vf.BinaryJudgeConfig(id="binary")).score(
        trace.task, trace
    )
    assert verdict == 1.0
    assert (
        "Capital of France?" in fake_judge_model[0]
    )  # the task prompt is in the judge prompt
    assert len(trace.info["judge"]) == 1  # the call is recorded onto the trace

    trace = make_trace(reply="It is Rome.")
    assert await vf.BinaryJudge().score(trace.task, trace) == 0.0


async def test_binary_score_missing_answer_field():
    trace = make_trace()
    judge = vf.BinaryJudge(vf.BinaryJudgeConfig(answer_field="gold"))
    with pytest.raises(ValueError, match="no 'gold' field"):
        await judge.score(trace.task, trace)


async def test_binary_score_messages_prompt(fake_judge_model):
    # A Messages-form prompt still reaches the judge as text (via Task.prompt_text).
    from verifiers.v1.types import TextContentPart, UserMessage as UM

    task = QATask(
        idx=0,
        prompt=[UM(content=[TextContentPart(text="Capital of France?")])],
        answer="Paris",
    )
    trace = vf.Trace(
        task=task,
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )
    assert await vf.BinaryJudge().score(task, trace) == 1.0
    assert "Capital of France?" in fake_judge_model[0]


async def test_binary_question_field(fake_judge_model):
    # question_field points {question} at a dedicated task field instead of the full prompt.
    class FieldTask(vf.Task):
        question: str = ""
        answer: str = ""

    task = FieldTask(
        idx=0,
        prompt="SYSTEM INSTRUCTIONS\n\nQuestion: Capital of France?",
        question="Capital of France?",
        answer="Paris",
    )
    trace = vf.Trace(
        task=task,
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )
    await vf.BinaryJudge(vf.BinaryJudgeConfig(question_field="question")).score(
        task, trace
    )
    assert "Capital of France?" in fake_judge_model[0]
    assert "SYSTEM INSTRUCTIONS" not in fake_judge_model[0]

    judge = vf.BinaryJudge(vf.BinaryJudgeConfig(question_field="typo"))
    with pytest.raises(ValueError, match="no 'typo' field"):
        await judge.score(task, trace)


def full_trace_fixture() -> vf.Trace:
    """A multi-turn trace: user -> assistant (reasoning + tool call) -> tool -> final reply."""
    from verifiers.v1.types import ToolCall, ToolMessage

    return vf.Trace(
        task=QATask(idx=0, prompt="Capital of France?", answer="Paris"),
        nodes=[
            MessageNode(
                parent=None,
                message=UserMessage(content="Capital of France?"),
                sampled=False,
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(
                    content="Let me look it up.",
                    reasoning_content="SECRET REASONING",
                    tool_calls=[
                        ToolCall(id="1", name="search", arguments='{"q": "france"}')
                    ],
                ),
                sampled=True,
            ),
            MessageNode(
                parent=1,
                message=ToolMessage(
                    tool_call_id="1",
                    name="search",
                    content="TOOL RESULT: Paris is the capital.",
                ),
                sampled=False,
            ),
            MessageNode(
                parent=2, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )


def test_transcript():
    transcript = full_trace_fixture().transcript
    assert "[user]\nCapital of France?" in transcript
    assert '[tool_call search({"q": "france"})]' in transcript
    assert "[tool search]\nTOOL RESULT: Paris is the capital." in transcript
    assert "It is Paris." in transcript
    assert "SECRET REASONING" not in transcript  # reasoning is excluded


async def test_view_modes(fake_judge_model):
    # last_reply (default): the judge sees only the final reply.
    trace = full_trace_fixture()
    await vf.BinaryJudge().score(trace.task, trace)
    assert "TOOL RESULT" not in fake_judge_model[0]
    # full_trace: the whole transcript (minus reasoning) fills {response}.
    trace = full_trace_fixture()
    await vf.BinaryJudge(vf.BinaryJudgeConfig(view="full_trace")).score(
        trace.task, trace
    )
    assert "TOOL RESULT: Paris is the capital." in fake_judge_model[1]
    assert "SECRET REASONING" not in fake_judge_model[1]


async def test_rubric_view_full_trace(tmp_path, monkeypatch):
    # The rubric judge shares the view field: criteria are judged against the transcript.
    prompts: list[str] = []

    async def fake_complete(
        self, messages, *, trace=None, schema=None, parse=None, **s
    ):
        prompts.append(messages)
        response = JudgeResponse(text="yes")
        if parse is not None:
            response.parsed = parse(response)
        return response

    monkeypatch.setattr(Judge, "complete", fake_complete)
    judge = rubric_judge(tmp_path, view="full_trace")
    trace = full_trace_fixture()
    await judge.score(trace.task, trace)
    assert all("TOOL RESULT" in prompt for prompt in prompts)


def test_config_prompt_overrides_class_template():
    judge = vf.BinaryJudge(
        vf.BinaryJudgeConfig(prompt="Q:{question} A:{answer} R:{response}")
    )
    assert judge.build_messages(question="q", answer="a", response="r") == "Q:q A:a R:r"


# --- rubric --------------------------------------------------------------------------------


def rubric_judge(
    tmp_path, body: str = RUBRIC_TOML, suffix: str = ".toml", **kwargs
) -> vf.RubricJudge:
    path = tmp_path / f"rubric{suffix}"
    path.write_text(body)
    return vf.RubricJudge(vf.RubricJudgeConfig(path=str(path), **kwargs))


def test_rubric_criteria_toml_and_json(tmp_path):
    toml = rubric_judge(tmp_path).criteria
    assert [c.name for c in toml] == ["mentions_paris", "is_polite"]
    assert [c.weight for c in toml] == [3.0, 1.0]
    # JSON: both the {"criteria": [...]} object and a bare list parse to the same rubric
    items = [c.model_dump() for c in toml]
    assert (
        rubric_judge(tmp_path, json.dumps({"criteria": items}), ".json").criteria
        == toml
    )
    assert rubric_judge(tmp_path, json.dumps(items), ".json").criteria == toml


def test_rubric_config_weight_overrides_file(tmp_path):
    judge = rubric_judge(tmp_path, weights={"mentions_paris": 1.0})
    assert [c.weight for c in judge.criteria] == [1.0, 1.0]


def test_rubric_rejects_bad_files(tmp_path):
    with pytest.raises(ValueError, match="no criteria"):
        _ = rubric_judge(tmp_path, "").criteria
    duplicate = RUBRIC_TOML.replace("is_polite", "mentions_paris")
    with pytest.raises(ValueError, match="duplicate"):
        _ = rubric_judge(tmp_path, duplicate).criteria
    with pytest.raises(ValueError, match="name no criterion"):
        _ = rubric_judge(tmp_path, weights={"typo": 2.0}).criteria
    # all-zero weights fail while loading the rubric — before any judge call is paid for
    with pytest.raises(ValueError, match="no positive criterion weight"):
        _ = rubric_judge(
            tmp_path, weights={"mentions_paris": 0.0, "is_polite": 0.0}
        ).criteria
    # a negative weight would invert a criterion and break the [0, 1] reward range
    with pytest.raises(ValueError, match="negative criterion weights"):
        _ = rubric_judge(tmp_path, weights={"mentions_paris": -1.0}).criteria


async def test_rubric_score(tmp_path, monkeypatch):
    # verdicts: mentions_paris=1 (w=3), is_polite=0 (w=1) -> weighted mean 0.75; each verdict
    # lands as a `<name>/<criterion>` metric and every call is recorded onto the trace.
    async def fake_complete(
        self, messages, *, trace=None, schema=None, parse=None, **sampling
    ):
        text = "yes" if "Paris" in messages.split("Criterion:")[-1] else "no"
        response = JudgeResponse(text=text)
        if parse is not None:
            response.parsed = parse(response)
        if trace is not None:
            trace.record_judge(response)
        return response

    monkeypatch.setattr(Judge, "complete", fake_complete)
    judge = rubric_judge(tmp_path)
    trace = make_trace()
    assert await judge.score(trace.task, trace) == 0.75
    assert trace.metrics == {"rubric/mentions_paris": 1.0, "rubric/is_polite": 0.0}
    assert len(trace.info["judge"]) == 2


# --- Taskset.score integration -------------------------------------------------------------


class JudgedConfig(vf.TasksetConfig):
    pass


class JudgedTaskset(vf.Taskset[QATask, JudgedConfig]):
    def load_tasks(self) -> list[QATask]:
        return []

    @vf.reward
    async def own(self, trace) -> float:
        return 0.25


async def test_taskset_score_runs_plugged_judges(tmp_path, fake_judge_model):
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(RUBRIC_TOML)
    cfg = JudgedConfig.model_validate(
        {
            "judges": [
                {"id": "binary", "weight": 0.5},
                {"id": "rubric", "path": str(rubric), "name": "quality"},
            ]
        }
    )
    taskset = JudgedTaskset(cfg)
    trace = make_trace()
    await taskset.score(trace, runtime=None)
    # the fake replies "yes" to everything mentioning Paris after "Response:" — for the rubric
    # template the criterion block follows the response block, so both criteria come back with
    # the response's Paris in scope; what matters here is placement and weighting, not verdicts.
    assert trace.rewards["own"] == 0.25  # decorated rewards still run
    assert trace.rewards["binary"] == 0.5  # 1.0 * weight 0.5, under the id-derived name
    assert "quality" in trace.rewards  # the rubric's aggregate, under its `name`
    assert len(trace.info["judge"]) == 3  # every judge call recorded


async def test_taskset_without_judges_scores_as_before():
    taskset = JudgedTaskset(JudgedConfig())
    trace = make_trace()
    await taskset.score(trace, runtime=None)
    assert trace.rewards == {"own": 0.25}
