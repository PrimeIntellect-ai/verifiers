"""Pluggable judges, scoring integration, and the typed provider-call lifecycle."""

import asyncio
import json
import re

import httpx
import pytest
from openai import APITimeoutError
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.judge import JudgeResponse
from verifiers.v1.loaders import judge_class, judge_config_type, load_judge
from verifiers.v1.push import trace_to_sample
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


class QAData(vf.TaskData):
    answer: str = ""


def make_trace(
    reply: str = "It is Paris.",
    answer: str = "Paris",
    task_cls: type[QAData] = QAData,
) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(
            type="Task",
            data=task_cls(idx=0, prompt="Capital of France?", answer=answer),
        ),
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


def model_response(
    text: str, *, finish_reason: str = "stop", refusal: str | None = None
) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "fake",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "refusal": refusal,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        }
    )


async def test_complete_uses_sdk_strict_schema(monkeypatch):
    request = {}

    class Verdict(BaseModel):
        score: int | None = None

    async def fake_response(self, **kwargs):
        request.update(kwargs)
        return model_response('{"score": 1}')

    monkeypatch.setattr(AsyncCompletions, "create", fake_response)
    trace = make_trace()
    response = await vf.Judge().complete("grade this", trace=trace, schema=Verdict)
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["score"]
    assert response.parsed == Verdict(score=1)
    assert trace.judge_calls[0].call.sampling is not None
    assert "response_format" not in trace.judge_calls[0].call.sampling.model_dump()
    assert trace.judge_calls[0].call.usage is not None
    assert trace.judge_calls[0].call.usage.total_tokens == 3
    restored = vf.Trace[QAData].model_validate_json(trace.model_dump_json())
    assert restored.judge_calls == trace.judge_calls
    assert trace_to_sample(trace)["info"]["judge_calls"] == [
        trace.judge_calls[0].model_dump(mode="json", exclude_none=True)
    ]


async def test_complete_rejects_truncated_structured_output(monkeypatch):
    class Verdict(BaseModel):
        score: int

    async def fake_response(self, **kwargs):
        return model_response('{"score": 1}', finish_reason="length")

    monkeypatch.setattr(AsyncCompletions, "create", fake_response)
    trace = make_trace()
    with pytest.raises(ValueError, match="structured output was truncated"):
        await vf.Judge().complete("grade this", trace=trace, schema=Verdict)
    assert trace.judge_calls[-1].call.finish_reason == "length"
    assert trace.judge_calls[-1].error is not None


async def test_complete_records_refusal_and_cancellation(monkeypatch):
    async def refuse(self, **kwargs):
        return model_response("", refusal="cannot grade")

    monkeypatch.setattr(AsyncCompletions, "create", refuse)
    trace = make_trace()
    with pytest.raises(ValueError, match="refused"):
        await vf.Judge().complete("grade this", trace=trace)
    assert trace.judge_calls[-1].refusal == "cannot grade"
    assert trace.judge_calls[-1].call.error is None
    assert trace.judge_calls[-1].error is not None

    async def cancel(self, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(AsyncCompletions, "create", cancel)
    trace = make_trace()
    with pytest.raises(asyncio.CancelledError):
        await vf.Judge().complete("grade this", trace=trace)
    provider_error = trace.judge_calls[-1].call.error
    assert provider_error is not None and provider_error.type == "CancelledError"
    assert trace.judge_calls[-1].error is None


@pytest.mark.parametrize(
    ("configured", "sampling", "expected"),
    [
        ({"max_completion_tokens": 30}, {}, {"max_completion_tokens": 30}),
        (
            {"max_tokens": 100},
            {"max_completion_tokens": 20},
            {"max_completion_tokens": 20},
        ),
    ],
)
async def test_complete_preserves_provider_sampling_aliases(
    monkeypatch, configured, sampling, expected
):
    request = {}

    async def fake_response(self, **kwargs):
        request.update(kwargs)
        return model_response("yes")

    monkeypatch.setattr(AsyncCompletions, "create", fake_response)
    await vf.Judge(vf.JudgeConfig(sampling=configured)).complete(
        "grade this", model="ignored", **sampling
    )
    assert request["model"] == "openai/gpt-5.4-nano"
    limits = {
        key: request[key]
        for key in ("max_tokens", "max_completion_tokens")
        if key in request
    }
    assert limits == expected


async def test_judge_provider_error_uses_shared_retry_type(monkeypatch):
    class JudgeTask(vf.Task[QAData]):
        @vf.reward
        async def judged(self, trace: vf.Trace) -> float:
            await vf.Judge().complete("grade this", trace=trace)
            return 1.0

    async def fail(self, **kwargs):
        raise APITimeoutError(request=httpx.Request("POST", "https://judge.test"))

    monkeypatch.setattr(AsyncCompletions, "create", fail)
    trace = make_trace()
    with pytest.raises(vf.ProviderError):
        await JudgeTask(trace.task.data).score(trace)
    assert trace.judge_calls[-1].error is None
    provider_error = trace.judge_calls[-1].call.error
    assert provider_error is not None and provider_error.type == "ProviderError"
    assert "judged" not in trace.rewards


@pytest.fixture
def fake_judge_model(monkeypatch):
    """Fake the judge's model call, recording each prompt for assertions. Rubric calls (a JSON
    `verdicts` instruction or a `schema`) reply one reasoned verdict per `- name: text` criterion,
    "yes" iff it mentions Paris; other judges reply plain "yes"/"no" by the response block."""
    prompts: list[str] = []

    async def fake_response(self, **kwargs):
        messages = kwargs["messages"][0]["content"]
        prompts.append(messages)
        # Rubric calls carry criteria lines + a JSON `verdicts` instruction; reply one verdict per
        # criterion (yes iff its text mentions Paris) with a reason. Other judges get plain yes/no.
        if "response_format" in kwargs or '"verdicts"' in messages:
            verdicts = [
                {
                    "name": name,
                    "reason": "cites Paris" if "Paris" in text else "no Paris",
                    "verdict": "yes" if "Paris" in text else "no",
                }
                for name, text in re.findall(
                    r"^- ([^:]+): (.+)$", messages, re.MULTILINE
                )
            ]
            text = json.dumps({"verdicts": verdicts})
        else:
            text = "yes" if "Paris" in messages.split("Response:")[-1] else "no"
        return model_response(text)

    monkeypatch.setattr(AsyncCompletions, "create", fake_response)
    return prompts


# --- plugin resolution + config narrowing --------------------------------------------------


def test_judge_plugin_resolution():
    assert judge_class("reference") is vf.ReferenceJudge
    assert judge_class("rubric") is vf.RubricJudge
    assert judge_config_type("reference") is vf.ReferenceJudgeConfig
    assert judge_config_type("rubric") is vf.RubricJudgeConfig
    assert isinstance(load_judge(vf.ReferenceJudgeConfig()), vf.ReferenceJudge)
    # built-ins pin their id, so a code-level default entry needs no explicit id
    assert vf.ReferenceJudgeConfig().id == "reference"
    assert vf.RubricJudgeConfig(path="x.toml").id == "rubric"

    # a judge without the config generic falls back to the base JudgeConfig
    class PlainJudge(vf.Judge[bool]):
        prompt = "{x}"

    assert type(PlainJudge().config) is vf.JudgeConfig
    assert type(vf.ReferenceJudge().config) is vf.ReferenceJudgeConfig


def test_taskset_config_narrows_judges(tmp_path):
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(RUBRIC_TOML)
    cfg = vf.TaskConfig.model_validate(
        {
            "judges": [
                {"id": "reference", "answer_field": "gold", "weight": 0.5},
                {"id": "rubric", "path": str(rubric), "name": "quality"},
            ]
        }
    )
    reference, rubric_cfg = cfg.judges
    assert (
        isinstance(reference, vf.ReferenceJudgeConfig)
        and reference.answer_field == "gold"
    )
    assert isinstance(rubric_cfg, vf.RubricJudgeConfig) and rubric_cfg.name == "quality"
    assert cfg.model_dump()["judges"][0]["answer_field"] == "gold"
    again = vf.TaskConfig.model_validate(cfg.model_dump())
    assert isinstance(again.judges[0], vf.ReferenceJudgeConfig)


def test_judges_entry_requires_id():
    with pytest.raises(ValueError, match="needs an `id`"):
        vf.TaskConfig.model_validate({"judges": [{"weight": 1.0}]})


def test_rubric_config_requires_path():
    # `path` is a required Path field: a plugged rubric judge without one fails at config time.
    with pytest.raises(ValueError, match="path"):
        vf.TaskConfig.model_validate({"judges": [{"id": "rubric"}]})


def test_judges_reject_shared_reward_keys():
    # Ids may repeat (same plugin, two configs) — what must be unique is the derived reward
    # key (`name`, else the id's package name), checked at config time.
    with pytest.raises(ValueError, match="share a reward key"):
        vf.TaskConfig.model_validate(
            {"judges": [{"id": "reference"}, {"id": "reference"}]}
        )
    cfg = vf.TaskConfig.model_validate(
        {
            "judges": [
                {"id": "reference", "name": "strict"},
                {"id": "reference", "name": "lenient"},
            ]
        }
    )
    assert [judge.name for judge in cfg.judges] == ["strict", "lenient"]

    # class-level DEFAULTS are held to the same rule (they bypass the before-hook)
    class TwoDefaults(vf.TaskConfig):
        judges: vf.Judges = Field(
            default_factory=lambda: [
                vf.ReferenceJudgeConfig(),
                vf.ReferenceJudgeConfig(),
            ]
        )

    with pytest.raises(ValueError, match="share a reward key"):
        TwoDefaults()


def test_reward_name_fallback():
    # name > id's package name > snake-cased class name (code-level judge with neither).
    class MyQualityJudge(vf.Judge[float]):
        prompt = "{x}"

    assert (
        vf.ReferenceJudge().reward_name == "reference"
    )  # class-name fallback (no id set)
    assert (
        vf.ReferenceJudge(vf.ReferenceJudgeConfig(id="org/my-judge@1.0.0")).reward_name
        == "my-judge"
    )
    assert vf.ReferenceJudge(vf.ReferenceJudgeConfig(name="gold")).reward_name == "gold"
    assert MyQualityJudge().reward_name == "my_quality"


async def test_base_judge_score_raises():
    with pytest.raises(NotImplementedError, match="implements no `score`"):
        await vf.Judge().score(task=QAData(idx=0, prompt="q"), trace=make_trace())


# --- reference --------------------------------------------------------------------------------


def test_reference_parse():
    judge = vf.ReferenceJudge()
    assert judge.parse(JudgeResponse(text="yes")) == 1.0
    assert judge.parse(JudgeResponse(text="Final answer: NO")) == 0.0
    # an unparseable verdict is a judge failure: raise (-> rollout error), don't score 0
    with pytest.raises(ValueError, match="no yes/no verdict"):
        judge.parse(JudgeResponse(text="gibberish"))


async def test_reference_score(fake_judge_model):
    trace = make_trace()
    verdict = await vf.ReferenceJudge(vf.ReferenceJudgeConfig(id="reference")).score(
        trace.task.data, trace
    )
    assert verdict == 1.0
    assert (
        "Capital of France?" in fake_judge_model[0]
    )  # the task prompt is in the judge prompt
    assert len(trace.judge_calls) == 1  # the call is recorded onto the trace
    assert trace.judge_calls[0].error is None

    trace = make_trace(reply="It is Rome.")
    assert await vf.ReferenceJudge().score(trace.task.data, trace) == 0.0
    assert trace.judge_calls[0].error is None

    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(answer_field="gold"))
    with pytest.raises(ValueError, match="no 'gold' field"):  # misconfig raises, not 0
        await judge.score(trace.task.data, trace)


async def test_reference_score_messages_prompt(fake_judge_model):
    # A Messages-form prompt still reaches the judge as text (via TaskData.prompt_text).
    from verifiers.v1.types import TextContentPart
    from verifiers.v1.types import UserMessage as UM

    task = QAData(
        idx=0,
        prompt=[UM(content=[TextContentPart(text="Capital of France?")])],
        answer="Paris",
    )
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=task),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )
    assert await vf.ReferenceJudge().score(task, trace) == 1.0
    assert "Capital of France?" in fake_judge_model[0]


async def test_reference_question_field(fake_judge_model):
    # question_field points {question} at a dedicated task field instead of the full prompt.
    class FieldTask(vf.TaskData):
        question: str = ""
        answer: str = ""

    task = FieldTask(
        idx=0,
        prompt="SYSTEM INSTRUCTIONS\n\nQuestion: Capital of France?",
        question="Capital of France?",
        answer="Paris",
    )
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=task),
        nodes=[
            MessageNode(parent=None, message=UserMessage(content="q"), sampled=False),
            MessageNode(
                parent=0, message=AssistantMessage(content="It is Paris."), sampled=True
            ),
        ],
    )
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(question_field="question")).score(
        task, trace
    )
    assert "Capital of France?" in fake_judge_model[0]
    assert "SYSTEM INSTRUCTIONS" not in fake_judge_model[0]

    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(question_field="typo"))
    with pytest.raises(ValueError, match="no 'typo' field"):
        await judge.score(task, trace)


def full_trace_fixture() -> vf.Trace:
    """A multi-turn trace: user -> assistant (reasoning + tool call) -> tool -> final reply."""
    from verifiers.v1.types import ToolCall, ToolMessage

    return vf.Trace(
        task=vf.TraceTask(
            type="Task", data=QAData(idx=0, prompt="Capital of France?", answer="Paris")
        ),
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
    await vf.ReferenceJudge().score(trace.task.data, trace)
    assert "TOOL RESULT" not in fake_judge_model[0]
    # full_trace: the whole transcript (minus reasoning) fills {response}.
    trace = full_trace_fixture()
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(view="full_trace")).score(
        trace.task.data, trace
    )
    assert "TOOL RESULT: Paris is the capital." in fake_judge_model[1]
    assert "SECRET REASONING" not in fake_judge_model[1]


def test_view_defaults():
    # reference grades the final answer; rubric grades the process by default.
    assert vf.ReferenceJudgeConfig().view == "last_reply"
    assert vf.RubricJudgeConfig(path="x.toml").view == "full_trace"


async def test_rubric_view_full_trace(tmp_path, fake_judge_model):
    # The rubric judge's default view: criteria are judged against the whole transcript.
    judge = rubric_judge(tmp_path)
    trace = full_trace_fixture()
    await judge.score(trace.task.data, trace)
    assert all("TOOL RESULT" in prompt for prompt in fake_judge_model)
    assert all("SECRET REASONING" not in prompt for prompt in fake_judge_model)


async def test_config_prompt_overrides_class_template(fake_judge_model):
    judge = vf.ReferenceJudge(
        vf.ReferenceJudgeConfig(prompt="Q:{question} A:{answer} R:{response}")
    )
    assert judge.build_messages(question="q", answer="a", response="r") == "Q:q A:a R:r"
    # A template needn't use every evaluate field: score also passes {positive}/{negative},
    # which str.format ignores when the (custom) prompt doesn't reference them.
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 1.0
    assert fake_judge_model[0] == "Q:Capital of France? A:Paris R:It is Paris."


async def test_prompt_file(tmp_path, fake_judge_model):
    # The prompt template can come from a file; the same {field} placeholders work.
    file = tmp_path / "judge.txt"
    file.write_text("Q:{question} A:{answer} R:{response}")
    trace = make_trace()
    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(prompt_file=file))
    assert await judge.score(trace.task.data, trace) == 1.0
    assert fake_judge_model[0] == "Q:Capital of France? A:Paris R:It is Paris."
    # a bad path fails at judge construction, not mid-eval at score time
    with pytest.raises(FileNotFoundError):
        vf.ReferenceJudge(vf.ReferenceJudgeConfig(prompt_file=tmp_path / "missing.txt"))
    # inline and file prompts are mutually exclusive
    with pytest.raises(ValueError, match="not both"):
        vf.ReferenceJudgeConfig(prompt="inline", prompt_file=file)


# --- reference input/verdict knobs ------------------------------------------------------------


async def test_reference_empty_response_short_circuits(fake_judge_model):
    # An empty reply scores 0 without paying for the (foregone) judge call.
    trace = make_trace(reply="")
    assert await vf.ReferenceJudge().score(trace.task.data, trace) == 0.0
    assert fake_judge_model == []
    assert "judge" not in trace.info


async def test_reference_list_answer(fake_judge_model):
    # A list-valued answer field is judged as multiple acceptable answers, one per line.
    class MultiTask(vf.TaskData):
        aliases: list[str] = Field(default_factory=list)

    task = MultiTask(idx=0, prompt="q?", aliases=["Paris", "Lutetia"])
    trace = make_trace()
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(answer_field="aliases")).score(
        task, trace
    )
    assert "Paris\nLutetia" in fake_judge_model[0]


async def test_reference_choices(fake_judge_model):
    # Verdict labels are configurable; the positive (first) label scores 1.0 and the
    # default prompt asks for the configured labels (not a hardcoded yes/no).
    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(choices=("A", "B")))
    assert judge.parse(JudgeResponse(text="A")) == 1.0
    assert judge.parse(JudgeResponse(text="Final verdict: B")) == 0.0
    with pytest.raises(ValueError, match="no A/B verdict"):
        judge.parse(JudgeResponse(text="gibberish"))
    trace = make_trace()
    with pytest.raises(
        ValueError
    ):  # the yes-replying fake is now an unparseable verdict
        await judge.score(trace.task.data, trace)
    assert 'Respond either "A" or "B"' in fake_judge_model[0]
    # degenerate labels are a config error (duplicates would score every verdict 1.0)
    for choices in (("yes", "yes"), ("A", "a"), ("", "no")):
        with pytest.raises(ValueError, match="two distinct, non-empty"):
            vf.ReferenceJudgeConfig(choices=choices)


async def test_error_attribution(monkeypatch, tmp_path):
    # The policy: a MODEL failure scores 0.0; a JUDGE failure errors the rollout (raises
    # out of Task.score) so training skips the sample instead of
    # punishing the model for a broken judge.
    async def gibberish_judge(self, **kwargs):
        return model_response("as an AI language model I cannot grade this")

    monkeypatch.setattr(AsyncCompletions, "create", gibberish_judge)
    taskset = JudgedTaskset(
        JudgedConfig.model_validate({"task": {"judges": [{"id": "reference"}]}})
    )
    # model failure: empty reply -> judge skipped, reward 0.0, NO error
    trace = make_trace(reply="")
    await JudgedTask(trace.task.data, taskset.config.task).score(trace, runtime=None)
    assert trace.rewards["reference"].score == 0.0
    # judge failure: unparseable verdict -> the rollout errors, no reward recorded
    trace = make_trace()
    with pytest.raises(vf.TaskError, match="no yes/no verdict"):
        await JudgedTask(trace.task.data, taskset.config.task).score(
            trace, runtime=None
        )
    assert "reference" not in trace.rewards
    assert trace.judge_calls[0].error is not None


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
    # the suffix check is case-insensitive: QUALITY.TOML is TOML, not JSON
    assert rubric_judge(tmp_path, suffix=".TOML").criteria == toml


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
    # negative/NaN/inf weights would invert a criterion or corrupt the weighted mean
    for weight in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="negative or non-finite"):
            _ = rubric_judge(tmp_path, weights={"mentions_paris": weight}).criteria


async def test_rubric_score(tmp_path, fake_judge_model):
    # verdicts: mentions_paris=1 (w=3), is_polite=0 (w=1) -> weighted mean 0.75, from ONE
    # judge call; each verdict lands as a `<name>/<criterion>` metric.
    judge = rubric_judge(tmp_path)
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 0.75
    assert trace.metrics == {"rubric/mentions_paris": 1.0, "rubric/is_polite": 0.0}
    assert len(trace.judge_calls) == 1  # one call for the whole rubric


async def test_rubric_verdict_mismatch_raises(tmp_path, monkeypatch):
    # A reply that doesn't verdict exactly the rubric's criteria is a judge failure: raise
    # (-> rollout error), don't guess or silently score 0.
    async def wrong_names(self, **kwargs):
        verdicts = {"verdicts": [{"name": "typo", "reason": "x", "verdict": "yes"}]}
        return model_response(json.dumps(verdicts))

    monkeypatch.setattr(AsyncCompletions, "create", wrong_names)
    judge = rubric_judge(tmp_path)
    trace = make_trace()
    with pytest.raises(ValueError, match="expected the batch"):
        await judge.score(trace.task.data, trace)
    assert trace.judge_calls[0].error is not None


CHOICES_TOML = '[[criteria]]\nname = "depth"\ntext = "How thorough?"\nchoices = ["none", "partial", "good"]\n'


async def test_rubric_choices_normalize(tmp_path, monkeypatch):
    # Ordered choices (worst→best) score by rank: "partial" of ["none","partial","good"] -> 0.5.
    async def graded(self, **kwargs):
        v = {"verdicts": [{"name": "depth", "reason": "r", "verdict": "partial"}]}
        return model_response(json.dumps(v))

    monkeypatch.setattr(AsyncCompletions, "create", graded)
    judge = rubric_judge(tmp_path, body=CHOICES_TOML, name="q")
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 0.5
    assert trace.metrics == {"q/depth": 0.5}


async def test_rubric_off_menu_answer_raises(tmp_path, monkeypatch):
    # A verdict that isn't one of the criterion's choices is a judge failure, not a 0.
    async def off_menu(self, **kwargs):
        v = {"verdicts": [{"name": "depth", "reason": "r", "verdict": "maybe"}]}
        return model_response(json.dumps(v))

    monkeypatch.setattr(AsyncCompletions, "create", off_menu)
    judge = rubric_judge(tmp_path, body=CHOICES_TOML)
    trace = make_trace()
    with pytest.raises(ValueError, match="expected one of"):
        await judge.score(trace.task.data, trace)


def test_rubric_choices_validation(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        _ = rubric_judge(
            tmp_path, body='[[criteria]]\nname = "x"\ntext = "t"\nchoices = ["only"]\n'
        ).criteria
    with pytest.raises(ValueError, match="duplicate options"):
        _ = rubric_judge(
            tmp_path,
            body='[[criteria]]\nname = "x"\ntext = "t"\nchoices = ["a", "a"]\n',
        ).criteria


async def test_rubric_reference_answer_optional(tmp_path, fake_judge_model):
    # off by default: no reference block in the prompt.
    t = make_trace()
    await rubric_judge(tmp_path).score(t.task.data, t)
    assert "Reference solution" not in fake_judge_model[-1]

    # answer_field set: the task's gold answer is shown to the judge.
    t = make_trace(answer="ZEBRA-GOLD")
    await rubric_judge(tmp_path, answer_field="answer").score(t.task.data, t)
    assert "Reference solution" in fake_judge_model[-1]
    assert "ZEBRA-GOLD" in fake_judge_model[-1]


class JudgedTask(vf.Task[QAData]):
    @vf.reward
    async def own(self, trace) -> float:
        return 0.25


class JudgedConfig(vf.TasksetConfig):
    pass


class JudgedTaskset(vf.Taskset[JudgedTask, JudgedConfig]):
    def load(self) -> list[JudgedTask]:
        return []


async def test_task_score_runs_plugged_judges(tmp_path, fake_judge_model):
    rubric = tmp_path / "rubric.toml"
    rubric.write_text(RUBRIC_TOML)
    cfg = JudgedConfig.model_validate(
        {
            "task": {
                "judges": [
                    {"id": "reference", "weight": 0.5},
                    {"id": "rubric", "path": str(rubric), "name": "quality"},
                ]
            }
        }
    )
    taskset = JudgedTaskset(cfg)
    trace = make_trace()
    await JudgedTask(trace.task.data, taskset.config.task).score(trace, runtime=None)
    assert trace.rewards["own"].score == 0.25  # decorated rewards still run
    assert trace.rewards["reference"] == vf.Reward(
        score=1.0, weight=0.5
    )  # raw score + weight, under the id-derived name
    assert (
        trace.rewards["quality"].score == 0.75
    )  # the rubric's aggregate, under its `name`
    assert len(trace.judge_calls) == 2  # every judge call recorded (rubric = one call)


async def test_task_without_judges_scores_as_before():
    trace = make_trace()
    await JudgedTask(trace.task.data).score(trace, runtime=None)
    assert trace.rewards == {"own": vf.Reward(score=0.25)}
