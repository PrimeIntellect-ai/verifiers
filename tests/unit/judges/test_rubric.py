"""The built-in `rubric` judge: criteria files, weights, verdict choices, and scoring."""

import json

import pytest

import verifiers.v1 as vf
from verifiers.v1.judge import Judge, JudgeResponse

RUBRIC_TOML = """
[[criteria]]
name = "mentions_paris"
text = "The response mentions Paris."
weight = 3.0

[[criteria]]
name = "is_polite"
text = "The response is polite."
"""


def rubric_judge(
    tmp_path, body: str = RUBRIC_TOML, suffix: str = ".toml", **kwargs
) -> vf.RubricJudge:
    path = tmp_path / f"rubric{suffix}"
    path.write_text(body)
    return vf.RubricJudge(vf.RubricJudgeConfig(path=str(path), **kwargs))


CHOICES_TOML = '[[criteria]]\nname = "depth"\ntext = "How thorough?"\nchoices = ["none", "partial", "good"]\n'


def test_rubric_config_requires_path():
    # `path` is a required Path field: a plugged rubric judge without one fails at config time.
    with pytest.raises(ValueError, match="path"):
        vf.TaskConfig.model_validate({"judges": [{"id": "rubric"}]})


async def test_rubric_view_full_trace(tmp_path, fake_judge_model, make_full_trace):
    # The rubric judge's default view: criteria are judged against the whole transcript.
    judge = rubric_judge(tmp_path)
    trace = make_full_trace()
    await judge.score(trace.task.data, trace)
    assert all("TOOL RESULT" in prompt for prompt in fake_judge_model)
    assert all("SECRET REASONING" not in prompt for prompt in fake_judge_model)


def test_rubric_criteria_toml_and_json(tmp_path):
    toml = rubric_judge(tmp_path).criteria
    assert [c.name for c in toml] == ["mentions_paris", "is_polite"]
    assert [c.weight for c in toml] == [3.0, 1.0]
    # JSON accepts a metadata-bearing object or a bare criteria list.
    items = [c.model_dump() for c in toml]
    assert (
        rubric_judge(
            tmp_path,
            json.dumps({"title": "Safety", "criteria": items}),
            ".json",
        ).criteria
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
        with pytest.raises(
            ValueError, match="greater than or equal to 0|finite number"
        ):
            _ = rubric_judge(tmp_path, weights={"mentions_paris": weight}).criteria
    with pytest.raises(ValueError, match="non-finite total"):
        _ = rubric_judge(
            tmp_path, weights={"mentions_paris": 1e308, "is_polite": 1e308}
        ).criteria


async def test_rubric_score(tmp_path, fake_judge_model, make_trace):
    # verdicts: mentions_paris=1 (w=3), is_polite=0 (w=1) -> weighted mean 0.75, from ONE
    # judge call; each verdict lands as a `<name>/<criterion>` metric.
    judge = rubric_judge(tmp_path)
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 0.75
    assert trace.metrics == {"rubric/mentions_paris": 1.0, "rubric/is_polite": 0.0}
    assert len(trace.info["judge_calls"]) == 1  # one call for the whole rubric


async def test_rubric_verdict_mismatch_raises(tmp_path, monkeypatch, make_trace):
    # A reply that doesn't verdict exactly the rubric's criteria is a judge failure: raise
    # (-> rollout error), don't guess or silently score 0.
    async def wrong_names(self, messages, *, trace=None, schema=None, parse=None, **s):
        verdicts = {"verdicts": [{"name": "typo", "reason": "x", "verdict": "yes"}]}
        return JudgeResponse(text=json.dumps(verdicts), parsed=None)

    monkeypatch.setattr(Judge, "complete", wrong_names)
    judge = rubric_judge(tmp_path)
    trace = make_trace()
    with pytest.raises(ValueError, match="expected the batch"):
        await judge.score(trace.task.data, trace)


async def test_rubric_choices_normalize(tmp_path, monkeypatch, make_trace):
    # Ordered choices (worst→best) score by rank: "partial" of ["none","partial","good"] -> 0.5.
    async def graded(self, messages, *, trace=None, schema=None, parse=None, **s):
        v = {"verdicts": [{"name": "depth", "reason": "r", "verdict": "partial"}]}
        return JudgeResponse(text=json.dumps(v), parsed=None)

    monkeypatch.setattr(Judge, "complete", graded)
    judge = rubric_judge(tmp_path, body=CHOICES_TOML, name="q")
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 0.5
    assert trace.metrics == {"q/depth": 0.5}


async def test_rubric_off_menu_answer_raises(tmp_path, monkeypatch, make_trace):
    # A verdict that isn't one of the criterion's choices is a judge failure, not a 0.
    async def off_menu(self, messages, *, trace=None, schema=None, parse=None, **s):
        v = {"verdicts": [{"name": "depth", "reason": "r", "verdict": "maybe"}]}
        return JudgeResponse(text=json.dumps(v), parsed=None)

    monkeypatch.setattr(Judge, "complete", off_menu)
    judge = rubric_judge(tmp_path, body=CHOICES_TOML)
    trace = make_trace()
    with pytest.raises(ValueError, match="expected one of"):
        await judge.score(trace.task.data, trace)


def test_rubric_choices_validation(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        _ = rubric_judge(
            tmp_path, body='[[criteria]]\nname = "x"\ntext = "t"\nchoices = ["only"]\n'
        ).criteria
    with pytest.raises(ValueError, match="duplicate options"):
        _ = rubric_judge(
            tmp_path,
            body='[[criteria]]\nname = "x"\ntext = "t"\nchoices = ["a", "a"]\n',
        ).criteria


async def test_rubric_reference_answer_optional(tmp_path, fake_judge_model, make_trace):
    # off by default: no reference block in the prompt.
    t = make_trace()
    await rubric_judge(tmp_path).score(t.task.data, t)
    assert "Reference solution" not in fake_judge_model[-1]

    # answer_field set: the task's gold answer is shown to the judge.
    t = make_trace(answer="ZEBRA-GOLD")
    await rubric_judge(tmp_path, answer_field="answer").score(t.task.data, t)
    assert "Reference solution" in fake_judge_model[-1]
    assert "ZEBRA-GOLD" in fake_judge_model[-1]


DROP_TOML = (
    '[[criteria]]\nname = "outcome"\ntext = "t"\n'
    'choices = { correct = 1.0, incorrect = 0.0 }\ndrop = ["task_defect"]\n'
)


async def test_rubric_drop_answer_drops_the_trace(tmp_path, monkeypatch, make_trace):
    # A drop answer records no reward and drops the trace with the answer as its reason.
    async def defect(self, messages, *, trace=None, schema=None, parse=None, **s):
        v = {"verdicts": [{"name": "outcome", "reason": "r", "verdict": "task_defect"}]}
        return JudgeResponse(text=json.dumps(v), parsed=None)

    monkeypatch.setattr(Judge, "complete", defect)
    judge = rubric_judge(tmp_path, body=DROP_TOML, name="q")
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) is None
    assert trace.dropped == "q: outcome=task_defect"
    assert trace.metrics == {
        "q/outcome/correct": 0.0,
        "q/outcome/incorrect": 0.0,
        "q/outcome/task_defect": 1.0,
    }


def test_rubric_drop_validation(tmp_path):
    def criteria(body: str):
        return rubric_judge(
            tmp_path, body='[[criteria]]\nname = "x"\ntext = "t"\n' + body
        ).criteria

    with pytest.raises(ValueError, match="finite"):
        criteria("choices = { a = 1.0, b = nan }\n")
    with pytest.raises(ValueError, match="both"):
        criteria('choices = { a = 1.0, b = 0.0 }\ndrop = ["b"]\n')
    with pytest.raises(ValueError, match="at least 2 answers"):
        criteria("choices = { a = 1.0 }\n")
    # one scored answer is enough next to a drop answer
    assert criteria('choices = { a = 1.0 }\ndrop = ["b"]\n')[0].answers == ["a", "b"]
