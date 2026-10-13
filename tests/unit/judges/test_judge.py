"""Pluggable judges: plugin resolution, config narrowing, views, and how `Task.score` runs them.
Judge model calls are faked at `Judge.complete`; no network."""

import pytest
from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.envs.agentic_judge import ScoreConfig
from verifiers.v1.judge import Judge, JudgeResponse
from verifiers.v1.utils.loaders import judge_class, judge_config_type, load_judge

RUBRIC_TOML = """
[[criteria]]
name = "mentions_paris"
text = "The response mentions Paris."
weight = 3.0

[[criteria]]
name = "is_polite"
text = "The response is polite."
"""


class JudgedTask(vf.Task):
    @vf.reward
    async def own(self, trace) -> float:
        return 0.25


class JudgedConfig(vf.TasksetConfig):
    pass


class JudgedTaskset(vf.Taskset[JudgedTask, JudgedConfig]):
    def load(self) -> list[JudgedTask]:
        return []


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
    # name > id > snake-cased class name (code-level judge with neither).
    class MyQualityJudge(vf.Judge[float]):
        prompt = "{x}"

    assert (
        vf.ReferenceJudge().reward_name == "reference"
    )  # class-name fallback (no id set)
    assert (
        vf.ReferenceJudge(vf.ReferenceJudgeConfig(id="my-judge")).reward_name
        == "my-judge"
    )
    assert vf.ReferenceJudge(vf.ReferenceJudgeConfig(name="gold")).reward_name == "gold"
    assert MyQualityJudge().reward_name == "my_quality"


async def test_base_judge_score_raises(make_trace, qa_data):
    with pytest.raises(NotImplementedError, match="implements no `score`"):
        await vf.Judge().score(task=qa_data(idx=0, prompt="q"), trace=make_trace())


async def test_view_modes(fake_judge_model, make_full_trace):
    # last_reply (default): the judge sees only the final reply.
    trace = make_full_trace()
    await vf.ReferenceJudge().score(trace.task.data, trace)
    assert "TOOL RESULT" not in fake_judge_model[0]
    # full_trace: the whole transcript (minus reasoning) fills {response}.
    trace = make_full_trace()
    await vf.ReferenceJudge(vf.ReferenceJudgeConfig(view="full_trace")).score(
        trace.task.data, trace
    )
    assert "TOOL RESULT: Paris is the capital." in fake_judge_model[1]
    assert "SECRET REASONING" not in fake_judge_model[1]


async def test_config_prompt_overrides_class_template(
    tmp_path, fake_judge_model, make_trace
):
    # Config `prompt` is a file path; the same {field} placeholders work.
    file = tmp_path / "judge.txt"
    file.write_text("Q:{question} A:{answer} R:{response}")
    judge = vf.ReferenceJudge(vf.ReferenceJudgeConfig(prompt=file))
    assert judge.build_messages(question="q", answer="a", response="r") == "Q:q A:a R:r"
    # A template needn't use every evaluate field: score also passes {positive}/{negative},
    # which str.format ignores when the (custom) prompt doesn't reference them.
    trace = make_trace()
    assert await judge.score(trace.task.data, trace) == 1.0
    assert fake_judge_model[0] == "Q:Capital of France? A:Paris R:It is Paris."
    # a bad path fails at judge construction, not mid-eval at score time
    with pytest.raises(FileNotFoundError):
        vf.ReferenceJudge(vf.ReferenceJudgeConfig(prompt=tmp_path / "missing.txt"))


async def test_error_attribution(monkeypatch, tmp_path, make_trace):
    # The policy: a MODEL failure scores 0.0; a JUDGE failure errors the rollout (raises
    # out of Task.score as a TaskError) so training skips the sample instead of
    # punishing the model for a broken judge.
    async def gibberish_judge(
        self, messages, *, trace=None, schema=None, parse=None, **s
    ):
        response = JudgeResponse(text="as an AI language model I cannot grade this")
        try:
            if parse is not None:
                response.parsed = parse(response)
            return response
        finally:
            if trace is not None:
                trace.record_judge_call(
                    name=self.reward_name,
                    request={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": messages}],
                    },
                    response=response,
                )

    monkeypatch.setattr(Judge, "complete", gibberish_judge)
    taskset = JudgedTaskset(
        JudgedConfig.model_validate({"task": {"judges": [{"id": "reference"}]}})
    )
    # model failure: empty reply -> judge skipped, reward 0.0, NO error
    trace = make_trace(reply="")
    await JudgedTask(trace.task.data, taskset.config.task).score(trace, runtime=None)
    assert trace.rewards["reference"].score == 0.0
    assert "judge_calls" not in trace.info  # the (foregone) judge call was never made
    # judge failure: unparseable verdict -> the rollout errors, no reward recorded
    trace = make_trace()
    with pytest.raises(vf.TaskError, match="no yes/no verdict"):
        await JudgedTask(trace.task.data, taskset.config.task).score(
            trace, runtime=None
        )
    assert trace.rewards["reference"] is None  # seeded: expected but never scored
    assert len(trace.info["judge_calls"]) == 1  # the billed call is still recorded


def test_judge_composition_weights_are_finite():
    for weight in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite number"):
            vf.JudgeConfig(weight=weight)
        for field in ("task_weight", "judge_weight"):
            with pytest.raises(ValueError, match="finite number"):
                ScoreConfig.model_validate({field: weight})


async def test_task_score_runs_plugged_judges(tmp_path, fake_judge_model, make_trace):
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
    assert trace.rewards["own"] == vf.Reward(score=0.25)  # decorated rewards still run
    assert trace.rewards["reference"] == vf.Reward(
        score=1.0, weight=0.5
    )  # raw score + weight, under the id-derived name
    assert (
        trace.rewards["quality"].score == 0.75
    )  # the rubric's aggregate, under its `name`
    assert (
        len(trace.info["judge_calls"]) == 2
    )  # every judge call recorded (rubric = one call)
