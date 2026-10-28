"""Task-declared Harbor criteria: the file, test and judge decisions, the fold, and what
lands on the trace. The judge provider is an in-process `httpx.MockTransport`."""

import base64
import io
import json
import re
import zipfile
from typing import ClassVar

import httpx
import openai
import pytest

import verifiers.v1 as vf
from verifiers.v1.clients import base as client_base
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.tasksets.harbor.criteria import (
    CriterionResult,
    TaskCriteria,
    TaskDecisionJudgeConfig,
    TaskRubricJudgeConfig,
    fold,
    load_task_criteria,
    matches,
    render,
)
from verifiers.v1.tasksets.harbor.taskset import (
    WITHOUT_SYS_RESOURCE,
    HarborData,
    HarborTask,
    HarborTaskConfig,
)

TEST_SH = ["sh", "-c", WITHOUT_SYS_RESOURCE, "vf-verifier", "bash", "/tests/test.sh"]

JUNIT = b"""<?xml version="1.0"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.test_outputs" name="test_parses"/>
  <testcase classname="tests.test_outputs" name="test_tail[short]"/>
  <testcase classname="tests.test_outputs" name="test_tail[long]">
    <failure message="assert 1 == 2"/>
  </testcase>
  <testcase classname="tests.test_outputs.TestApi" name="test_roundtrip"/>
  <testcase classname="tests.test_outputs" name="test_flaky"><skipped/></testcase>
</testsuite></testsuites>
"""


class FakeRuntime:
    """A grading box: test.sh runs, and `files` are what it holds afterwards."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files

    async def run(self, argv, env):
        if argv == TEST_SH:
            return ProgramResult(exit_code=0, stdout="", stderr="")
        if "find /logs/verifier" in argv[-1]:
            return ProgramResult(exit_code=0, stdout="", stderr="")
        if "/logs/agent" in argv[-1] or argv[0] == "chmod":
            return ProgramResult(exit_code=0, stdout="", stderr="")  # agent logs staged
        if argv[:2] == ["sh", "-c"] and argv[2].startswith("test "):
            return ProgramResult(
                exit_code=0 if argv[-1] in self.files else 1, stdout="", stderr=""
            )
        raise AssertionError(f"unexpected command {argv}")

    async def read(self, path, max_bytes=None):
        if path not in self.files:
            raise SandboxError(f"{path} not found")
        return self.files[path]


@pytest.fixture
def provider(monkeypatch):
    """The judge provider. `provider.answer(body)` replies to each request body (a
    decision request, or a chat completion); every body is kept in `requests`."""

    class Provider:
        requests: ClassVar[list[dict]] = []
        answer = None

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        Provider.requests.append(body)
        reply = Provider.answer(body)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    monkeypatch.setattr(
        client_base,
        "build_async_httpx",
        lambda config: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    Provider.requests = []
    return Provider


def decision(score_for):
    """A decision provider scoring each question by `score_for(instructions, state)`."""

    def answer(body):
        question = body["questions"]["grade"]
        score = score_for(question["instructions"], body["state"])
        return {
            "answers": {"grade": {"score": score * (len(question["criteria"]) - 1)}}
        }

    return answer


def chat(verdict_for):
    """A chat provider answering each rubric criterion line with `verdict_for(name, text)`."""

    def answer(body):
        prompt = body["messages"][0]["content"]
        verdicts = [
            {"name": name, "reason": "because", "verdict": verdict_for(name, text)}
            for name, text in re.findall(r"^- ([^:]+): (.+)$", prompt, re.MULTILINE)
        ]
        return {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"verdicts": verdicts}),
                    },
                }
            ],
        }

    return answer


CRITERIA = {
    "criteria": [
        {
            "id": "parses",
            "text": "Parses the records.",
            "kind": "test",
            "tests": ["test_outputs.py::test_parses", "api_ok"],
            "critical": True,
        },
        {
            "id": "tail",
            "text": "Handles every truncated tail.",
            "kind": "test",
            "tests": ["test_tail"],
            "weight": 2.0,
        },
        {
            "id": "report",
            "text": "The report explains the cache invalidation.",
            "kind": "judge",
            "evidence": ["/app/REPORT.md", "/app/plot.png"],
        },
        {
            "id": "migration",
            "text": "Any schema migration is reversible.",
            "kind": "judge",
            "evidence": ["/app/REPORT.md"],
            "conditional": True,
        },
    ]
}

BOX = {
    "/logs/verifier/reward.json": b'{"reward": 0.25, "api_ok": 1}',
    "/logs/verifier/junit.xml": JUNIT,
    "/app/REPORT.md": b"The cache is invalidated on write because readers...",
    "/app/plot.png": b"\x89PNG fake",
}


def make_task(criteria=CRITERIA, judge=None) -> HarborTask:
    config = HarborTaskConfig(criteria_judge=judge or TaskDecisionJudgeConfig())
    data = HarborData(
        idx=0,
        prompt="Fix the parser.",
        task_dir="/nonexistent",
        criteria=TaskCriteria.model_validate(criteria),
    )
    return HarborTask(data, config)


def make_trace(task: HarborTask) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="HarborTask", data=task.data),
    )


def explains_and_no_migration(instructions, state):
    if "applies" in instructions:
        return 0.1  # no migration happened: not applicable
    return 0.9 if "cache invalidation" in instructions else 0.0


async def test_criteria_decide_fold_and_land_on_the_trace(provider):
    provider.answer = decision(explains_and_no_migration)
    task = make_task()
    trace = make_trace(task)

    scores = await task.run_verifier(FakeRuntime(dict(BOX)), trace)

    # parses (critical) is met; tail fails one parametrization; report met; migration n/a.
    assert scores == {"reward": 0.0}
    assert trace.metrics["criteria_score"] == pytest.approx(1 / 3)
    assert trace.metrics["verifier_reward"] == 0.25
    assert trace.metrics["criteria/parses"] == 1.0
    assert trace.metrics["criteria/tail"] == 0.0
    assert trace.metrics["criteria/report"] == 1.0
    assert "criteria/migration" not in trace.metrics
    assert trace.metrics["criteria/migration/not applicable"] == 1.0
    report = trace.info["criteria"]
    assert report["passed"] is False and report["critical_unmet"] == []
    by_id = {c["id"]: c for c in report["criteria"]}
    assert by_id["tail"]["tests"] == {"test_tail": "failed"}
    assert by_id["parses"]["tests"] == {
        "test_outputs.py::test_parses": "passed",
        "api_ok": "passed",
    }
    assert by_id["migration"]["answer"] == "not applicable"
    # The image went to the decision judge as an image; the report as text.
    report_calls = [r for r in provider.requests if "cache" in r["state"]["criterion"]]
    png = base64.b64encode(BOX["/app/plot.png"]).decode()
    assert report_calls[0]["images"] == [f"data:image/png;base64,{png}"]
    assert "invalidated on write" in report_calls[0]["state"]["evidence"]
    # One call for report; migration's applicability call decided it, so no second.
    assert len(trace.info["judge_calls"]) == 2


async def test_an_unmet_critical_criterion_zeroes_the_reward(provider):
    provider.answer = decision(lambda instructions, state: 1.0)
    task = make_task()
    box = dict(BOX, **{"/logs/verifier/reward.json": b'{"api_ok": 0}'})
    trace = make_trace(task)
    assert await task.run_verifier(FakeRuntime(box), trace) == {"reward": 0.0}
    assert trace.info["criteria"]["critical_unmet"] == ["parses"]
    assert trace.metrics["criteria_score"] == 0.0


async def test_a_judge_failure_errors_the_rollout_instead_of_scoring(provider):
    provider.answer = lambda body: httpx.Response(400, json={"error": "bad"})
    task = make_task()
    trace = make_trace(task)
    with pytest.raises(openai.BadRequestError):
        await task.run_verifier(FakeRuntime(dict(BOX)), trace)
    assert "criteria" not in trace.info


async def test_a_missing_junit_report_errors_the_rollout(provider):
    task = make_task({"criteria": [CRITERIA["criteria"][1]]})
    box = {"/logs/verifier/reward.json": b'{"reward": 1}'}
    with pytest.raises(TaskError, match="JUnit"):
        await task.run_verifier(FakeRuntime(box), make_trace(task))


async def test_a_criteria_task_needs_no_reward_file(provider):
    task = make_task({"pass_threshold": 0.5, "criteria": CRITERIA["criteria"][1:2]})
    junit = JUNIT.replace(b'<failure message="assert 1 == 2"/>', b"")
    trace = make_trace(task)
    box = {"/logs/verifier/junit.xml": junit}
    assert await task.run_verifier(FakeRuntime(box), trace) == {"reward": 1.0}
    assert trace.metrics["criteria_score"] == 1.0


async def test_the_rubric_judge_answers_in_one_call_and_na_only_when_conditional(
    provider,
):
    provider.answer = chat(
        lambda name, text: "not applicable" if name == "migration" else "yes"
    )
    criteria = {"criteria": [CRITERIA["criteria"][0], *CRITERIA["criteria"][2:]]}
    criteria["criteria"][1] = {
        **criteria["criteria"][1],
        "evidence": ["/app/REPORT.md"],
    }
    judge = TaskRubricJudgeConfig(model="some/model")
    task = make_task(criteria, judge)
    trace = make_trace(task)

    assert await task.run_verifier(FakeRuntime(dict(BOX)), trace) == {"reward": 1.0}
    assert len(provider.requests) == 1
    prompt = provider.requests[0]["messages"][0]["content"]
    assert "invalidated on write" in prompt  # the evidence, read from the box
    assert trace.metrics["criteria/report"] == 1.0
    assert trace.metrics["criteria/migration/not applicable"] == 1.0
    assert {c["id"]: c["answer"] for c in trace.info["criteria"]["criteria"]} == {
        "parses": "yes",
        "report": "yes",
        "migration": "not applicable",
    }


async def test_a_rubric_judge_answering_na_for_an_unconditional_criterion_errors(
    provider,
):
    provider.answer = chat(lambda name, text: "not applicable")
    criteria = {
        "criteria": [{**CRITERIA["criteria"][2], "evidence": ["/app/REPORT.md"]}]
    }
    task = make_task(criteria, TaskRubricJudgeConfig(model="some/model"))
    with pytest.raises(ValueError, match="not applicable"):
        await task.run_verifier(FakeRuntime(dict(BOX)), make_trace(task))


async def test_the_rubric_judge_refuses_image_evidence(provider):
    task = make_task(
        {"criteria": [CRITERIA["criteria"][2]]}, TaskRubricJudgeConfig(model="m")
    )
    with pytest.raises(TaskError, match="image"):
        await task.run_verifier(FakeRuntime(dict(BOX)), make_trace(task))
    assert provider.requests == []


async def test_missing_evidence_is_shown_to_the_judge(provider):
    provider.answer = decision(lambda instructions, state: 0.0)
    task = make_task({"criteria": [CRITERIA["criteria"][2]]})
    box = {"/logs/verifier/reward.json": b"{}"}
    trace = make_trace(task)
    assert await task.run_verifier(FakeRuntime(box), trace) == {"reward": 0.0}
    assert "does not exist" in provider.requests[0]["state"]["evidence"]


def result(id, answer, weight=1.0, critical=False):
    return CriterionResult(
        id=id,
        kind="judge",
        critical=critical,
        weight=weight,
        conditional=True,
        answer=answer,
    )


def test_fold():
    assert fold([result("a", "yes", critical=True), result("b", "no")], 1).score == 0
    assert fold([result("a", "no", critical=True), result("b", "yes")], 1).score == 0
    mixed = fold(
        [result("a", "yes", 3), result("b", "no"), result("c", "not applicable")], 0.75
    )
    assert mixed.score == 0.75 and mixed.passed
    assert fold([result("a", "yes", critical=True)], 1).score == 1.0
    assert fold([result("a", "not applicable", critical=True)], 1).critical_unmet == []


@pytest.mark.parametrize(
    "selector,case,expected",
    [
        ("test_parses", "tests.test_outputs.test_parses", True),
        ("tests/test_outputs.py::test_parses", "tests.test_outputs.test_parses", True),
        ("test_outputs.test_parses", "tests.test_outputs.test_parses", True),
        ("test_tail", "tests.test_outputs.test_tail[long]", True),
        ("test_tail[short]", "tests.test_outputs.test_tail[long]", False),
        ("TestApi::test_roundtrip", "tests.test_outputs.TestApi.test_roundtrip", True),
        ("parses", "tests.test_outputs.test_parses", False),
    ],
)
def test_selectors(selector, case, expected):
    assert matches(selector, case) is expected


def test_the_criteria_file_loads_from_tests_and_is_checked(tmp_path):
    assert load_task_criteria(tmp_path) is None
    (tmp_path / "tests").mkdir()
    path = tmp_path / "tests" / "criteria.toml"
    path.write_text(
        'pass_threshold = 0.8\n[[criteria]]\nid = "a"\ntext = "t"\nkind = "judge"\n'
        'evidence = ["/app/out.txt"]\nconditional = true\n'
    )
    criteria = load_task_criteria(tmp_path)
    assert criteria.pass_threshold == 0.8 and criteria.criteria[0].conditional
    path.write_text('[[criteria]]\nid = "a"\ntext = "t"\nkind = "judge"\n')
    with pytest.raises(ValueError, match="evidence"):
        load_task_criteria(tmp_path)
    path.write_text(
        '[[criteria]]\nid = "a"\ntext = "t"\nkind = "test"\ntests = ["x"]\n'
        "conditional = true\n"
    )
    with pytest.raises(ValueError, match="conditional"):
        load_task_criteria(tmp_path)


def test_a_task_without_criteria_keeps_its_data():
    data = HarborData(prompt="p", task_dir="/x")
    assert "criteria" not in data.model_dump()


def test_the_criteria_judge_resolves_by_id():
    rubric = HarborTaskConfig.model_validate(
        {"criteria_judge": {"id": "rubric", "model": "m", "max_criteria": 2}}
    )
    assert isinstance(rubric.criteria_judge, TaskRubricJudgeConfig)
    assert rubric.criteria_judge.max_criteria == 2
    restored = HarborTaskConfig.model_validate_json(rubric.model_dump_json())
    assert isinstance(restored.criteria_judge, TaskRubricJudgeConfig)
    assert isinstance(HarborTaskConfig().criteria_judge, TaskDecisionJudgeConfig)
    with pytest.raises(ValueError, match="criteria_judge"):
        HarborTaskConfig.model_validate({"criteria_judge": {"id": "reference"}})


def office(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def test_documents_and_spreadsheets_render_as_text():
    docx = office(
        {
            "word/document.xml": f"<w:document {W}><w:body>"
            "<w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>world</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Second</w:t></w:r></w:p></w:body></w:document>"
        }
    )
    assert render("/app/a.docx", docx).text == "Hello world\nSecond"
    xlsx = office(
        {
            "xl/workbook.xml": f'<workbook {S} {R}><sheets><sheet name="Q3" r:id="rId1"/>'
            "</sheets></workbook>",
            "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.'
            'openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
            "xl/sharedStrings.xml": f"<sst {S}><si><t>revenue</t></si></sst>",
            "xl/worksheets/sheet1.xml": f'<worksheet {S}><sheetData><row><c t="s">'
            "<v>0</v></c><c><v>42.5</v></c></row></sheetData></worksheet>",
        }
    )
    assert render("/app/b.xlsx", xlsx).text == "## sheet Q3\nrevenue\t42.5"
    assert render("/app/c.bin", b"\0\1\2").text == "(binary file, 3 bytes)"
    assert "more characters" in render("/app/d.txt", b"x" * 200_000).text
