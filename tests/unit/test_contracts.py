"""Contracts: a reply parsed into a value, its errors by line or field, and the repair
turns `Agent.until_valid` makes until a reply meets the contract."""

import pytest
from pydantic import BaseModel, Field

import verifiers.v1 as vf
from verifiers.v1.contracts import ContractError, json_object, parse_markdown


class Verdict(BaseModel):
    ok: bool
    reasons: list[str] = Field(min_length=1)


def test_json_object_takes_a_fenced_reply_and_skips_an_echoed_example():
    assert json_object('```json\n{"ok": true}\n```') == {"ok": True}
    text = 'Shape: {"verdicts": [...]}\nAnswer: {"verdicts": [1]} trailing'
    assert json_object(text, key="verdicts") == {"verdicts": [1]}
    with pytest.raises(ContractError):
        json_object("no object here")


def test_a_json_contract_reports_errors_by_field():
    contract = vf.Contract.json(Verdict)
    assert contract.check('{"ok": true, "reasons": ["r"]}') == Verdict(
        ok=True, reasons=["r"]
    )
    with pytest.raises(ContractError) as broken:
        contract.check('{"ok": "maybe", "reasons": []}')
    fields = sorted(line.split(":")[0] for line in broken.value.errors)
    assert fields == ["ok", "reasons"]


class Meta(BaseModel):
    skills: list[str] = Field(min_length=1)
    base_image: str


def check_line(text: str) -> tuple[str, str]:
    severity, _, rest = text.partition(" ")
    if severity not in ("[critical]", "[standard]"):
        raise ValueError(f"unknown severity {severity!r}")
    return severity.strip("[]"), rest


DESIGN = """---
skills: [skill-a, skill-b]
base_image: bank/python
---
## Task
Parse logs.

## Checks
- [critical] skill-a/hurdle — output matches
- [standard] skill-b/other — runs fast
"""


def test_a_markdown_reply_parses_frontmatter_sections_and_items():
    doc = parse_markdown(
        DESIGN,
        headings=["Task", "Checks"],
        frontmatter=Meta,
        items={"Checks": check_line},
    )
    assert doc.frontmatter == Meta(
        skills=["skill-a", "skill-b"], base_image="bank/python"
    )
    assert doc.sections["Task"] == "Parse logs."
    assert [severity for severity, _ in doc.items["Checks"]] == ["critical", "standard"]


def test_a_broken_markdown_reply_reports_every_error_by_line():
    broken = DESIGN.replace("base_image: bank/python\n", "").replace(
        "[standard]", "[optional]"
    )
    with pytest.raises(ContractError) as found:
        parse_markdown(
            broken,
            headings=["Task", "Checks", "Crux"],
            frontmatter=Meta,
            items={"Checks": check_line},
        )
    errors = found.value.errors
    assert any(line.startswith("frontmatter.base_image") for line in errors)
    assert any(line.startswith("missing section") and "Crux" in line for line in errors)
    assert any(line.startswith("line 9:") for line in errors)  # the bad check line


class ScriptedAgent(vf.Agent):
    """Answers each run with the next scripted reply (None: a failed run)."""

    def __init__(self, replies: list[str | None]) -> None:
        super().__init__(vf.AgentConfig(model="test/fake-model"))
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def run(self, task=None, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        trace = vf.Trace(
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            task=vf.TraceTask(
                type="Task", data=vf.TaskData(idx=0, prompt=None), key="t", hash="h"
            ),
        )
        if reply is None:
            trace.errors.append(
                vf.Error(type="ProviderError", message="upstream 503", status_code=503)
            )
        else:
            trace.root_reply, trace.ok = reply, True
        return trace


@pytest.mark.asyncio
async def test_until_valid_repairs_in_the_same_conversation_with_the_errors_inserted(
    tmp_path,
):
    repair = tmp_path / "repair.txt"
    repair.write_text("Fix these:\n{errors}\n")
    agent = ScriptedAgent(['{"ok": 1}', '{"ok": true, "reasons": ["r"]}'])
    reply = await agent.until_valid(
        "judge it", vf.Contract.json(Verdict), attempts=3, repair=repair, key="judge"
    )
    assert reply.value == Verdict(ok=True, reasons=["r"])
    assert reply.attempts == 2
    first, second = agent.calls
    assert (first["key"], second["key"]) == ("judge", "judge.repair1")
    assert second["resume"] is reply.traces[0]
    assert second["note"].startswith("Fix these:\n- ")
    assert "reasons" in second["note"]


@pytest.mark.asyncio
async def test_until_valid_fails_the_role_after_its_attempts_and_types_a_failed_run():
    agent = ScriptedAgent(["nope", "still nope"])
    with pytest.raises(vf.ContractFailed) as failed:
        await agent.until_valid("judge it", vf.Contract.json(Verdict), attempts=2)
    assert failed.value.attempts == 2
    assert len(agent.calls) == 2
    with pytest.raises(vf.RunFailed) as lost:
        await ScriptedAgent([None]).until_valid("judge it", vf.Contract.json(Verdict))
    assert lost.value.error.status_code == 503
