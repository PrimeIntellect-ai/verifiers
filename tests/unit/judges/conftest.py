"""Fake chat and decision responses so judge tests need no network."""

import json
import re

import pytest

from verifiers.v1.judge import Judge, JudgeResponse


@pytest.fixture
def fake_judge_model(monkeypatch):
    """Fake the judge's model call, recording each prompt for assertions. Rubric calls (a JSON
    `verdicts` instruction or a `schema`) reply one reasoned verdict per `- name: text` criterion,
    "yes" iff it mentions Paris; other judges reply plain "yes"/"no" by the response block."""
    prompts: list[str] = []

    async def fake_complete(
        self, messages, *, trace=None, schema=None, parse=None, **sampling
    ):
        prompts.append(messages)
        # Rubric calls carry criteria lines + a JSON `verdicts` instruction; reply one verdict per
        # criterion (yes iff its text mentions Paris) with a reason. Other judges get plain yes/no.
        if schema is not None or '"verdicts"' in messages:
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
            response = JudgeResponse(
                text=json.dumps({"verdicts": verdicts}),
                parsed=schema.model_validate({"verdicts": verdicts})
                if schema
                else None,
            )
        else:
            response = JudgeResponse(
                text="yes" if "Paris" in messages.split("Response:")[-1] else "no"
            )
        if parse is not None:
            response.parsed = parse(response)
        if trace is not None:
            from verifiers.v1.dialects.chat import message_to_wire

            wire = (
                [{"role": "user", "content": messages}]
                if isinstance(messages, str)
                else [message_to_wire(message) for message in messages]
            )
            kwargs = {"model": self.config.model, "messages": wire}
            kwargs.update(sampling)
            trace.record_judge_call(
                name=self.reward_name,
                request={"model": kwargs["model"], "messages": kwargs["messages"]},
                response=response,
            )
        return response

    monkeypatch.setattr(Judge, "complete", fake_complete)

    class DecisionClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, endpoint, *, cast_to, body):
            prompt = body["state"]["prompt"]
            prompts.append(prompt)
            answers = {}
            for name, question in body["questions"].items():
                if name == "verdict":
                    choice = "yes" if "Paris" in prompt.split("Response:")[-1] else "no"
                else:
                    choice = "yes" if "Paris" in question["instructions"] else "no"
                answers[name] = {"type": "choice", "choice": choice}
            return {"answers": answers, "usage": {"inputTokens": 10, "outputTokens": 0}}

    monkeypatch.setattr(
        "verifiers.v1.judge.build_async_openai", lambda config: DecisionClient()
    )
    return prompts
