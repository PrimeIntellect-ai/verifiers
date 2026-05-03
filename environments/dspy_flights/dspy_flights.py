from __future__ import annotations

import re

import verifiers.v1 as vf


ACTION_RE = re.compile(r"<action>\s*([A-Z_]+)\s*</action>", re.I)


@vf.reward(weight=1.0)
async def exact_action(task, state) -> float:
    return float(parse_action(state.get("answer", "")) == task["answer"])


@vf.metric
async def dspy_calls(task, state) -> float:
    return float(len(state.get("trajectory", [])))


def source():
    system = (
        "You are a flight customer-service assistant. Return only one XML tag: "
        "<action>BOOK_FLIGHT</action>, <action>CHANGE_FLIGHT</action>, "
        "<action>CANCEL_FLIGHT</action>, or <action>FILE_TICKET</action>."
    )
    rows = [
        {
            "request": ("Adam wants to book a flight from SFO to JFK on 09/01/2025."),
            "answer": "BOOK_FLIGHT",
        },
        {
            "request": (
                "Chelsie already has confirmation ABC123 and wants to cancel it."
            ),
            "answer": "CANCEL_FLIGHT",
        },
        {
            "request": (
                "Bob wants to move his existing SFO to JFK booking to a later "
                "flight on the same date."
            ),
            "answer": "CHANGE_FLIGHT",
        },
    ]
    for index, row in enumerate(rows):
        yield {
            "example_id": index,
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user", "content": row["request"]},
            ],
            **row,
        }


def parse_action(text: object) -> str:
    match = ACTION_RE.search(str(text))
    return match.group(1).upper() if match else ""


async def run_dspy_flight_program(task, state, client):
    import dspy

    class FlightSupport(dspy.Signature):
        """Classify a flight customer-service request into the requested XML action."""

        request: str = dspy.InputField()
        action: str = dspy.OutputField(
            desc=(
                "Exactly one of <action>BOOK_FLIGHT</action>, "
                "<action>CHANGE_FLIGHT</action>, <action>CANCEL_FLIGHT</action>, "
                "or <action>FILE_TICKET</action>."
            )
        )

    model = state["runtime"]["model"]
    lm = dspy.LM(
        f"openai/{model}",
        api_base=state["endpoint_base_url"],
        api_key="intercepted",
        cache=False,
    )
    predictor = dspy.Predict(FlightSupport)
    with dspy.context(lm=lm):
        result = await predictor.acall(request=task["request"])
    answer = str(result.action)
    state["answer"] = answer
    state["completion"] = [{"role": "assistant", "content": answer}]
    return state


def load_taskset(config=None):
    return vf.Taskset(
        source=source,
        rewards=[exact_action],
        metrics=[dspy_calls],
        config=config,
    )


def load_harness(config=None):
    return vf.Harness(
        program={"entrypoint": "dspy_flights:run_dspy_flight_program"},
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
