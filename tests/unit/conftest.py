"""Traces shared by the judge and trace unit tests: a one-turn Q/A trace and a multi-turn
trace with reasoning and a tool call."""

import pytest

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, UserMessage


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runtime",
        help="run test_runtimes.py against this runtime only: a type, or a runtime "
        "config as JSON, such as a custom runtime plugin",
    )


class QAData(vf.TaskData):
    answer: str = ""


def _make_trace(
    reply: str = "It is Paris.",
    answer: str = "Paris",
    task_cls: type[QAData] = QAData,
) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
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


def _make_full_trace() -> vf.Trace:
    """A multi-turn trace: user -> assistant (reasoning + tool call) -> tool -> final reply."""
    from verifiers.v1.types import ToolCall, ToolMessage

    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
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
                    provider_state=[{"type": "reasoning", "data": "SECRET STATE"}],
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


@pytest.fixture
def make_trace():
    return _make_trace


@pytest.fixture
def make_full_trace():
    return _make_full_trace


@pytest.fixture
def qa_data():
    return QAData
