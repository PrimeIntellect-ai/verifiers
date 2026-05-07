from environments.langchain_deep_agents_env.langchain_deep_agents_env import (
    calculate as langchain_calculate,
)
from environments.openai_agents_env.openai_agents_env import (
    calculate as openai_calculate,
)


def test_agent_calculate_allows_arithmetic() -> None:
    assert openai_calculate("2 + 3 * 4") == "14"
    assert langchain_calculate("2 ** 8") == "256"


def test_agent_calculate_rejects_python_introspection() -> None:
    payload = "().__class__.__base__.__subclasses__()"

    assert openai_calculate(payload).startswith("Error:")
    assert langchain_calculate(payload).startswith("Error:")
