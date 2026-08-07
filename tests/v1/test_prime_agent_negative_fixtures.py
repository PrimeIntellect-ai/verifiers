"""Prime Agent negative fixtures export exactly one taskset and environment."""

import importlib

import verifiers.v1 as vf
from verifiers.v1.utils.loaders import _plugin_class

FIXTURES = {
    "prime_agent_negatives_v1": (
        "PrimeAgentKilledChildTaskset",
        "PrimeAgentKilledChildEnv",
    ),
    "prime_agent_failing_gate_v1": (
        "PrimeAgentFailingGateTaskset",
        "PrimeAgentFailingGateEnv",
    ),
}


def test_negative_fixture_exports_are_unambiguous() -> None:
    """Each taskset id resolves its own taskset and bundled environment."""
    for module_name, (taskset_name, env_name) in FIXTURES.items():
        module = importlib.import_module(module_name)
        assert _plugin_class(module, vf.Taskset, "taskset").__name__ == taskset_name
        assert _plugin_class(module, vf.Env, "environment").__name__ == env_name
