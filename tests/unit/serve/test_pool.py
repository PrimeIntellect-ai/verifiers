"""The env server ships its env config to workers as plain data: `env_config_data`
dumps it and `resolve_env_config` rebuilds the env's own config class, role-typed seats
included."""

import pytest

from verifiers.v1.serve import env_config_data
from verifiers.v1.utils.loaders import resolve_env_config


@pytest.mark.parametrize(
    "data",
    [
        {"taskset": {"id": "kuhn-poker"}, "player0": {"max_turns": 3}},
        {
            "id": "best-of-n",
            "n": 2,
            "taskset": {"id": "reverse-text"},
            "agent": {"harness": {"id": "null"}, "max_turns": 2},
        },
    ],
    ids=["role-typed-seats", "bundled-env"],
)
def test_env_config_survives_the_wire(data: dict) -> None:
    config = resolve_env_config(data)
    rebuilt = resolve_env_config(env_config_data(config))
    assert type(rebuilt) is type(config)
    assert rebuilt == config
