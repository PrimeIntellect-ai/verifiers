"""Every checked-in v1 eval config parses, and resolved configs get stable output paths.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
`vf-eval` / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import re
import tomllib
from pathlib import Path

import pytest

from verifiers.v1.cli.output import output_path
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.envs.single_agent import SingleAgentEnvConfig

NULL_HARNESS = HarnessConfig(id="null")


class AdversarialEnvConfig(EnvConfig):
    attacker: AgentConfig = AgentConfig(harness=NULL_HARNESS)
    target: AgentConfig = AgentConfig(harness=NULL_HARNESS)


class MultipleAttackersEnvConfig(EnvConfig):
    attacker_1: AgentConfig = AgentConfig(harness=NULL_HARNESS)
    attacker_2: AgentConfig = AgentConfig(harness=NULL_HARNESS)
    target: AgentConfig = AgentConfig(harness=NULL_HARNESS)


CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    assert config.env.taskset.id


def test_output_path_uses_single_role_run_model() -> None:
    config = EvalConfig(
        env=SingleAgentEnvConfig(agent=AgentConfig(harness=NULL_HARNESS)),
        model="openai/gpt-5",
        uuid="run-id",
    )

    assert output_path(config) == Path("outputs/no-taskset--openai--gpt-5--null/run-id")


def test_output_path_uses_single_role_model_override() -> None:
    config = EvalConfig(
        env=SingleAgentEnvConfig(
            agent=AgentConfig(model="anthropic/claude-opus-5", harness=NULL_HARNESS)
        ),
        model="unused/fallback",
        uuid="run-id",
    )

    assert output_path(config) == Path(
        "outputs/no-taskset--anthropic--claude-opus-5--null/run-id"
    )


def test_output_path_names_manual_and_automated_roles() -> None:
    env = AdversarialEnvConfig(
        attacker=AgentConfig(model="manual-attacker", harness=NULL_HARNESS),
        target=AgentConfig(model="anthropic/claude-opus-5", harness=NULL_HARNESS),
    )
    config = EvalConfig.model_construct(env=env, model="unused/fallback", uuid="run-id")

    assert env.agent_models(config.model) == {
        "attacker": "manual-attacker",
        "target": "anthropic/claude-opus-5",
    }
    assert output_path(config) == Path(
        "outputs/no-taskset--attacker=manual-attacker+"
        "target=anthropic--claude-opus-5--null/run-id"
    )


def test_output_path_keeps_roles_with_repeated_models() -> None:
    config = EvalConfig.model_construct(
        env=AdversarialEnvConfig(),
        model="deepseek/deepseek-v4-flash",
        uuid="run-id",
    )

    assert output_path(config) == Path(
        "outputs/no-taskset--attacker=deepseek--deepseek-v4-flash+"
        "target=deepseek--deepseek-v4-flash--null/run-id"
    )


def test_output_path_names_three_roles_in_declaration_order() -> None:
    config = EvalConfig.model_construct(
        env=MultipleAttackersEnvConfig(
            attacker_1=AgentConfig(model="openai/gpt-5", harness=NULL_HARNESS),
            attacker_2=AgentConfig(
                model="anthropic/claude-opus-5", harness=NULL_HARNESS
            ),
        ),
        model="deepseek/deepseek-v4-flash",
        uuid="run-id",
    )

    assert output_path(config) == Path(
        "outputs/no-taskset--attacker_1=openai--gpt-5+"
        "attacker_2=anthropic--claude-opus-5+"
        "target=deepseek--deepseek-v4-flash--null/run-id"
    )


def test_output_path_encodes_reserved_model_characters() -> None:
    config = EvalConfig(
        env=SingleAgentEnvConfig(
            agent=AgentConfig(
                model="provider/model+variant=one% two\\three\n",
                harness=NULL_HARNESS,
            )
        ),
        uuid="run-id",
    )

    assert output_path(config) == Path(
        "outputs/no-taskset--provider--model%2Bvariant%3Done%25%20two%5Cthree%0A--"
        "null/run-id"
    )


def test_output_path_bounds_long_parent_component_deterministically() -> None:
    first = EvalConfig(
        env=SingleAgentEnvConfig(
            agent=AgentConfig(model=f"provider/{'x' * 300}", harness=NULL_HARNESS)
        ),
        uuid="run-id",
    )
    same = first.model_copy(deep=True)
    changed = EvalConfig(
        env=SingleAgentEnvConfig(
            agent=AgentConfig(model=f"provider/{'x' * 299}y", harness=NULL_HARNESS)
        ),
        uuid="run-id",
    )

    parent = output_path(first).parent.name
    assert len(parent.encode()) == 200
    assert re.fullmatch(r".+--[0-9a-f]{16}", parent)
    assert parent.endswith("--d79566b83a5020a7")
    assert output_path(same).parent.name == parent
    assert output_path(changed).parent.name != parent


def test_output_path_preserves_explicit_output_dir() -> None:
    explicit = Path("custom/results")
    config = EvalConfig.model_construct(
        env=AdversarialEnvConfig(),
        model="deepseek/deepseek-v4-flash",
        output_dir=explicit,
    )

    assert output_path(config) == explicit
