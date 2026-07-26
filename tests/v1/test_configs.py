"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
`vf-eval` / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import tomllib
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.cli.eval import EvalConfig
from verifiers.v1.envs.sandbox_judge import SandboxJudgeEnvConfig
from verifiers.v1.runtimes import DockerConfig, ModalConfig, SubprocessConfig

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    # resolved to a v1 taskset or a v0 env id
    assert config.env.taskset.id or config.id


@pytest.mark.parametrize(
    "runtime",
    [
        SubprocessConfig(),
        DockerConfig(),
        DockerConfig(allow=["attacker.example"]),
        DockerConfig(block=["example.com"]),
        ModalConfig(network_access=True),
        ModalConfig(network_access=False),
    ],
    ids=[
        "subprocess",
        "docker-unrestricted",
        "docker-partial-allow",
        "docker-partial-block",
        "modal-with-network",
        "modal-without-model-route",
    ],
)
def test_sandbox_judge_requires_isolation(runtime) -> None:
    with pytest.raises(ValueError, match="judge|network"):
        SandboxJudgeEnvConfig(judge=vf.AgentConfig(runtime=runtime))


def test_sandbox_judge_rejects_host_solver() -> None:
    with pytest.raises(TypeError, match="solver.*subprocess"):
        SandboxJudgeEnvConfig(
            solver=vf.AgentConfig(runtime=SubprocessConfig()),
        )


def test_sandbox_judge_score_weights_are_finite() -> None:
    with pytest.raises(ValueError, match="finite number"):
        SandboxJudgeEnvConfig(score={"task_weight": float("inf")})
