"""Every checked-in v1 eval config parses.

Mirrors prime-rl's config test: glob the configs and assert each validates into its config
type. The root `configs/*.toml` are the `uv run eval @ <file>` v1 configs (EvalConfig);
`endpoints.toml` isn't an eval config, and `configs/eval|rl|gepa/` are the legacy
`vf-eval` / training formats (different, non-v1 config classes), so both are out of scope here.
"""

import tomllib
from pathlib import Path

import pytest

from verifiers.v1.configs.eval import EvalConfig

CONFIGS = sorted(
    p
    for p in (Path(__file__).resolve().parents[2] / "configs").glob("*.toml")
    if p.name != "endpoints.toml"
)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_eval_config_parses(path: Path) -> None:
    config = EvalConfig.model_validate(tomllib.load(path.open("rb")))
    # resolved to a v1 taskset or a v0 env id
    assert (config.env.taskset is not None and config.env.taskset.id) or config.id


def test_output_path_compounds_env_id():
    """The default output dir carries the run's full identity: pairing `--env.id`
    prefixes the taskset (same compounding as `EnvConfig.env_id`), so a best-of-n
    run never shares a parent dir with the plain run of the same taskset."""
    from verifiers.v1.cli.output import output_path

    plain = EvalConfig(env={"taskset": {"id": "echo-v1"}})
    paired = EvalConfig(env={"id": "best-of-n", "taskset": {"id": "echo-v1"}})
    assert output_path(plain).parent.name.startswith("echo-v1--")
    assert output_path(paired).parent.name.startswith("best-of-n+echo-v1--")


def test_replay_lifts_the_saved_eval_taskset():
    """Replay layers the source run's saved config as its base. An eval run keeps
    its taskset on the [env] block — replay's root taskset must pick it up; an
    earlier replay's own root taskset stays authoritative."""
    from verifiers.v1.configs.replay import ReplayConfig

    lifted = ReplayConfig.model_validate(
        {"env": {"taskset": {"id": "echo-v1"}, "max_turns": 2}, "model": "m"}
    )
    assert lifted.taskset.id == "echo-v1"
    rooted = ReplayConfig.model_validate(
        {"taskset": {"id": "echo-v1"}, "env": {"taskset": {"id": "other-v1"}}}
    )
    assert rooted.taskset.id == "echo-v1"


async def test_replay_refuses_multi_agent_runs(tmp_path):
    """Multi-agent runs don't support replay: offline re-scoring runs per trace and
    can't re-run the env's cross-trace score(). The saved config names the env, so
    the refusal is up front — before any episode is read."""
    import pytest

    from verifiers.v1.cli.output import save_config
    from verifiers.v1.cli.replay import run_replay
    from verifiers.v1.configs.replay import ReplayConfig

    source = tmp_path / "run"
    save_config(
        EvalConfig(env={"id": "best-of-n", "taskset": {"id": "echo-v1"}}), source
    )
    config = ReplayConfig.model_validate({"taskset": {"id": "echo-v1"}, "rich": False})
    with pytest.raises(SystemExit, match="multi-agent"):
        await run_replay(config, source, tmp_path / "out")


async def test_replay_skips_traceless_episodes(tmp_path):
    """An episode whose env hooks failed before any agent ran holds no traces —
    nothing to re-score, so replay drops it and re-scores the rest."""
    import asyncio

    import verifiers.v1 as vf
    from verifiers.v1.cli.output import append_episode, save_config
    from verifiers.v1.cli.replay import run_replay
    from verifiers.v1.configs.replay import ReplayConfig
    from verifiers.v1.trace import Episode, Trace, TraceTask

    task = TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi"))
    failed = Episode(env="echo-v1", task=task)
    scored = Episode(env="echo-v1", task=task)
    scored.traces.append(Trace(task=task))
    source = tmp_path / "run"
    save_config(EvalConfig(env={"taskset": {"id": "echo-v1"}}), source)
    lock = asyncio.Lock()
    await append_episode(source, failed, lock)
    await append_episode(source, scored, lock)
    config = ReplayConfig.model_validate({"taskset": {"id": "echo-v1"}, "rich": False})
    traces = await run_replay(config, source, tmp_path / "out")
    assert len(traces) == 1
