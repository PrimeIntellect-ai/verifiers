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


def test_conflicting_cli_taskset_ids_are_refused():
    """Two `--env.taskset.id` occurrences naming different ids (the positional
    shorthand counts as the first) are refused up front — the narrowing pins the
    first while the config merge takes the last, so letting them through yields a
    baffling wrong-config-type error."""
    from verifiers.v1.cli.resolve import narrow_config, with_positional_taskset

    argv = with_positional_taskset(["echo-v1", "--env.taskset.id", "other-v1"])
    with pytest.raises(SystemExit, match="'echo-v1' and 'other-v1'"):
        narrow_config(EvalConfig, argv)
    # The same id twice is not a conflict.
    narrow_config(
        EvalConfig, with_positional_taskset(["echo-v1", "--env.taskset.id", "echo-v1"])
    )


def test_dangling_harness_id_is_refused_with_the_builtins():
    """A dangling `--env.agent.harness.id` (no value) parses as boolean True; the
    narrowing must answer with the field path and the built-in harness ids, not a
    raw TypeError from deep inside the loader."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"harness\.id needs an id.*bash"):
        EvalConfig.model_validate(
            {"env": {"taskset": {"id": "echo-v1"}, "agent": {"harness": {"id": True}}}}
        )


def test_nested_env_errors_keep_their_flag_path():
    """Sub-models validate inside `mode=\"before\"` narrowing validators, which
    would surface their error locs without the `env` (and `harness`) segments —
    the CLI would render a flag the user never typed."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as e:
        EvalConfig.model_validate(
            {
                "env": {
                    "taskset": {"id": "echo-v1"},
                    "agent": {"harness": {"id": "bash", "runtime": {"type": "dockr"}}},
                }
            }
        )
    assert e.value.errors()[0]["loc"][:4] == ("env", "agent", "harness", "runtime")


def test_role_guard_keys_on_the_default_instance():
    """The definition-time role guard must use the machinery's membership test
    (`isinstance(field.default, AgentConfig)`): `timeout: AgentConfig | None =
    AgentConfig()` IS a role to `_declared_agent_configs`, so it must be refused
    as shadowing; an AgentConfig annotation without a default instance is never a
    role, so it must be refused as missing one — whatever the annotation form."""
    import verifiers.v1 as vf

    with pytest.raises(TypeError, match="shadow"):

        class Shadowing(vf.EnvConfig):
            timeout: vf.AgentConfig | None = vf.AgentConfig()

    with pytest.raises(TypeError, match="default"):

        class Uninstantiated(vf.EnvConfig):
            solver: vf.AgentConfig | None = None


def test_legacy_id_with_v1_taskset_is_refused():
    """A legacy `--id` next to a v1 `env.taskset` used to be silently inert
    (`is_legacy` is False, the v0 env never loads); the mix is refused with a
    pointer at `--env.id`, the likely intent."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="--env.id"):
        EvalConfig.model_validate(
            {"id": "wordle", "env": {"taskset": {"id": "echo-v1"}}}
        )


def test_env_level_harness_key_points_at_the_seat():
    """`--env.harness.*` (v0 muscle-memory, one level up from the seat) gets a
    pointer home instead of a bare extra_forbidden."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"--env\.agent\.harness"):
        EvalConfig.model_validate(
            {"env": {"taskset": {"id": "echo-v1"}, "harness": {"id": "bash"}}}
        )


def test_server_flips_an_unset_rich_default():
    """`--rich` defaults on but is in-process only: an unset `rich` defaults off
    under `--server`; only an explicitly set `rich` is refused with it."""
    from pydantic import ValidationError

    served = EvalConfig.model_validate(
        {"env": {"taskset": {"id": "echo-v1"}}, "server": True}
    )
    assert served.rich is False
    with pytest.raises(ValidationError, match="--rich"):
        EvalConfig.model_validate(
            {"env": {"taskset": {"id": "echo-v1"}}, "server": True, "rich": True}
        )


def test_replay_lifts_the_saved_eval_taskset():
    """Replay layers the source run's saved config as its base. An eval run keeps
    its taskset on the [env] block — replay's root taskset must pick it up; an
    earlier replay's own root taskset stays authoritative."""
    from verifiers.v1.configs.replay import ReplayConfig

    lifted = ReplayConfig.model_validate(
        {"env": {"taskset": {"id": "echo-v1"}, "agent": {"max_turns": 2}}, "model": "m"}
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


async def test_replay_guard_reads_the_saved_taskset(tmp_path):
    """The multi-agent verdict comes from the SOURCE run's saved config — a
    replay-layer taskset override must not reclassify a multi-agent run (duet-v1
    exports its own multi-agent env) as replayable."""
    import pytest

    from verifiers.v1.cli.output import save_config
    from verifiers.v1.cli.replay import run_replay
    from verifiers.v1.configs.replay import ReplayConfig

    source = tmp_path / "run"
    save_config(EvalConfig(env={"taskset": {"id": "duet-v1"}}), source)
    config = ReplayConfig.model_validate({"taskset": {"id": "echo-v1"}, "rich": False})
    with pytest.raises(SystemExit, match="multi-agent"):
        await run_replay(config, source, tmp_path / "out")


async def test_replay_skips_traceless_episodes(tmp_path):
    """An episode whose env hooks failed before any agent ran holds no traces —
    nothing to re-score, so replay drops it and re-scores the rest, keeping each
    kept trace's source env identity on the replay output."""
    import asyncio
    import json

    import verifiers.v1 as vf
    from verifiers.v1.cli.output import TRACES_FILE, append_episode, save_config
    from verifiers.v1.cli.replay import run_replay
    from verifiers.v1.configs.replay import ReplayConfig
    from verifiers.v1.trace import EpisodeInfo, EpisodeRecord, Trace, TraceTask

    task = TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi"))
    failed = EpisodeRecord(episode=EpisodeInfo(env="echo-v1"))
    scored = EpisodeRecord(episode=EpisodeInfo(env="echo-v1"))
    scored.traces.append(Trace(task=task))
    source = tmp_path / "run"
    save_config(EvalConfig(env={"taskset": {"id": "echo-v1"}}), source)
    lock = asyncio.Lock()
    await append_episode(source, failed, lock)
    await append_episode(source, scored, lock)
    config = ReplayConfig.model_validate({"taskset": {"id": "echo-v1"}, "rich": False})
    out = tmp_path / "out"
    traces = await run_replay(config, source, out)
    assert len(traces) == 1
    (line,) = (out / TRACES_FILE).read_text().splitlines()
    assert json.loads(line)["episode"]["env"] == "echo-v1"
