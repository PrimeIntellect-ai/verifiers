import asyncio
from pathlib import Path

import verifiers.v1 as vf
from verifiers.gepa.config import GEPAV1Config
from verifiers.gepa.v1_adapter import (
    VerifiersV1GEPAAdapter,
    make_v1_gepa_dataset,
    shared_initial_prompt_v1,
)
from verifiers.scripts import gepa as gepa_script
from verifiers.scripts.gepa import (
    ResolvedGEPAClients,
    _rewrite_positional_target,
    _run_v1_or_legacy_gepa,
    _uses_v1_config,
)
from verifiers.types import ClientConfig


class FakeEpisode:
    def __init__(self, task: vf.Task) -> None:
        self.task = task

    async def run(self, semaphore=None):
        async with semaphore:
            trace = vf.Trace(task=self.task)
            trace.record_reward("system_prompt_seen", self.task.system_prompt == "new")
            trace.stop("done")
            return [trace]


class FakeEnv:
    def __init__(self) -> None:
        self.seen_tasks: list[vf.Task] = []

    def episode(self, task, ctx, n=1):
        assert n == 1
        self.seen_tasks.append(task)
        return FakeEpisode(task)


class FakeClient(vf.Client):
    async def get_response(self, *args, **kwargs):  # pragma: no cover - not called here
        raise AssertionError("fake env should not call the client")


def test_gepa_v1_config_parses_native_env_shape():
    config = GEPAV1Config.model_validate(
        {
            "taskset": {"id": "echo-v1"},
            "harness": {"id": "null"},
            "model": "test-model",
            "gepa": {"max_calls": 3, "num_train": 2, "num_val": 1},
            "sampling": {"temperature": 0, "max_tokens": 16},
        }
    )

    assert config.taskset.id == "echo-v1"
    assert config.harness.id == "null"
    assert config.environment_label == "echo-v1"
    assert config.gepa.max_calls == 3
    assert config.sampling.max_tokens == 16


def test_gepa_cli_routes_v1_taskset_ids_to_taskset_config(tmp_path: Path):
    config_path = tmp_path / "gepa.toml"
    config_path.write_text(
        'model = "test-model"\n[taskset]\nid = "echo-v1"\n[gepa]\nmax_calls = 3\n'
    )

    assert _rewrite_positional_target(["echo-v1"]) == ["--taskset.id", "echo-v1"]
    assert _rewrite_positional_target(["--id", "echo-v1"]) == [
        "--taskset.id",
        "echo-v1",
    ]
    assert _rewrite_positional_target(["echo-v0"]) == ["--id", "echo-v0"]
    assert _uses_v1_config(["@", str(config_path)])


def test_gepa_v1_shape_falls_back_to_legacy_env(monkeypatch):
    config = GEPAV1Config.model_validate(
        {
            "taskset": {"id": "echo-v0"},
            "model": "test-model",
            "save_results": False,
            "gepa": {"max_calls": 3, "num_train": 2, "num_val": 1},
            "sampling": {"temperature": 0, "max_tokens": 16},
        }
    )
    calls = []

    def fake_v0_run(**kwargs):
        calls.append(kwargs)

    def fake_v1_run(**kwargs):
        raise AssertionError("legacy env should not enter the v1 GEPA runner")

    monkeypatch.setattr(gepa_script, "run_gepa_optimization", fake_v0_run)
    monkeypatch.setattr(gepa_script, "run_gepa_v1_optimization", fake_v1_run)

    _run_v1_or_legacy_gepa(
        config,
        ResolvedGEPAClients(
            model="test-model",
            reflection_model="teacher-model",
            client_config=ClientConfig(),
            reflection_client_config=ClientConfig(),
        ),
    )

    assert config.is_legacy
    assert calls[0]["env_id"] == "echo-v0"
    assert calls[0]["env_configs"][0].id == "echo-v0"
    assert calls[0]["sampling_args"] == {"temperature": 0.0, "max_tokens": 16}


def test_v1_gepa_dataset_repeats_tiny_tasksets_deterministically():
    tasks = [
        vf.Task(idx=0, prompt="a", system_prompt="sys"),
        vf.Task(idx=1, prompt="b", system_prompt="sys"),
    ]

    first = make_v1_gepa_dataset(tasks, n=5, seed=0)
    second = make_v1_gepa_dataset(tasks, n=5, seed=0)

    assert [row["example_id"] for row in first] == [0, 1, 2, 3, 4]
    assert [row["task"].idx for row in first] == [row["task"].idx for row in second]
    assert shared_initial_prompt_v1(tasks) == "sys"


def test_v1_gepa_adapter_injects_candidate_system_prompt():
    task = vf.Task(idx=0, prompt="question", system_prompt="old")
    runner = asyncio.Runner()
    env = FakeEnv()
    try:
        adapter = VerifiersV1GEPAAdapter(
            env=env,
            client=FakeClient(),
            model="test-model",
            sampling=vf.SamplingConfig(max_tokens=16),
            runner=runner,
            max_concurrent=1,
        )
        batch = adapter.evaluate(
            [{"example_id": 7, "task": task}],
            {"system_prompt": "new"},
            capture_traces=True,
        )
    finally:
        runner.close()

    assert batch.scores == [1.0]
    assert batch.outputs[0]["example_id"] == 7
    assert batch.outputs[0]["task"]["system_prompt"] == "new"
    assert env.seen_tasks[0].system_prompt == "new"
