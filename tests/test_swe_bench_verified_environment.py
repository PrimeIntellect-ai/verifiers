import pytest
from datasets import Dataset

import verifiers as vf
from environments.swe_bench_verified import swe_bench_verified


GOLD_PATCH = """diff --git a/src/demo.py b/src/demo.py
--- a/src/demo.py
+++ b/src/demo.py
@@ -1 +1 @@
-old
+new
"""


def fake_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "instance_id": "demo__repo-1",
                "repo": "demo/repo",
                "base_commit": "abc123",
                "environment_setup_commit": "setup123",
                "version": "1.0",
                "problem_statement": "Fix the demo behavior.",
                "hints_text": "Look at src/demo.py.",
                "FAIL_TO_PASS": '["tests/test_demo.py::test_fix"]',
                "PASS_TO_PASS": ["tests/test_demo.py::test_existing"],
                "patch": GOLD_PATCH,
                "test_patch": "diff --git a/tests/test_demo.py b/tests/test_demo.py\n",
            },
            {
                "instance_id": "demo__repo-2",
                "repo": "demo/repo",
                "base_commit": "def456",
                "environment_setup_commit": "setup456",
                "version": "1.0",
                "problem_statement": "Fix another behavior.",
                "hints_text": "",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": [],
                "patch": "diff --git a/src/other.py b/src/other.py\n",
                "test_patch": "",
            },
        ]
    )


def test_load_environment_builds_limited_taskset(monkeypatch):
    calls = {}

    def fake_load_dataset(dataset_name: str, **kwargs):
        calls["dataset_name"] = dataset_name
        calls["kwargs"] = kwargs
        return fake_dataset()

    monkeypatch.setattr(swe_bench_verified, "load_dataset", fake_load_dataset)

    env = swe_bench_verified.load_environment(
        config=swe_bench_verified.SWEBenchVerifiedEnvConfig(
            taskset=swe_bench_verified.SWEBenchVerifiedTasksetConfig(max_examples=1)
        )
    )
    task = next(iter(env.taskset))

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, swe_bench_verified.SWEBenchVerifiedTaskset)
    assert calls["dataset_name"] == swe_bench_verified.DEFAULT_DATASET_NAME
    assert calls["kwargs"]["split"] == "test"
    assert task["taskset_id"] == "swe-bench/verified"
    assert task["task_id"] == "demo__repo-1"
    assert task["answer"] == GOLD_PATCH
    assert task["info"]["repo"] == "demo/repo"
    assert task["info"]["fail_to_pass"] == ["tests/test_demo.py::test_fix"]
    assert "<patch>...</patch>" in task["prompt"][0]["content"]
    assert "tests/test_demo.py::test_existing" in task["prompt"][0]["content"]


def test_extract_patch_supports_tags_fences_and_raw_diff():
    assert swe_bench_verified.extract_patch(
        [{"role": "assistant", "content": f"<patch>\n{GOLD_PATCH}\n</patch>"}]
    ) == GOLD_PATCH.strip()
    assert swe_bench_verified.extract_patch(
        [{"role": "assistant", "content": f"```diff\n{GOLD_PATCH}\n```"}]
    ) == GOLD_PATCH.strip()
    assert swe_bench_verified.extract_patch(f"Here is the patch:\n{GOLD_PATCH}") == (
        GOLD_PATCH.strip()
    )


def test_patch_metrics_and_official_submission():
    partial_patch = """diff --git a/src/demo.py b/src/demo.py
--- a/src/demo.py
+++ b/src/demo.py
@@ -1 +1 @@
-old
+newer
"""
    task = {
        "task_id": "fallback-id",
        "info": {"instance_id": "demo__repo-1"},
    }

    assert swe_bench_verified.changed_files(partial_patch) == {"src/demo.py"}
    assert swe_bench_verified.changed_file_overlap(partial_patch, GOLD_PATCH) == 1.0
    assert 0.0 < swe_bench_verified.patch_similarity(partial_patch, GOLD_PATCH) < 1.0
    assert swe_bench_verified.official_submission(task, partial_patch) == {
        "instance_id": "demo__repo-1",
        "model_patch": partial_patch,
    }


@pytest.mark.asyncio
async def test_patch_reward_records_submission_and_scores_exact_match(monkeypatch):
    monkeypatch.setattr(
        swe_bench_verified, "load_dataset", lambda *args, **kwargs: fake_dataset()
    )
    taskset = swe_bench_verified.load_taskset(
        config=swe_bench_verified.SWEBenchVerifiedTasksetConfig(max_examples=1)
    )
    env = vf.Env(taskset=taskset)
    task = next(iter(taskset))
    state = await env.harness.setup_state(task, vf.State.for_task(task))
    state["completion"] = [
        {"role": "assistant", "content": f"<patch>{GOLD_PATCH}</patch>"}
    ]

    await env.harness.runtime.score_rollout(task, state)

    assert state["reward"] == 1.0
    assert state["swe_bench_verified_patch"] == GOLD_PATCH.strip()
    assert state["swe_bench_verified_submission"] == {
        "instance_id": "demo__repo-1",
        "model_patch": GOLD_PATCH.strip(),
    }
    assert state["swe_bench_verified_exact_match"] is True
