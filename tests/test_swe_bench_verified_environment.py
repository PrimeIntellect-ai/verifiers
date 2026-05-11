from __future__ import annotations

import importlib
import sys
from pathlib import Path

from datasets import Dataset

import verifiers.v1 as vf


def load_module(monkeypatch):
    env_dir = Path(__file__).parents[1] / "environments" / "swe_bench_verified"
    monkeypatch.syspath_prepend(str(env_dir))
    sys.modules.pop("swe_bench_verified", None)
    return importlib.import_module("swe_bench_verified")


def fake_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "repo": "example/repo",
                "instance_id": "example__repo-1",
                "base_commit": "abc123",
                "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n",
                "test_patch": "diff --git a/test_a.py b/test_a.py\n",
                "problem_statement": "Fix the bug.",
                "hints_text": "Look at a.py.",
                "created_at": "2024-01-01T00:00:00Z",
                "version": "1.0",
                "FAIL_TO_PASS": '["test_a.py::test_fix"]',
                "PASS_TO_PASS": '["test_a.py::test_existing"]',
                "environment_setup_commit": "def456",
                "difficulty": "medium",
            }
        ]
    )


def test_swe_bench_verified_loads_taskset(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: fake_dataset())

    env = module.load_environment(train_limit=1, eval_limit=1)
    task = next(iter(env.taskset))

    assert isinstance(env, vf.Env)
    assert env.taskset.taskset_id == "swe-bench/verified"
    assert task["task_id"] == "example__repo-1"
    assert task["answer"].startswith("diff --git")
    assert task["info"]["repo"] == "example/repo"
    assert task["info"]["FAIL_TO_PASS"] == '["test_a.py::test_fix"]'
    assert "Repository: example/repo" in task["prompt"][0]["content"]
    assert "Look at a.py." in task["prompt"][0]["content"]


def test_swe_bench_verified_filters_rows(monkeypatch):
    module = load_module(monkeypatch)

    rows = fake_dataset().add_item(
        {
            "repo": "other/repo",
            "instance_id": "other__repo-1",
            "base_commit": "abc123",
            "patch": "diff --git a/b.py b/b.py\n",
            "test_patch": "",
            "problem_statement": "Fix another bug.",
            "hints_text": "",
            "created_at": "",
            "version": "",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
            "environment_setup_commit": "",
            "difficulty": "hard",
        }
    )
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: rows)

    taskset = module.load_taskset(repos=["other/repo"], difficulties=["hard"])
    tasks = list(taskset.source())

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "other__repo-1"


def test_extract_patch_prefers_patch_tags(monkeypatch):
    module = load_module(monkeypatch)

    completion = [
        {"role": "assistant", "content": "notes\n<patch>\ndiff --git a/x b/x\n</patch>"}
    ]

    assert module.extract_patch(completion) == "diff --git a/x b/x"


def test_normalize_patch_ignores_index_lines(monkeypatch):
    module = load_module(monkeypatch)

    left = "diff --git a/a b/a\nindex 123..456 100644\n--- a/a\n+++ b/a\n"
    right = "diff --git a/a b/a\n--- a/a\n+++ b/a\n"

    assert module.normalize_patch(left) == module.normalize_patch(right)


def test_patch_file_paths_reads_diff_headers_and_plus_files(monkeypatch):
    module = load_module(monkeypatch)

    patch = """diff --git a/src/a.py b/src/a.py
index 123..456 100644
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-old
+new
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ /dev/null
"""

    assert module.patch_file_paths(patch) == {"src/a.py", "src/b.py"}


def test_official_submission_uses_swe_bench_jsonl_shape(monkeypatch):
    module = load_module(monkeypatch)
    row = module.build_record(fake_dataset()[0])

    submission = module.official_submission(row, row["answer"])

    assert submission["instance_id"] == "example__repo-1"
    assert submission["model_patch"].startswith("diff --git")
    assert "index " not in submission["model_patch"]


async def test_exact_patch_reward_accepts_gold_patch(monkeypatch):
    module = load_module(monkeypatch)
    row = module.build_record(fake_dataset()[0])
    task = vf.Task(row)
    state = vf.State.for_task(task)
    state["completion"] = [
        {"role": "assistant", "content": f"<patch>\n{row['answer']}\n</patch>"}
    ]

    assert await module.exact_patch(task, state) == 1.0
    assert await module.patch_similarity(task, state) == 1.0
    assert await module.changed_file_overlap(task, state) == 1.0


async def test_exact_patch_reward_rejects_wrong_patch(monkeypatch):
    module = load_module(monkeypatch)
    row = module.build_record(fake_dataset()[0])
    task = vf.Task(row)
    state = vf.State.for_task(task)
    state["completion"] = [{"role": "assistant", "content": "diff --git a/z b/z\n"}]

    assert await module.exact_patch(task, state) == 0.0
    assert await module.patch_similarity(task, state) < 1.0
    assert await module.changed_file_overlap(task, state) == 0.0
