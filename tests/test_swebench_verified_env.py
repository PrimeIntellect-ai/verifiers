from unittest.mock import patch

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.harnesses.acp_agent import ACPHarness
from verifiers.envs.experimental.harnesses.opencode import OpenCodeHarness
from verifiers.envs.experimental.swebench_verified_env import SWEBenchVerifiedEnv
from verifiers.envs.experimental.tasksets.swebench_verified import (
    SWEBenchVerifiedTaskSet,
    build_swebench_image_name,
)


def _sample_swebench_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "repo": "pytest-dev/pytest",
                "instance_id": "pytest-dev__pytest-12345",
                "base_commit": "deadbeef",
                "patch": "",
                "test_patch": "",
                "problem_statement": "Fix the failing assertion in the parser.",
                "hints_text": "Look near the assertion rewrite logic.",
                "created_at": "2024-01-01T00:00:00Z",
                "version": "7.4",
                "FAIL_TO_PASS": '["tests/test_parser.py::test_rewrite"]',
                "PASS_TO_PASS": '["tests/test_main.py::test_help"]',
                "environment_setup_commit": "deadbeef",
                "difficulty": "medium",
            }
        ]
    )


def test_build_swebench_image_name_normalizes_instance_id():
    image = build_swebench_image_name(
        "pytest-dev__pytest-12345",
        namespace="swebench",
        arch="x86_64",
        tag="latest",
    )
    assert image == "swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-12345:latest"


def test_taskset_loads_hf_rows_into_rollout_dataset():
    with patch(
        "verifiers.envs.experimental.tasksets.swebench_verified.load_dataset",
        return_value=_sample_swebench_dataset(),
    ):
        taskset = SWEBenchVerifiedTaskSet(max_examples=1)

    row = taskset.get_dataset()[0]
    assert row["task"] == "pytest-dev__pytest-12345"
    assert "Fix the failing assertion in the parser." in row["question"]
    assert "Look near the assertion rewrite logic." in row["question"]
    assert "tests/test_parser.py::test_rewrite" in row["question"]
    assert row["info"]["FAIL_TO_PASS"] == ["tests/test_parser.py::test_rewrite"]
    assert row["info"]["PASS_TO_PASS"] == ["tests/test_main.py::test_help"]
    assert (
        row["info"]["docker_image"]
        == "swebench/sweb.eval.x86_64.pytest-dev_1776_pytest-12345:latest"
    )


def test_swebench_verified_env_composes_new_abstractions():
    with patch(
        "verifiers.envs.experimental.tasksets.swebench_verified.load_dataset",
        return_value=_sample_swebench_dataset(),
    ):
        env = SWEBenchVerifiedEnv(max_examples=1, disabled_tools=["question", "task"])

    assert isinstance(env.harness, OpenCodeHarness)
    assert isinstance(env.taskset, SWEBenchVerifiedTaskSet)
    assert env.taskset.agent_workdir == "/testbed"
    assert env.dataset is not None
    assert env.dataset[0]["task"] == "pytest-dev__pytest-12345"

    response = vf.Response(
        id="resp-1",
        created=0,
        model="test-model",
        message=vf.ResponseMessage(
            content="done\n",
            reasoning_content="",
            tool_calls=[
                vf.ToolCall(
                    id="call-1",
                    name="READ",
                    arguments='{\n  "path": "foo.py"\n}',
                )
            ],
            finish_reason="tool_calls",
            is_truncated=False,
        ),
    )

    normalized = env.normalize_response(response)
    assert normalized.message.content == "done"
    assert normalized.message.tool_calls is not None
    assert normalized.message.tool_calls[0].name == "read"
    assert normalized.message.tool_calls[0].arguments == '{"path":"foo.py"}'


def test_swebench_verified_env_accepts_acp_harness():
    with patch(
        "verifiers.envs.experimental.tasksets.swebench_verified.load_dataset",
        return_value=_sample_swebench_dataset(),
    ):
        harness = ACPHarness(command=("opencode", "acp"), cwd="/tmp/acp")
        env = SWEBenchVerifiedEnv(max_examples=1, harness=harness)

    assert env.harness is harness
    assert isinstance(env.taskset, SWEBenchVerifiedTaskSet)
