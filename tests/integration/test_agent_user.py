"""The task's non-root agent user: a scripted agent is denied root-only actions
(agent-user-v1), and every harness installs and runs as that user against a scripted
upstream (agent-user-harness-v1), on the runtimes that can switch users."""

import pytest
from grid import grid, runtimes_grid


@pytest.mark.parametrize(
    "preinstalled_tools", [False, True], ids=["fresh", "preinstalled"]
)
@pytest.mark.parametrize(
    "harness_runtime", runtimes_grid(needs=("user",)), indirect=True
)
async def test_agent_user_cannot_act_as_root(
    run_v1, harness_runtime, preinstalled_tools, tmp_path
):
    """A scripted, model-free agent runs as the task's non-root `agent_user`. Root-only
    actions are denied, while its uv script, runtime file I/O, and colocated tool server
    all act as the user. With `preinstalled_tools`, the tool's package was first installed
    by the default user, as on a reused runtime."""
    (trace,) = await run_v1(
        "agent-user-v1",
        harness="agent-user-v1",
        runtime=harness_runtime,
        output_dir=tmp_path,
        taskset_overrides={"preinstalled_tools": preinstalled_tools},
    )
    report = trace.info["agent_user_report"]
    for key in ("user", "uv_user", "written_owner", "tool_user"):
        assert report[key] == "vf-agent", key
    assert report["secret_exists"] == "ok" and report["write_home"] == "ok"
    root_actions = ("read_secret", "write_etc", "chmod_secret", "signal_init")
    assert {action: report[action] for action in root_actions} == dict.fromkeys(
        root_actions, "denied"
    )
    assert report["read_secret_api"] == "denied"
    assert report["tool_uv"] == "ok"
    assert not report["uv_python"].startswith("/root/")
    assert report["uv_script"] != report["root_uv_script"]
    assert trace.reward == 1.0


@pytest.mark.parametrize(
    "harness,harness_runtime", grid(needs=("user",)), indirect=True
)
async def test_agent_user_harness(
    run_v1, harness, harness_runtime, scripted_model, tmp_path
):
    """Each harness installs and runs a turn as a non-root agent user against a scripted
    upstream, so the run fails only if the harness can't set up or run without root."""
    (trace,) = await run_v1(
        "agent-user-harness-v1",
        harness=harness,
        runtime=harness_runtime,
        output_dir=tmp_path,
        client=scripted_model,
    )
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.num_turns >= 1
