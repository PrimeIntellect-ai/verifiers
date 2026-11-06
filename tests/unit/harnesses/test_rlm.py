"""The rlm harness maps its session's stop reason onto the trace's stop condition."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import verifiers.v1 as vf
import verifiers.v1.harnesses.rlm.harness as rlm_harness
from verifiers.v1.harnesses.rlm.harness import (
    RLM_SESSION_METADATA_KEY,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.runtimes import ProgramResult


@pytest.fixture
def runtime():
    root = SimpleNamespace(run=AsyncMock(return_value=ProgramResult(0, "", "")))
    return SimpleNamespace(
        user=None,
        run=AsyncMock(return_value=ProgramResult(0, "", "")),
        with_user=lambda user: root,
        prepare_uv_script=AsyncMock(),
    )


async def test_rlm_install_cache_supports_unprivileged_default_image(
    monkeypatch, runtime
):
    ensure_installed = AsyncMock()
    monkeypatch.setattr(rlm_harness, "ensure_installed", ensure_installed)
    harness = RLMHarness(RLMHarnessConfig(id="rlm", env={"PATH": "/configured/bin"}))

    await harness.setup(runtime)

    ensure_installed.assert_awaited_once()
    call = ensure_installed.await_args.kwargs
    directory = call["directory"]
    binary = f"{directory}/bin/rlm"
    assert directory.startswith("/var/tmp/vf-rlm-")
    assert call["ready"] == f"[ -f {directory}/.ready ] && [ -x {binary} ]"
    assert f"chmod a+x {binary} && test -x {binary}" in call["install"]
    assert "RLM_SYSTEM_PACKAGES=disabled RLM_USE_EXISTING_UV=1" in call["install"]
    root_calls = runtime.with_user(None).run.await_args_list
    assert len(root_calls) == 2
    assert root_calls[0].args[1] == {}
    assert root_calls[1].args[1] == {"PATH": "/configured/bin"}
    runtime.run.assert_awaited_once()
    assert runtime.run.await_args.args[1] == {"PATH": "/configured/bin"}


async def test_rlm_agent_user_skills_use_unprivileged_writable_cache(
    monkeypatch, runtime
):
    ensure_installed = AsyncMock()
    install_skills = AsyncMock()
    monkeypatch.setattr(rlm_harness, "ensure_installed", ensure_installed)
    harness = RLMHarness(
        RLMHarnessConfig(id="rlm", skills=[{"runtime": "/task/skills"}])
    )
    monkeypatch.setattr(harness, "install_skills", install_skills)

    runtime.user = "vf-agent"
    await harness.setup(runtime)

    install_skills.assert_awaited_once()
    skills_dir = install_skills.await_args.args[1]
    assert skills_dir.startswith("/var/tmp/vf-rlm-skills-")
    assert skills_dir.endswith("/home/skills")
    assert ensure_installed.await_args.kwargs["directory"] == harness._install_dir()


@pytest.mark.parametrize("missing_in", ["root_downloader", "root_uv", "agent"])
async def test_rlm_missing_prerequisites_fail_before_install_or_acp(
    monkeypatch, runtime, missing_in
):
    ensure_installed = AsyncMock()
    monkeypatch.setattr(rlm_harness, "ensure_installed", ensure_installed)
    if missing_in == "root_uv":
        runtime.with_user(None).run.side_effect = [
            ProgramResult(0, "", ""),
            ProgramResult(1, "", ""),
        ]
    else:
        view = runtime if missing_in == "agent" else runtime.with_user(None)
        view.run.return_value = ProgramResult(1, "", "")
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))

    with pytest.raises(RuntimeError, match="requires preinstalled"):
        await harness.setup(runtime)

    ensure_installed.assert_not_awaited()
    runtime.prepare_uv_script.assert_not_awaited()


@pytest.mark.parametrize(
    "reason,condition",
    [
        ("max_total_tokens", "max_total_tokens"),
        ("max_total_turns", "max_turns"),
        ("token_budget", "max_output_tokens"),
        ("compaction_failed", "compaction_failed"),
        ("done", "agent_completed"),
        (None, "agent_completed"),
    ],
)
def test_rlm_stop_reason_survives_completion_and_serialization(reason, condition):
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="q")),
    )
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))
    metadata = {
        RLM_SESSION_METADATA_KEY: {
            "session_id": trace.id,
            "metrics": {"turns": 1},
            "last_stop_reason": reason,
        }
    }
    harness.acp_turn_result(
        trace, vf.ACPTurn(reply="partial", response_metadata=metadata)
    )
    trace.stop("agent_completed")
    harness.acp_close_result(trace, metadata)

    restored = vf.WireTrace.model_validate_json(trace.model_dump_json())
    assert restored.stop_condition == condition
    assert restored.is_truncated is (condition != "agent_completed")
