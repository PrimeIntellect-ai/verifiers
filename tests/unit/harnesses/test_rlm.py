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


async def test_rlm_install_cache_supports_unprivileged_default_image(monkeypatch):
    ensure_installed = AsyncMock()
    monkeypatch.setattr(rlm_harness, "ensure_installed", ensure_installed)
    harness = RLMHarness(RLMHarnessConfig(id="rlm"))

    await harness.setup(SimpleNamespace(user=None, prepare_uv_script=AsyncMock()))

    ensure_installed.assert_awaited_once()
    call = ensure_installed.await_args.kwargs
    directory = call["directory"]
    binary = f"{directory}/bin/rlm"
    assert directory.startswith("/var/tmp/vf-rlm-")
    assert call["ready"] == f"[ -f {directory}/.ready ] && [ -x {binary} ]"
    assert f"chmod a+x {binary} && test -x {binary}" in call["install"]


async def test_rlm_agent_user_skills_use_unprivileged_writable_cache(monkeypatch):
    ensure_installed = AsyncMock()
    install_skills = AsyncMock()
    monkeypatch.setattr(rlm_harness, "ensure_installed", ensure_installed)
    harness = RLMHarness(
        RLMHarnessConfig(id="rlm", skills=[{"runtime": "/task/skills"}])
    )
    monkeypatch.setattr(harness, "install_skills", install_skills)

    await harness.setup(SimpleNamespace(user="vf-agent", prepare_uv_script=AsyncMock()))

    install_skills.assert_awaited_once()
    skills_dir = install_skills.await_args.args[1]
    assert skills_dir.startswith("/var/tmp/vf-rlm-skills-")
    assert skills_dir.endswith("/home/skills")
    assert ensure_installed.await_args.kwargs["directory"] == harness._install_dir()


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
