"""RLM's per-run state directory is made by root for the runtime's user.

rlm created it itself, as the agent's user, which fails where the image takes /tmp from
everyone but root: CyberStrike's python-fastapi-boundary-hardening runs `chmod 0755
/tmp` and its agent as `agent` ("Permission denied: '/tmp/vf-rlm-state'", 2026-10-28)."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from verifiers.v1.harnesses.rlm.harness import (
    RLM_STATE_DIR,
    RLMHarness,
    RLMHarnessConfig,
)


@dataclass
class Runtime:
    user: str | None
    calls: list[tuple[str | None, list[str]]] = field(default_factory=list)
    exit_code: int = 0

    def with_user(self, user: str | None) -> "Runtime":
        return Runtime(user, self.calls, self.exit_code)

    async def run(self, argv: list[str], env: dict[str, str]) -> SimpleNamespace:
        self.calls.append((self.user, argv))
        return SimpleNamespace(exit_code=self.exit_code, stdout="", stderr="denied")


async def test_root_makes_the_state_dir_and_hands_it_to_the_user():
    runtime = Runtime("agent")
    trace = SimpleNamespace(id="t1")
    await RLMHarness(RLMHarnessConfig())._make_state_dir(trace, runtime)  # type: ignore[arg-type]
    assert runtime.calls == [
        (
            None,
            [
                "sh",
                "-c",
                f"mkdir -p {RLM_STATE_DIR}/t1/home && chown -R agent: {RLM_STATE_DIR}/t1",
            ],
        )
    ]
    await RLMHarness(RLMHarnessConfig()).cleanup(trace, runtime)  # type: ignore[arg-type]
    assert runtime.calls[-1] == (None, ["rm", "-rf", f"{RLM_STATE_DIR}/t1"])


async def test_a_root_runtime_needs_no_chown_and_a_failure_is_named():
    runtime = Runtime(None)
    await RLMHarness(RLMHarnessConfig())._make_state_dir(
        SimpleNamespace(id="t2"), runtime
    )  # type: ignore[arg-type]
    assert runtime.calls[0][1][-1] == f"mkdir -p {RLM_STATE_DIR}/t2/home"
    with pytest.raises(RuntimeError, match="RLM state directory"):
        await RLMHarness(RLMHarnessConfig())._make_state_dir(  # type: ignore[arg-type]
            SimpleNamespace(id="t3"), Runtime("agent", exit_code=1)
        )
