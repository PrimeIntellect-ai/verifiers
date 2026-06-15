from unittest.mock import AsyncMock

import pytest

from harnesses.default import DefaultHarness, DefaultHarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import SamplingConfig


@pytest.mark.asyncio
async def test_harness_failure_includes_both_output_streams() -> None:
    runtime = AsyncMock()
    runtime.run_uv_script.return_value = ProgramResult(
        exit_code=1,
        stdout="fatal error",
        stderr="diagnostic warning",
    )
    harness = DefaultHarness(DefaultHarnessConfig())
    ctx = RolloutContext(
        client=AsyncMock(), model="test-model", sampling=SamplingConfig()
    )
    trace = Trace(task=Task(idx=0, instruction="test"))

    with pytest.raises(ProgramError) as exc:
        await harness.run(ctx, trace, runtime, "http://endpoint", "secret", {})

    assert "stdout: fatal error" in str(exc.value)
    assert "stderr: diagnostic warning" in str(exc.value)
