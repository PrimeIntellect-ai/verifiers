import pytest

from verifiers.v1.harnesses.noop import NoopHarness, NoopHarnessConfig


@pytest.mark.asyncio
async def test_noop_harness_exits_without_model_or_runtime_calls():
    harness = NoopHarness(NoopHarnessConfig(id="noop"))

    result = await harness.launch(None, None, None, "", "", {})

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
