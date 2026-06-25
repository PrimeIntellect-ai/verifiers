import pytest

from verifiers.v1.harnesses.mini_swe_agent.harness import (
    MiniSWEAgentHarness,
    MiniSWEAgentHarnessConfig,
)


class RecordingRuntime:
    def __init__(self) -> None:
        self.envs: list[dict[str, str] | None] = []

    async def prepare_uv_script(
        self, script: str | bytes, env: dict[str, str] | None = None
    ) -> list[str]:
        self.envs.append(env)
        return ["python", "/tmp/mini_swe_agent.py"]


@pytest.mark.asyncio
async def test_mini_swe_agent_setup_can_install_harness_dependencies_in_offline_tasks():
    harness = MiniSWEAgentHarness(
        MiniSWEAgentHarnessConfig(
            env={
                "UV_OFFLINE": "1",
                "PIP_NO_INDEX": "1",
                "PIP_CONFIG_FILE": "/etc/pip.conf",
                "TASK_ENV": "kept",
            }
        )
    )
    runtime = RecordingRuntime()

    await harness.setup(runtime)

    assert runtime.envs == [
        {
            "UV_OFFLINE": "false",
            "PIP_NO_INDEX": "0",
            "PIP_CONFIG_FILE": "/dev/null",
            "TASK_ENV": "kept",
        }
    ]
