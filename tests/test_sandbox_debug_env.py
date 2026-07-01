import pytest

import verifiers as vf
from verifiers.envs.experimental.composable import (
    SandboxDebugEnv,
    SandboxDebugRubric,
    SandboxSpec,
    SandboxTaskSet,
    SWEDebugEnv,
)


class DummySandboxTaskSet(SandboxTaskSet):
    def __init__(self):
        from datasets import Dataset

        super().__init__(
            dataset=Dataset.from_list([{"question": "noop", "info": {}, "answer": ""}])
        )

    def get_instruction(self, info: dict) -> str:
        return "Run the debug step."

    def get_rubric(self) -> vf.Rubric:
        return vf.Rubric()

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec()


def test_swe_debug_env_is_deprecated_subclass_not_alias():
    assert SWEDebugEnv is not SandboxDebugEnv
    assert issubclass(SWEDebugEnv, SandboxDebugEnv)
    assert SandboxDebugRubric.__name__ == "SandboxDebugRubric"

    with pytest.warns(DeprecationWarning, match="native v1 `debug` CLI"):
        env = SWEDebugEnv(
            DummySandboxTaskSet(),
            debug_step="command",
            debug_command="true",
        )

    assert isinstance(env, SandboxDebugEnv)
    assert env.labels == ["sandbox-debug"]
