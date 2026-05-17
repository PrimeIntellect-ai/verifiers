import sys
import types

import verifiers as vf
from verifiers.scripts import init


def module_from_template(template: str, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(template, module.__dict__)
    return module


def test_v1_init_template_uses_taskset_subclass() -> None:
    module = module_from_template(init.ENVIRONMENT_TEMPLATE, "generated_v1_env")

    env = module.load_environment()

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, module.EnvTaskset)
    assert isinstance(env.taskset.config, module.EnvTasksetConfig)
    assert env.taskset.config.source is None
    assert env.taskset.rows()[0]["answer"] == "cba"
    assert env.taskset.rewards[0].__name__ == "exact_answer"
    assert "def source(" not in init.ENVIRONMENT_TEMPLATE
    assert "Taskset(source=" not in init.ENVIRONMENT_TEMPLATE


def test_v1_harness_init_template_uses_taskset_subclass() -> None:
    module = module_from_template(
        init.HARNESS_ENVIRONMENT_TEMPLATE,
        "generated_v1_harness_env",
    )

    env = module.load_environment()

    assert isinstance(env, vf.Env)
    assert isinstance(env.taskset, module.EnvTaskset)
    assert isinstance(env.taskset.config, module.EnvTasksetConfig)
    assert isinstance(env.harness.config, module.EnvHarnessConfig)
    assert env.taskset.config.source is None
    assert env.taskset.rows()[0]["answer"] == "cba"
    assert env.taskset.rewards[0].__name__ == "exact_answer"
    assert "def source(" not in init.HARNESS_ENVIRONMENT_TEMPLATE
    assert "Taskset(source=" not in init.HARNESS_ENVIRONMENT_TEMPLATE
