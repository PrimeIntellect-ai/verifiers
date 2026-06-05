"""Loaders: resolve an env id to its taskset / harness / environment.

The convention from v1 (a module exposes `load_taskset`/`load_harness`/
`load_environment`), with none of the 412-line config-introspection machinery.
An env id resolves to a bundled `examples.<name>` module, else a top-level module
of that name. Each env module's typed factories validate their own configs.
"""

import importlib
import importlib.util
from types import ModuleType

from verifiers.v2.environment import EnvConfig, Environment
from verifiers.v2.harness import Harness, HarnessConfig
from verifiers.v2.taskset import Taskset, TasksetConfig


def import_env(env_id: str) -> ModuleType:
    name = env_id.replace("-", "_")
    bundled = f"verifiers.v2.examples.{name}"
    if importlib.util.find_spec(bundled) is not None:
        return importlib.import_module(bundled)
    return importlib.import_module(name)


def load_taskset(env_id: str, config: TasksetConfig | None = None) -> Taskset:
    # Pass config through (possibly None) so the env factory supplies its own
    # typed config default rather than us forcing a base TasksetConfig.
    return import_env(env_id).load_taskset(config)


def load_harness(env_id: str, config: HarnessConfig | None = None) -> Harness:
    return import_env(env_id).load_harness(config)


def load_environment(env_id: str, config: EnvConfig | None = None) -> Environment:
    return import_env(env_id).load_environment(config)
