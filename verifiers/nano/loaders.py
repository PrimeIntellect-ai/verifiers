"""Loaders: resolve an env id to its taskset / agent / environment.

The convention from v1 (a module exposes `load_taskset`/`load_agent`/
`load_environment`), with none of the 412-line config-introspection machinery.
An env id resolves to a bundled `examples.<name>` module, else a top-level module
of that name. Each env module's typed factories validate their own configs.
"""

import importlib
import importlib.util
from types import ModuleType

from verifiers.nano.agent import Agent, AgentConfig
from verifiers.nano.environment import EnvConfig, Environment
from verifiers.nano.taskset import Taskset, TasksetConfig


def import_env(env_id: str) -> ModuleType:
    name = env_id.replace("-", "_")
    bundled = f"verifiers.nano.examples.{name}"
    if importlib.util.find_spec(bundled) is not None:
        return importlib.import_module(bundled)
    return importlib.import_module(name)


def load_taskset(env_id: str, config: TasksetConfig | None = None) -> Taskset:
    # Pass config through (possibly None) so the env factory supplies its own
    # typed config default rather than us forcing a base TasksetConfig.
    return import_env(env_id).load_taskset(config)


def load_agent(env_id: str, config: AgentConfig | None = None) -> Agent:
    return import_env(env_id).load_agent(config)


def load_environment(env_id: str, config: EnvConfig | None = None) -> Environment:
    return import_env(env_id).load_environment(config)
