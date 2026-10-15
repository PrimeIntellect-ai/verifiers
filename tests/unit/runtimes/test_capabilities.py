"""Runtime capabilities: a custom runtime plugin loads from its import path, and every
runtime refuses a task feature it does not declare before anything is provisioned."""

import contextlib
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import TypeAdapter

import verifiers.v1 as vf
from verifiers.v1.runtimes import (
    CustomRuntimeConfig,
    RuntimeBackend,
    RuntimeConfig,
    backend_cls,
)
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.runtimes.strategies import _deployment_strategy
from verifiers.v1.utils.compile import resolve_runtime_config, validate_pairing
from verifiers.v1.utils.loaders import harness_class, harness_config_type


class HostPlugin(RuntimeBackend):
    """Declares nothing beyond the defaults: no image, user, or network policy."""


class ContainerPlugin(RuntimeBackend):
    supports_image: ClassVar[bool] = True
    supports_user: ClassVar[bool] = True


NOT_A_BACKEND = object()


def config(runtime: str, **fields) -> RuntimeConfig:
    if runtime in ("host-plugin", "container-plugin"):
        plugin = HostPlugin if runtime == "host-plugin" else ContainerPlugin
        fields = {
            "type": "custom",
            "import_path": f"{__name__}.{plugin.__name__}",
            **fields,
        }
    else:
        fields = {"type": runtime, **fields}
    return TypeAdapter(RuntimeConfig).validate_python(fields)


def task(**data) -> vf.Task:
    return vf.Task(data=vf.TaskData(idx=0, prompt=None, **data))


def test_custom_runtime_loads_the_plugin_class() -> None:
    plugin = config("container-plugin", kwargs={"region": "us"})
    assert isinstance(plugin, CustomRuntimeConfig)
    assert plugin.kwargs == {"region": "us"}
    assert backend_cls(plugin) is ContainerPlugin


@pytest.mark.parametrize(
    "import_path,error,match",
    [
        ("HostPlugin", ValueError, "must be a dotted path"),
        ("no_such_package.Backend", ValueError, "cannot import custom runtime"),
        (f"{__name__}.Missing", ValueError, "cannot import custom runtime"),
        (f"{__name__}.NOT_A_BACKEND", TypeError, "must be a RuntimeBackend subclass"),
    ],
)
def test_custom_runtime_rejects_a_bad_import_path(import_path, error, match) -> None:
    with pytest.raises(error, match=match):
        backend_cls(CustomRuntimeConfig(import_path=import_path))


IMAGE = ["docker", "prime", "modal", "e2b", "container-plugin"]
USER = ["docker", "prime", "container-plugin"]
NETWORK = ["prime", "modal", "e2b"]
COMPOSE = ["prime", "modal"]
ALL = [
    "subprocess",
    "docker",
    "prime",
    "modal",
    "e2b",
    "host-plugin",
    "container-plugin",
]


def expect(supported: bool, match: str):
    return (
        contextlib.nullcontext()
        if supported
        else pytest.raises(ValueError, match=match)
    )


@pytest.mark.parametrize("runtime", ALL)
def test_task_image_needs_image_support(runtime) -> None:
    with expect(runtime in IMAGE, "cannot run images"):
        resolved = resolve_runtime_config(config(runtime), task(image="alpine"))
        assert resolved.image == "alpine"


@pytest.mark.parametrize("runtime", ALL)
def test_agent_user_needs_user_support(runtime) -> None:
    with expect(runtime in USER, "cannot run as another user"):
        resolve_runtime_config(config(runtime), task(agent_user="agent"))


@pytest.mark.parametrize("runtime", ALL)
def test_network_policy_needs_policy_support(runtime) -> None:
    with expect(runtime in NETWORK, "does not enforce network policies"):
        resolved = resolve_runtime_config(
            config(runtime), task(network_allow=["pypi.org"])
        )
        assert resolved.allow == ["pypi.org"]


@pytest.mark.parametrize("runtime", ALL)
def test_container_harness_needs_image_support(runtime) -> None:
    codex = harness_class("codex")(harness_config_type("codex")(id="codex"))
    with expect(runtime in IMAGE, "needs a container runtime"):
        validate_pairing(codex, vf.Task, config(runtime))


@pytest.mark.parametrize("runtime", ALL)
def test_compose_needs_compose_support(runtime) -> None:
    cls = backend_cls(config(runtime))
    backend = SimpleNamespace(supports_compose=cls.supports_compose, type=runtime)
    with expect(runtime in COMPOSE, "cannot run Compose services"):
        _deployment_strategy(None, backend, ComposeSpec(directory="."))
