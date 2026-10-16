"""Every supported runtime declares each capability as expected, and the framework's
check for that capability lets supported runs through and refuses the rest with an
informative message, before anything is provisioned."""

import contextlib
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

import verifiers.v1 as vf
from verifiers.v1.errors import HarnessError, ToolsetError
from verifiers.v1.mcp import launch
from verifiers.v1.mcp.toolset import ToolsetConfig
from verifiers.v1.runtimes import Runtime, RuntimeConfig, backend_cls
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.runtimes.strategies import _deployment_strategy
from verifiers.v1.utils.compile import resolve_runtime_config, validate_pairing
from verifiers.v1.utils.loaders import harness_class, harness_config_type

RUNTIMES = ["subprocess", "docker", "prime"]
SUPPORTED = {
    "task_images": {"docker", "prime"},
    "user_switching": {"docker", "prime"},
    "port_exposure": {"subprocess", "docker"},
    "nested_containers": {"prime"},
    "streaming_processes": {"subprocess", "docker", "prime"},
    "network_policies": {"prime"},
}


class Proceeded(Exception):
    """The check passed and the framework went on to its next step."""


def config(runtime: str) -> RuntimeConfig:
    return TypeAdapter(RuntimeConfig).validate_python({"type": runtime})


def task(**data) -> vf.Task:
    return vf.Task(data=vf.TaskData(idx=0, prompt=None, **data))


def harness(harness_id: str) -> vf.Harness:
    return harness_class(harness_id)(harness_config_type(harness_id)(id=harness_id))


def fake_backend(runtime: str) -> SimpleNamespace:
    """The runtime's declared flags, without constructing (or authenticating) it."""
    cls = backend_cls(config(runtime))
    flags = {name: getattr(cls, name) for name in dir(cls) if name.isupper()}
    return SimpleNamespace(type=runtime, **flags)


def check_task_images(runtime: str, monkeypatch) -> None:
    resolve_runtime_config(config(runtime), task(image="alpine"))
    validate_pairing(harness("codex"), vf.Task, config(runtime))


def check_user_switching(runtime: str, monkeypatch) -> None:
    resolve_runtime_config(config(runtime), task(agent_user="agent"))


def check_network_policies(runtime: str, monkeypatch) -> None:
    resolve_runtime_config(config(runtime), task(network_allow=["pypi.org"]))


def check_nested_containers(runtime: str, monkeypatch) -> None:
    _deployment_strategy(None, fake_backend(runtime), ComposeSpec(directory="."))


async def check_port_exposure(runtime: str, monkeypatch) -> None:
    def provision(*args, **kwargs):
        raise Proceeded

    monkeypatch.setattr(launch, "provision_backend", provision)
    server = SimpleNamespace(
        config=ToolsetConfig(runtime=config(runtime)), server_name="tools"
    )
    with contextlib.suppress(Proceeded):
        async with launch.serve(server):
            pass


async def check_streaming_processes(runtime: str, monkeypatch) -> None:
    acp = harness("claude-code")

    async def prepare_acp(*args, **kwargs):
        raise Proceeded

    monkeypatch.setattr(acp, "prepare_acp", prepare_acp)
    view = Runtime(fake_backend(runtime))
    with contextlib.suppress(Proceeded):
        await acp.session(None, None, view, "", "", {}, None)


CHECKS = {
    "task_images": (check_task_images, ValueError, "cannot run images"),
    "user_switching": (
        check_user_switching,
        ValueError,
        "cannot run as another user",
    ),
    "port_exposure": (check_port_exposure, ToolsetError, "cannot expose a port"),
    "nested_containers": (
        check_nested_containers,
        ValueError,
        "cannot run nested containers",
    ),
    "streaming_processes": (
        check_streaming_processes,
        HarnessError,
        "cannot stream processes",
    ),
    "network_policies": (
        check_network_policies,
        ValueError,
        "does not enforce network policies",
    ),
}


@pytest.mark.parametrize("runtime", RUNTIMES)
@pytest.mark.parametrize("capability", SUPPORTED)
async def test_runtime_capability(capability, runtime, monkeypatch) -> None:
    check, error, message = CHECKS[capability]
    refusal = (
        contextlib.nullcontext()
        if runtime in SUPPORTED[capability]
        else pytest.raises(error, match=f"the {runtime} runtime {message}")
    )
    with refusal:
        result = check(runtime, monkeypatch)
        if result is not None:
            await result
