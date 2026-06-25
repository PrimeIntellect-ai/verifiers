import pytest

from verifiers.v1.runtimes import DockerConfig
from verifiers.v1.runtimes import DockerMount
from verifiers.v1.runtimes import DockerRuntime
from verifiers.v1.runtimes import runtime_host_endpoint_url
from verifiers.v1.runtimes.base import ProgramResult


@pytest.mark.asyncio
async def test_docker_runtime_passes_configured_bind_mounts_to_docker_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_docker(*args: str) -> ProgramResult:
        calls.append(args)
        if args[:2] == ("version", "--format"):
            return ProgramResult(exit_code=0, stdout="25.0.0\n", stderr="")
        if args[:1] == ("run",):
            return ProgramResult(exit_code=0, stdout="abc123def456\n", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr("verifiers.v1.runtimes.docker.docker", fake_docker)
    runtime = DockerRuntime(
        DockerConfig(
            mounts=[
                DockerMount(
                    source="/host/library",
                    target="/workspace/library",
                    read_only=True,
                )
            ]
        ),
        name="vf-test",
    )

    await runtime.start()

    assert (
        "--mount",
        "type=bind,source=/host/library,target=/workspace/library,readonly",
    ) in zip(calls[1], calls[1][1:])


def test_docker_runtime_uses_docker_desktop_host_endpoint_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("verifiers.v1.runtimes.docker.sys.platform", "darwin")
    runtime = DockerRuntime(DockerConfig(), name="vf-test")

    assert runtime.host_endpoint_url(55521) == "http://host.docker.internal:55521"
    assert (
        runtime_host_endpoint_url(DockerConfig(), 55521)
        == "http://host.docker.internal:55521"
    )
