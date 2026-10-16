import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import verifiers.v1 as vf
from verifiers.v1.errors import SandboxError
from verifiers.v1.mcp import launch
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, service_vms
from verifiers.v1.runtimes.service_vms import _start, in_workdir
from verifiers.v1.runtimes.strategies import _deployment_strategy


@pytest.mark.parametrize("name", ["main", "Tools", "my tools", "-tools", "x" * 52])
def test_service_names_become_lan_tags(name):
    with pytest.raises(ValueError, match="service VM name"):
        vf.ServiceVMSpec({name: vf.ServiceVM(image="x", command=["y"])})


def test_service_vms_run_on_prime():
    spec = vf.ServiceVMSpec({"tools": vf.ServiceVM(image="x", command=["y"])})
    with pytest.raises(ValueError, match="Prime"):
        _deployment_strategy(None, DockerConfig(), spec)


def test_in_workdir_creates_and_enters_the_directory():
    assert in_workdir(["true"], None) == ["true"]
    assert in_workdir(["python", "x y.py"], "/srv/app") == [
        "sh",
        "-c",
        "mkdir -p /srv/app && cd /srv/app && exec python 'x y.py'",
    ]


def fake_runtime(calls, *, ready=True, exit_status=None):
    async def run_background(argv, env, log):
        calls.append(("service", argv, env))

    async def run(argv, env):
        if argv[0] == "cat":  # the exit-status file exists once the service has exited
            if exit_status is None:
                return SimpleNamespace(exit_code=1, stdout="")
            return SimpleNamespace(exit_code=0, stdout=exit_status)
        if argv[0] == "tail":
            return SimpleNamespace(exit_code=0, stdout="boom")
        calls.append(("probe", argv, env))
        return SimpleNamespace(exit_code=0 if ready else 1, stdout="")

    return SimpleNamespace(run_background=run_background, run=run)


def test_the_probe_runs_like_the_service():
    service = vf.ServiceVM(
        image="python:3.11-slim",
        command=["python", "serve.py"],
        environment={"PORT": "8000"},
        workdir="/srv",
        readiness=vf.Probe(["python", "check.py"]),
    )
    calls = []
    asyncio.run(_start(fake_runtime(calls), "api", service))
    (_, command, service_env), (_, probe, probe_env) = calls
    assert "mkdir -p /srv && cd /srv && exec python serve.py" in command[2]
    assert probe == ["sh", "-c", "mkdir -p /srv && cd /srv && exec python check.py"]
    assert service_env == probe_env == {"PORT": "8000"}


def test_a_service_that_never_becomes_ready_fails_setup():
    service = vf.ServiceVM(
        image="x",
        command=["serve"],
        readiness=vf.Probe(["check"], interval=0, attempts=3),
    )
    calls = []
    with pytest.raises(SandboxError, match="'api' never became ready: boom"):
        asyncio.run(_start(fake_runtime(calls, ready=False), "api", service))
    assert sum(kind == "probe" for kind, *_ in calls) == 3


@pytest.mark.parametrize("readiness", [None, vf.Probe(["check"], interval=0)])
def test_a_service_that_exits_fails_setup(monkeypatch, readiness):
    monkeypatch.setattr(service_vms, "STARTED_SECONDS", 0)
    service = vf.ServiceVM(image="x", command=["serve"], readiness=readiness)
    runtime = fake_runtime([], ready=False, exit_status="3")
    with pytest.raises(SandboxError, match="'api' exited with 3: boom"):
        asyncio.run(_start(runtime, "api", service))


def test_a_service_without_a_probe_need_only_keep_running(monkeypatch):
    monkeypatch.setattr(service_vms, "STARTED_SECONDS", 0)
    service = vf.ServiceVM(image="x", command=["serve"])
    asyncio.run(_start(fake_runtime([]), "api", service))


@pytest.mark.parametrize(
    ("allow", "egress"), [(None, ["*"]), ([], []), (["pypi.org"], ["pypi.org"])]
)
def test_service_vms_take_their_own_image_resources_and_egress(
    monkeypatch, allow, egress
):
    provisioned = []

    @asynccontextmanager
    async def provision_backend(config, name):
        provisioned.append((name, config))
        raise RuntimeError("provisioned")
        yield

    monkeypatch.setattr(service_vms, "provision_backend", provision_backend)
    workspace = SimpleNamespace(
        config=PrimeConfig(gpu="A100", cpu=2, memory=4, allow=[]), name="ws"
    )
    options = {} if allow is None else {"allow": allow}
    service = vf.ServiceVM(image="tools:1", command=["serve"], cpu=8, **options)
    started = []

    async def start_workspace():
        started.append(True)

    async def enter():
        async with service_vms.service_vms(
            workspace, {"api": service}, start_workspace
        ):
            pass

    with pytest.raises(RuntimeError, match="provisioned"):
        asyncio.run(enter())
    ((name, vm),) = provisioned
    assert name == "ws-api"
    assert (vm.image, vm.gpu, vm.cpu, vm.memory, vm.allow) == (
        "tools:1",
        None,
        8,
        4,
        egress,
    )
    assert started == [True]  # the workspace boots alongside the services


def test_each_vm_enforces_its_own_policy_with_the_framework_routes():
    from verifiers.v1.runtimes.strategies import _ServiceVMDeployment

    applied = []

    def backend(name):
        async def prepare_execution(routes):
            applied.append((name, routes))

        return SimpleNamespace(prepare_execution=prepare_execution)

    strategy = _ServiceVMDeployment()
    strategy._service_backends = [backend("forth")]
    deployment = SimpleNamespace(harness_backend=backend("main"))
    asyncio.run(strategy.prepare_execution(deployment, ["https://tunnel.example"]))
    assert sorted(applied) == [
        ("forth", ["https://tunnel.example"]),
        ("main", ["https://tunnel.example"]),
    ]


def test_a_tool_server_in_a_service_vm_is_reached_by_its_lan_name(monkeypatch):
    async def serve_in_runtime(server, runtime, **kwargs):
        assert kwargs["listen_all"]
        return 1234

    monkeypatch.setattr(launch, "serve_in_runtime", serve_in_runtime)
    server = SimpleNamespace(
        config=vf.ToolsetConfig(service="tools"), server_name="echo"
    )
    runtime = SimpleNamespace(is_local=False, host_url=lambda url: url)

    async def url(hosts):
        async with launch._serve(server, services={"tools": runtime}, hosts=hosts) as s:
            return s.url

    assert asyncio.run(url({"tools": "sb1.sandbox.internal"})) == (
        "http://sb1.sandbox.internal:1234/mcp"
    )
    assert asyncio.run(url({})) == "http://tools:1234/mcp"
