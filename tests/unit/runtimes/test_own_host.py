import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import PrimeConfig, lan
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.runtimes.lan import (
    _start,
    in_workdir,
    interpolate,
    own_host_programs,
    own_host_services,
    seconds,
    service_program,
)

ATLAS = {
    "image": "ghcr.io/example/atlas:1",
    "command": ["sh", "-c", "PORT=$$PORT exec serve --port ${PORT:-1984}"],
    "environment": {"TOKEN": "a$$b", "UNSET": None},
    "healthcheck": {
        "test": ["CMD-SHELL", "curl -sf http://127.0.0.1:1984/list-tools"],
        "interval": "5s",
        "timeout": "10s",
        "retries": 12,
        "start_period": "1m",
    },
    "deploy": {"resources": {"limits": {"cpus": "2", "memory": "4g"}}},
}


def test_interpolation_follows_compose():
    environment = {"SET": "value", "EMPTY": ""}
    assert interpolate("$$SET", environment) == "$SET"
    assert interpolate("${SET}/$SET", environment) == "value/value"
    assert interpolate("${EMPTY:-fallback} ${MISSING-x} $MISSING.", environment) == (
        "fallback x ."
    )
    assert interpolate("[${EMPTY-x}] ${SET:+on} [${EMPTY:+on}]", environment) == (
        "[] on []"
    )
    assert interpolate("${MISSING:-${ALSO:-${SET}}}!", environment) == "value!"
    assert interpolate("${SET:-${ALSO:-x}}!", environment) == "value!"
    assert interpolate("${MISSING:-$${SET}}", environment) == "${SET}"
    with pytest.raises(ValueError, match="MISSING is required: set it"):
        interpolate("${MISSING:?set it}", environment)
    with pytest.raises(ValueError, match="unterminated"):
        interpolate("${MISSING:-${SET}", environment)


def test_durations():
    assert seconds("1m30s", 0) == 90
    assert seconds("500ms", 0) == 0.5
    assert seconds(None, 7) == 7
    with pytest.raises(ValueError):
        seconds("5 seconds", 0)


def test_service_program_reads_an_authored_service():
    program = service_program("atlas", ATLAS, {})
    assert program.image == "ghcr.io/example/atlas:1"
    assert program.argv == ["sh", "-c", "PORT=$PORT exec serve --port 1984"]
    assert program.environment == {"TOKEN": "a$b"}
    assert program.healthcheck == [
        "sh",
        "-c",
        "curl -sf http://127.0.0.1:1984/list-tools",
    ]
    assert (program.interval, program.timeout, program.attempts) == (5, 10, 24)
    assert (program.cpu, program.memory) == (2.0, 4.0)


def test_service_program_forms():
    program = service_program(
        "api",
        {
            "image": "python:3.11-slim",
            "entrypoint": "python -m",
            "command": "http.server 8000",
            "working_dir": "/srv",
            "environment": ["A=1", "B=x=y"],
            "mem_limit": "512m",
        },
        {},
    )
    assert program.argv == ["python", "-m", "http.server", "8000"]
    assert program.workdir == "/srv"
    assert program.environment == {"A": "1", "B": "x=y"}
    assert program.healthcheck is None
    assert program.memory == 0.5


def test_service_program_interpolates_before_reading():
    program = service_program(
        "api",
        {
            "image": "x",
            "command": "y",
            "cpus": "${CPUS:-2}",
            "mem_limit": "${MEMORY:-4g}",
            "environment": ["${PREFIX}_TOKEN=secret"],
        },
        {"PREFIX": "API"},
    )
    assert (program.cpu, program.memory) == (2.0, 4.0)
    assert program.environment == {"API_TOKEN": "secret"}


def test_valueless_variables_pass_through_from_the_project_environment():
    for variables in (
        ["KEPT=1", "PASSED", "UNSET"],
        {"KEPT": 1, "PASSED": None, "UNSET": None},
    ):
        program = service_program(
            "api",
            {"image": "x", "command": "y", "environment": variables},
            {"PASSED": "from-project"},
        )
        assert program.environment == {"KEPT": "1", "PASSED": "from-project"}


GPU = {"reservations": {"devices": [{"capabilities": ["gpu"]}]}}


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        ({"image": "x", "command": "y", "volumes": ["data:/data"]}, "volumes"),
        ({"image": "x", "command": "y", "build": "."}, "build"),
        ({"command": "y"}, "needs an image"),
        ({"image": "x"}, "needs a command"),
        ({"image": "x", "command": "y", "deploy": {"resources": GPU}}, "devices"),
        ({"image": "x", "command": "y", "deploy": {"replicas": 2}}, "replicas"),
    ],
)
def test_service_program_refuses_what_a_vm_cannot_honor(definition, message):
    with pytest.raises(ValueError, match=message):
        service_program("svc", definition, {})


def write(tmp_path, services):
    (tmp_path / "environment").mkdir(exist_ok=True)
    (tmp_path / "environment" / "docker-compose.yaml").write_text(
        json.dumps({"services": services})
    )
    return str(tmp_path)


def test_own_host_programs(tmp_path):
    directory = write(tmp_path, {"atlas": ATLAS})
    programs = own_host_programs(ComposeSpec(directory, own_host=("atlas",)), {})
    assert list(programs) == ["atlas"]


@pytest.mark.parametrize(
    ("services", "own_host", "message"),
    [
        ({"atlas": ATLAS}, ("other",), "not in the Compose file"),
        ({"atlas": ATLAS, "db": {"image": "postgres"}}, ("atlas",), "db"),
        ({"atlas": ATLAS, "main": {"image": "custom"}}, ("atlas",), "main"),
    ],
)
def test_own_host_programs_need_the_agent_vm_as_workspace(
    tmp_path, services, own_host, message
):
    spec = ComposeSpec(write(tmp_path, services), own_host=own_host)
    with pytest.raises(ValueError, match=message):
        own_host_programs(spec, {})


def test_in_workdir_creates_and_enters_the_directory():
    assert in_workdir(["true"], None) == ["true"]
    assert in_workdir(["python", "x y.py"], "/srv/app") == [
        "sh",
        "-c",
        "mkdir -p /srv/app && cd /srv/app && exec python 'x y.py'",
    ]


def test_health_check_runs_like_the_service():
    program = service_program(
        "api",
        {
            "image": "python:3.11-slim",
            "command": "python serve.py",
            "working_dir": "/srv",
            "environment": {"PORT": "8000"},
            "healthcheck": {"test": "python check.py $$PORT"},
        },
        {},
    )
    calls = []

    async def run_background(argv, env, log):
        calls.append(("service", argv, env))

    async def run(argv, env):
        calls.append(("check", argv, env))
        return SimpleNamespace(exit_code=0, stdout="")

    asyncio.run(
        _start(SimpleNamespace(run_background=run_background, run=run), "api", program)
    )
    (_, service, service_env), (_, check, check_env) = calls
    assert "mkdir -p /srv && cd /srv && exec python serve.py" in service[2]
    assert check == [
        "sh",
        "-c",
        "mkdir -p /srv && cd /srv && exec sh -c 'python check.py $PORT'",
    ]
    assert service_env == check_env == {"PORT": "8000"}


def test_service_vms_take_no_gpu_from_the_workspace(monkeypatch):
    provisioned = []

    @asynccontextmanager
    async def provision_backend(config, name):
        provisioned.append(config)
        raise RuntimeError("provisioned")
        yield

    monkeypatch.setattr(lan, "provision_backend", provision_backend)
    workspace = SimpleNamespace(config=PrimeConfig(gpu="A100"), _client=None, name="ws")
    program = service_program("api", {"image": "x", "command": "y"}, {})

    async def enter():
        async with own_host_services(workspace, {"api": program}):
            pass

    with pytest.raises(RuntimeError, match="provisioned"):
        asyncio.run(enter())
    (vm,) = provisioned
    assert (vm.gpu, vm.image, vm.allow) == (None, "x", ["*"])


@pytest.mark.parametrize(("status", "exited"), [(1, False), (0, True)])
def test_a_service_without_a_health_check_must_still_be_running(
    monkeypatch, status, exited
):
    monkeypatch.setattr(lan, "STARTED_SECONDS", 0)
    program = service_program("api", {"image": "x", "command": "serve"}, {})

    async def run_background(argv, env, log):
        pass

    async def run(argv, env):
        # `cat` of the exit-status file succeeds only once the service has exited.
        return SimpleNamespace(exit_code=status, stdout="3")

    runtime = SimpleNamespace(run_background=run_background, run=run)
    if exited:
        with pytest.raises(SandboxError, match="'api' exited with 3"):
            asyncio.run(_start(runtime, "api", program))
    else:
        asyncio.run(_start(runtime, "api", program))
