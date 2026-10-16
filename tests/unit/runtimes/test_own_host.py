import json

import pytest

from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.runtimes.lan import (
    interpolate,
    own_host_programs,
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
    assert program.argv == ["sh", "-c", "cd /srv && exec python -m http.server 8000"]
    assert program.environment == {"A": "1", "B": "x=y"}
    assert program.healthcheck is None
    assert program.memory == 0.5


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        ({"image": "x", "command": "y", "volumes": ["data:/data"]}, "volumes"),
        ({"image": "x", "command": "y", "build": "."}, "build"),
        ({"command": "y"}, "needs an image"),
        ({"image": "x"}, "needs a command"),
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
