"""Compose deployment's rules are callable on a Compose file before any build, the
runtime refuses by the same rules, and a failed command's error keeps its cause."""

import json
from pathlib import Path

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerConfig
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.runtimes.compose import (
    ComposeSpec,
    compose_file_errors,
    compose_service_errors,
    output_tail,
)


def test_a_compose_file_needs_project_scoped_names():
    authored = {
        "services": {"main": {"build": "."}, "db": {"container_name": "db"}},
        "volumes": {"data": {"name": "shared"}, "cache": None},
        "networks": {"outside": {"external": True}},
    }
    errors = compose_file_errors(authored)
    assert len(errors) == 3
    assert any("volumes" in e for e in errors)
    assert any("networks" in e for e in errors)
    assert any(e.startswith("service db:") for e in errors)
    assert compose_file_errors({"services": {"main": {"build": "."}}}) == []


@pytest.mark.parametrize(
    "service",
    [
        {"network_mode": "host"},
        {"gpus": "all"},
        {"deploy": {"resources": {"reservations": {"devices": [{"count": 1}]}}}},
        {"scale": 2},
        {"deploy": {"replicas": 3}},
        {"ports": [{"target": 80, "published": "8080"}]},
    ],
)
def test_resolved_services_refused(service):
    errors = compose_service_errors({"services": {"web": service}})
    assert len(errors) == 1 and errors[0].startswith("service web:")


def test_resolved_services_accepted():
    resolved = {
        "services": {
            "main": {"ports": [{"target": 80}, {"target": 81, "published": ""}]},
            "db": {"deploy": {"replicas": 1}, "network_mode": "service:main"},
        }
    }
    assert compose_service_errors(resolved) == []


def test_output_tail_keeps_the_end():
    result = ProgramResult(1, "", "progress\n" * 10_000 + "container db is unhealthy")
    tail = output_tail(result, limit=100)
    assert len(tail) == 103 and tail.endswith("container db is unhealthy")
    assert output_tail(ProgramResult(1, "out", "err")) == "err\nout"


class FakeDocker:
    """The Docker CLI as Compose deployment calls it: `config` resolves the authored
    services, `up` fails after a long log."""

    def __init__(self, services, log):
        self.services = services
        self.log = log
        self.overlay = None

    async def __call__(self, *argv, env=None, input=None):
        if argv[:3] == ("docker", "compose", "version"):
            return ProgramResult(0, "v2", "")
        if "config" in argv:
            return ProgramResult(0, json.dumps({"services": self.services}), "")
        if "up" in argv:
            files = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
            self.overlay = json.loads(Path(files[-1]).read_text())
            return ProgramResult(1, "", self.log)
        return ProgramResult(0, "", "")


def _task(tmp_path, compose):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "docker-compose.yaml").write_text(compose)
    return ComposeSpec(str(tmp_path))


async def _up(monkeypatch, tmp_path, compose, services, log="done"):
    from verifiers.v1.runtimes import compose as module

    docker = FakeDocker(services, log)
    monkeypatch.setattr(module, "cli", docker)
    with pytest.raises(SandboxError) as raised:
        async with module.compose_services(
            DockerConfig(), _task(tmp_path, compose), {}, trust_compose=True
        ):
            pytest.fail("up failed")
    return docker, str(raised.value)


async def test_up_failure_keeps_the_cause(monkeypatch, tmp_path):
    log = (
        "#5 building\n" * 20_000
        + "dependency failed to start: container db is unhealthy"
    )
    _, error = await _up(
        monkeypatch,
        tmp_path,
        "services:\n  main:\n    image: app\n",
        {"main": {"image": "app"}},
        log,
    )
    assert error.startswith("Compose up failed:")
    assert error.endswith("container db is unhealthy")
    assert len(error) < 20 * 2**10


async def test_runtime_refuses_by_the_same_rules(monkeypatch, tmp_path):
    from verifiers.v1.runtimes import compose as module

    services = {"main": {"image": "app", "ports": [{"target": 80, "published": "80"}]}}
    monkeypatch.setattr(module, "cli", FakeDocker(services, ""))
    with pytest.raises(SandboxError, match="dynamically assigned host ports"):
        async with module.compose_services(
            DockerConfig(),
            _task(tmp_path, "services:\n  main:\n    image: app\n"),
            {},
            trust_compose=True,
        ):
            pass
    (tmp_path / "environment" / "named.yaml").write_text(
        "services:\n  main:\n    image: app\n    container_name: fixed\n"
    )
    with pytest.raises(SandboxError, match="container_name"):
        async with module.compose_services(
            DockerConfig(),
            ComposeSpec(str(tmp_path), file="environment/named.yaml"),
            {},
            trust_compose=True,
        ):
            pass


async def test_a_service_naming_the_image_it_builds_is_built(monkeypatch, tmp_path):
    # Compose alone would pull a published `netbox` and skip the build.
    services = {
        "main": {"image": "netbox", "build": {"context": "."}},
        "db": {"image": "postgres:16"},
        "cache": {
            "image": "redis",
            "build": {"context": "."},
            "pull_policy": "missing",
        },
    }
    docker, _ = await _up(
        monkeypatch,
        tmp_path,
        "services:\n  main:\n    image: netbox\n    build: .\n",
        services,
    )
    overlay = docker.overlay["services"]
    assert overlay["main"]["pull_policy"] == "build"
    assert "pull_policy" not in overlay.get("db", {})
    assert "pull_policy" not in overlay.get("cache", {})
