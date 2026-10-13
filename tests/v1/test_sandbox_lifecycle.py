"""Run labels on Prime sandboxes and tunnels, label sweeps, and digest-cached image
builds, against fakes of the Prime SDK."""

import asyncio
import contextlib
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import prime_sandboxes
import prime_tunnel
import pytest
from prime_sandboxes import APIError
from prime_sandboxes.models import (
    BulkDeleteSandboxResponse,
    ImageListItem,
    ImageListResponse,
)
from prime_tunnel.core import client as tunnel_client_module
from prime_tunnel.models import TunnelInfo, TunnelListPage

import verifiers.v1.interception.tunnel.prime as tunnel_prime
from verifiers.v1.runtimes import prime, prime_images, sweep
from verifiers.v1.runtimes.prime_images import (
    ImageBuildError,
    context_digest,
    delete_image,
    digest_tag,
    ensure_image,
)

NOW = datetime(2026, 10, 13, tzinfo=UTC)


@pytest.fixture(autouse=True)
def run_id(monkeypatch):
    monkeypatch.setenv("VF_RUN_ID", "run-1")
    monkeypatch.setattr(prime, "BASE_LABELS", [])


def test_run_labels_are_base_labels_then_the_run_scope():
    prime.set_base_sandbox_labels(["campaign", "run-1"])
    assert prime.run_labels() == ["campaign", "run-1"]
    prime.set_base_sandbox_labels(["campaign"])
    assert prime.run_labels() == ["campaign", "run-1"]


async def test_tunnels_carry_the_run_labels(monkeypatch):
    created: list[dict] = []

    class FakeTunnel:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

        async def start(self) -> str:
            return "https://t.example/"

        def sync_stop(self) -> None:
            pass

    @contextlib.asynccontextmanager
    async def unlimited():
        yield

    monkeypatch.setattr(prime_tunnel, "Tunnel", FakeTunnel)
    monkeypatch.setattr(tunnel_prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(tunnel_prime, "tunnel_limiter", unlimited)
    prime.set_base_sandbox_labels(["campaign"])
    async with tunnel_prime.PrimeTunnel().expose(8000) as url:
        assert url == "https://t.example"
    assert created == [{"local_port": 8000, "labels": ["campaign", "run-1"]}]


# -- sweeps ------------------------------------------------------------------------------


class FakeSandboxClient:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self.outcomes = outcomes
        self.calls: list[list[str]] = []
        self.closed = False

    async def bulk_delete(self, labels):
        self.calls.append(labels)
        outcome = self.outcomes[labels[0]]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self):
        self.closed = True


def deleted(ids, failed=()):
    return BulkDeleteSandboxResponse(
        succeeded=list(ids), failed=[{"id": i} for i in failed], message=""
    )


async def test_sandbox_sweep_deletes_one_label_per_call_and_survives_a_failure(
    monkeypatch,
):
    client = FakeSandboxClient(
        {
            "a": deleted(["s1", "s2"]),
            "b": RuntimeError("gateway"),
            "c": deleted([], ["s3"]),
        }
    )
    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", lambda: client)
    assert await sweep.sweep_sandboxes(["a", "b", "a", "c"]) == {
        "a": 2,
        "b": -1,
        "c": 0,
    }
    assert client.calls == [["a"], ["b"], ["c"]]
    assert client.closed


def tunnel(tunnel_id, labels, user="me", terminated=False):
    return TunnelInfo(
        tunnel_id=tunnel_id,
        hostname="h",
        url="https://h",
        frp_token="",
        server_host="",
        labels=labels,
        expires_at=NOW,
        user_id=user,
        terminated_at=NOW if terminated else None,
    )


class FakeTunnelClient:
    pages: ClassVar[dict[str, object]] = {}

    def __init__(self) -> None:
        self.config = SimpleNamespace(user_id="me")
        self.deletes: list[str] = []
        self.closed = False
        FakeTunnelClient.last = self

    async def list_tunnels_page(self, labels, page, per_page):
        pages = self.pages[labels[0]]
        if isinstance(pages, Exception):
            raise pages
        return TunnelListPage(tunnels=pages[page - 1], has_next=page < len(pages))

    async def delete_tunnel(self, tunnel_id):
        self.deletes.append(tunnel_id)
        if tunnel_id == "broken":
            raise RuntimeError("timeout")
        return tunnel_id != "gone"

    async def close(self):
        self.closed = True


async def test_tunnel_sweep_deletes_only_this_users_live_tunnels_with_the_label(
    monkeypatch,
):
    FakeTunnelClient.pages = {
        "a": [
            [
                tunnel("t1", ["a"]),
                tunnel("t2", ["other"]),
                tunnel("t3", ["a"], user="them"),
            ],
            [
                tunnel("t4", ["a", "x"]),
                tunnel("t5", ["a"], terminated=True),
                tunnel("gone", ["a"]),
            ],
        ],
        "b": RuntimeError("list failed"),
        "c": [[tunnel("broken", ["c"])]],
    }
    monkeypatch.setattr(tunnel_client_module, "TunnelClient", FakeTunnelClient)
    assert await sweep.sweep_tunnels(["a", "b", "c"]) == {"a": 3, "b": -1, "c": 0}
    assert sorted(FakeTunnelClient.last.deletes) == ["broken", "gone", "t1", "t4"]
    assert FakeTunnelClient.last.closed


async def test_sweep_reports_both_kinds_and_skips_no_labels(monkeypatch):
    monkeypatch.setattr(
        prime_sandboxes,
        "AsyncSandboxClient",
        lambda: FakeSandboxClient({"a": deleted(["s"])}),
    )
    FakeTunnelClient.pages = {"a": [[tunnel("t", ["a"])]]}
    monkeypatch.setattr(tunnel_client_module, "TunnelClient", FakeTunnelClient)
    report = await sweep.sweep(["a"])
    assert report.sandboxes == {"a": 1} and report.tunnels == {"a": 1}
    assert await sweep.sweep([]) == sweep.SweepReport()


def test_prime_authenticated_reads_the_key(monkeypatch):
    monkeypatch.setattr(sweep, "load_prime_config", dict)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    assert not sweep.prime_authenticated()
    monkeypatch.setenv("PRIME_API_KEY", "k")
    assert sweep.prime_authenticated()


# -- images ------------------------------------------------------------------------------


@pytest.fixture
def context(tmp_path) -> Path:
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\nCOPY app /app\n")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "run.sh").write_text("echo hi\n")
    return tmp_path


def test_context_digest_covers_content_mode_and_salt(context):
    base = context_digest(context)
    assert context_digest(context) == base
    assert context_digest(context, salt="episode-2") != base
    script = context / "app" / "run.sh"
    os.chmod(script, 0o755)
    executable = context_digest(context)
    assert executable != base
    script.write_text("echo bye\n")
    assert context_digest(context) not in {base, executable}
    (context / ".dockerignore").write_text("app\n")
    with pytest.raises(ValueError, match="dockerignore"):
        context_digest(context)


def row(name, tag, status, artifact="VM_SANDBOX"):
    return ImageListItem.model_validate(
        {
            "id": f"{name}-{tag}-{status}",
            "artifactType": artifact,
            "imageName": name,
            "imageTag": tag,
            "status": status,
            "createdAt": NOW,
            "displayRef": f"prime/me/{name}:{tag}",
        }
    )


class FakeImageClient:
    """A registry whose listing advances one scripted status per poll after a push."""

    def __init__(self, rows: list[ImageListItem], after_push: list[str]) -> None:
        self.rows = rows
        self.after_push = list(after_push)
        self.pushes: list[tuple[str, str]] = []
        self.started: list[str] = []
        self.searches: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list(self, team_id, search, offset, limit):
        self.searches.append(search)
        rows = self.rows
        if self.pushes and self.after_push:
            name, tag = self.pushes[-1]
            rows = [row(name, tag, self.after_push.pop(0))]
            self.rows = rows
        noise = [
            row(search, "other", "COMPLETED"),
            row(search, "x", "COMPLETED", "VM_SANDBOX"),
        ]
        return ImageListResponse(
            data=[*noise, *rows], totalCount=1, offset=offset, limit=limit
        )

    async def initiate_build(self, request):
        assert request.dockerfile_path == prime_images.PACKAGED_DOCKERFILE
        assert request.visibility.value == "PRIVATE"
        self.pushes.append((request.image_name, request.image_tag))
        return SimpleNamespace(
            build_id=f"b{len(self.pushes)}", upload_url="https://upload"
        )

    async def start_build(self, build_id):
        self.started.append(build_id)


@pytest.fixture
def registry(monkeypatch):
    uploads: list[bytes] = []

    class FakeHttp:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def put(self, url, content, headers):
            uploads.append(content)
            return SimpleNamespace(raise_for_status=lambda: None)

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeHttp)
    monkeypatch.setattr(prime_images, "POLL_SECONDS", 0)

    def install(client: FakeImageClient) -> FakeImageClient:
        monkeypatch.setattr(prime_sandboxes, "AsyncImageClient", lambda: client)
        return client

    install.uploads = uploads
    return install


async def test_a_completed_digest_is_reused_without_a_push(context, registry):
    tag = digest_tag(context_digest(context))
    client = registry(FakeImageClient([row("task", tag, "COMPLETED")], []))
    image = await ensure_image(context, "task")
    assert image.ref == f"prime/me/task:{tag}" and image.status == "COMPLETED"
    assert client.pushes == []


async def test_an_absent_image_is_pushed_and_awaited(context, registry):
    client = registry(FakeImageClient([], ["BUILDING", "BUILDING", "COMPLETED"]))
    image = await ensure_image(context, "task", tag="v1")
    assert image.ref == "prime/me/task:v1"
    assert client.pushes == [("task", "v1")] and client.started == ["b1"]
    assert len(registry.uploads) == 1 and registry.uploads[0][:2] == b"\x1f\x8b"


async def test_a_running_build_is_awaited_not_repushed(context, registry):
    client = registry(FakeImageClient([row("task", "v1", "BUILDING")], []))

    async def complete_later():
        await asyncio.sleep(0.01)
        client.rows = [row("task", "v1", "COMPLETED")]

    asyncio.get_running_loop().create_task(complete_later())
    assert (await ensure_image(context, "task", tag="v1")).status == "COMPLETED"
    assert client.pushes == []


async def test_a_failed_earlier_build_is_rebuilt(context, registry):
    client = registry(FakeImageClient([row("task", "v1", "FAILED")], ["COMPLETED"]))
    assert (await ensure_image(context, "task", tag="v1")).status == "COMPLETED"
    assert client.pushes == [("task", "v1")]


async def test_a_build_that_fails_now_raises(context, registry):
    registry(FakeImageClient([], ["BUILDING", "FAILED"]))
    with pytest.raises(ImageBuildError, match="FAILED"):
        await ensure_image(context, "task", tag="v1")


async def test_a_build_past_its_deadline_raises(context, registry):
    registry(FakeImageClient([row("task", "v1", "BUILDING")], []))
    with pytest.raises(ImageBuildError, match="BUILDING"):
        await ensure_image(context, "task", tag="v1", timeout=0)


async def test_concurrent_calls_share_one_build(context, registry):
    client = registry(FakeImageClient([], ["BUILDING", "COMPLETED", "COMPLETED"]))
    first, second = await asyncio.gather(
        ensure_image(context, "task", tag="v1"), ensure_image(context, "task", tag="v1")
    )
    assert first == second and client.pushes == [("task", "v1")]


async def test_invalid_names_and_tags_are_refused(context):
    with pytest.raises(ValueError, match="name"):
        await ensure_image(context, "Task")
    with pytest.raises(ValueError, match="tag"):
        await ensure_image(context, "task", tag="-bad")


async def test_delete_image(monkeypatch):
    calls: list[tuple] = []

    class FakeApi:
        async def request(self, method, endpoint, params=None):
            calls.append((method, endpoint, params))
            if endpoint.endswith("missing"):
                raise APIError("HTTP 404: not found")
            if endpoint.endswith("broken"):
                raise APIError("HTTP 500: boom")
            return {}

        async def aclose(self):
            pass

    monkeypatch.setattr(prime_sandboxes, "AsyncAPIClient", FakeApi)
    monkeypatch.setenv("PRIME_TEAM_ID", "t1")
    assert await delete_image("task", "v1")
    assert not await delete_image("task", "missing")
    with pytest.raises(APIError):
        await delete_image("task", "broken")
    assert calls[0] == ("DELETE", "/images/task/v1", {"teamId": "t1"})
