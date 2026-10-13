"""Image builds, listings and deletes all name the configured team, so a pushed image
lands in the registry the listing searches."""

from types import SimpleNamespace

import httpx
import prime_sandboxes
import pytest
from prime_sandboxes.models import ImageListItem, ImageListResponse

from verifiers.v1.runtimes import prime_images
from verifiers.v1.runtimes.prime_images import delete_image, ensure_image


def _row(name, tag, status):
    return ImageListItem.model_validate(
        {
            "id": f"{name}-{tag}",
            "artifactType": "VM_SANDBOX",
            "imageName": name,
            "imageTag": tag,
            "status": status,
            "createdAt": "2026-10-13T00:00:00Z",
            "displayRef": f"prime/team/{name}:{tag}",
        }
    )


class RecordingImageClient:
    """A registry that lists the pushed image COMPLETED, recording each call's team."""

    def __init__(self):
        self.list_teams: list[str | None] = []
        self.build_teams: list[str | None] = []
        self.pushed: tuple[str, str] | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list(self, team_id, search, offset, limit):
        self.list_teams.append(team_id)
        rows = [_row(*self.pushed, "COMPLETED")] if self.pushed else []
        return ImageListResponse(
            data=rows, totalCount=len(rows), offset=offset, limit=limit
        )

    async def initiate_build(self, request):
        self.build_teams.append(request.team_id)
        self.pushed = (request.image_name, request.image_tag)
        return SimpleNamespace(build_id="b1", upload_url="https://upload")

    async def start_build(self, build_id):
        pass


@pytest.fixture
def client(monkeypatch, tmp_path):
    class FakeHttp:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def put(self, url, content, headers):
            return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(httpx, "AsyncClient", FakeHttp)
    monkeypatch.setattr(prime_images, "POLL_SECONDS", 0)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.prime/config.json
    recording = RecordingImageClient()
    monkeypatch.setattr(prime_sandboxes, "AsyncImageClient", lambda: recording)
    return recording


@pytest.fixture
def context(tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "Dockerfile").write_text("FROM alpine\n")
    return ctx


async def test_build_and_listing_name_the_env_team(client, context, monkeypatch):
    monkeypatch.setenv("PRIME_TEAM_ID", "team-1")
    image = await ensure_image(context, "task", tag="v1")
    assert image.status == "COMPLETED"
    assert client.build_teams == ["team-1"]
    assert client.list_teams and set(client.list_teams) == {"team-1"}


async def test_team_comes_from_the_prime_config_file(client, context, monkeypatch):
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    prime_dir = context.parent / ".prime"
    prime_dir.mkdir()
    (prime_dir / "config.json").write_text('{"team_id": "team-file"}')
    await ensure_image(context, "task", tag="v1")
    assert client.build_teams == ["team-file"]
    assert set(client.list_teams) == {"team-file"}


async def test_no_team_is_the_personal_registry(client, context, monkeypatch):
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    await ensure_image(context, "task", tag="v1")
    assert client.build_teams == [None]
    # "" stops the SDK from substituting its configured team.
    assert set(client.list_teams) == {""}


async def test_delete_names_the_configured_team(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        async def request(self, method, endpoint, params=None):
            calls.append(params)
            return {}

        async def aclose(self):
            pass

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(prime_sandboxes, "AsyncAPIClient", FakeApi)
    monkeypatch.setenv("PRIME_TEAM_ID", "team-1")
    assert await delete_image("task", "v1")
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    assert await delete_image("task", "v1")
    assert calls == [{"teamId": "team-1"}, None]
