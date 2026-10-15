"""Image listings and deletes all name the configured team, so they search the registry
sandboxes are created in."""

import prime_sandboxes
from prime_sandboxes.models import ImageListItem, ImageListResponse

from verifiers.v1.runtimes.prime_images import delete_image, find_image


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
    """A registry holding one image, recording each listing's team."""

    def __init__(self, *rows):
        self.rows = list(rows)
        self.list_teams: list[str | None] = []

    async def list(self, team_id, search, offset, limit):
        self.list_teams.append(team_id)
        return ImageListResponse(
            data=self.rows, totalCount=len(self.rows), offset=offset, limit=limit
        )


async def test_listing_names_the_env_team(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_TEAM_ID", "team-1")
    client = RecordingImageClient(_row("task", "v1", "COMPLETED"))
    image = await find_image(client, "task", "v1")
    assert image is not None and image.status == "COMPLETED"
    assert image.ref == "prime/team/task:v1"
    assert client.list_teams == ["team-1"]


async def test_team_comes_from_the_prime_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    (tmp_path / ".prime").mkdir()
    (tmp_path / ".prime" / "config.json").write_text('{"team_id": "team-file"}')
    client = RecordingImageClient()
    assert await find_image(client, "task", "v1") is None
    assert client.list_teams == ["team-file"]


async def test_no_team_is_the_personal_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_TEAM_ID", "")
    client = RecordingImageClient()
    await find_image(client, "task", "v1")
    # "" stops the SDK from substituting its configured team.
    assert client.list_teams == [""]


async def test_a_completed_row_wins_over_a_failed_repush(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    client = RecordingImageClient(
        _row("task", "v1", "FAILED"), _row("task", "v1", "COMPLETED")
    )
    image = await find_image(client, "task", "v1")
    assert image is not None and image.status == "COMPLETED"


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
