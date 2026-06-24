import io
import json
import tarfile
from urllib.parse import parse_qs, urlparse

import pytest

from verifiers.v1.tasksets.harbor_v1 import taskset as harbor


def task_archive(name: str) -> bytes:
    files = {
        "task.toml": (
            f'[task]\nname = "{name}"\n\n[environment]\ndocker_image = "ubuntu:24.04"\n'
        ),
        "instruction.md": f"Complete {name}.",
        "tests/test.sh": "#!/bin/sh\n",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_downloads_task_without_harbor_cli_or_api_key(tmp_path, monkeypatch) -> None:
    calls = []
    archive = task_archive("one")

    def fake_urlopen(request):
        calls.append(request)
        if "/rest/v1/dataset_version_tag?" in request.full_url:
            return io.BytesIO(b"[]")
        if request.full_url.endswith("/rest/v1/rpc/resolve_task_version"):
            assert json.loads(request.data) == {
                "p_org": "acme",
                "p_name": "one",
                "p_ref": "stable",
            }
            return io.BytesIO(json.dumps({"archive_path": "one.tgz"}).encode())
        if request.full_url.endswith(
            "/storage/v1/object/authenticated/packages/one.tgz"
        ):
            return io.BytesIO(archive)
        raise AssertionError(request.full_url)

    monkeypatch.setattr(harbor, "CACHE", tmp_path)
    monkeypatch.setattr(harbor, "urlopen", fake_urlopen)
    monkeypatch.setenv("HARBOR_SUPABASE_URL", "https://registry.example/")
    monkeypatch.setenv("HARBOR_SUPABASE_PUBLISHABLE_KEY", "public-key")
    monkeypatch.setenv("HARBOR_API_KEY", "must-not-be-used")

    output = harbor.dataset_dir("acme/one@stable")
    assert (output / "one" / "instruction.md").read_text() == "Complete one."
    assert harbor.dataset_dir("acme/one@stable") == output
    assert len(calls) == 3
    assert all("api-key-exchange" not in request.full_url for request in calls)
    assert all(request.get_header("Apikey") == "public-key" for request in calls)
    assert all(request.get_header("Authorization") is None for request in calls)


def test_downloads_dataset_archives_from_membership_query(
    tmp_path, monkeypatch
) -> None:
    calls = []
    archives = {name: task_archive(name) for name in ("one", "two")}

    def fake_urlopen(request):
        calls.append(request)
        path = urlparse(request.full_url).path
        if path.endswith("/dataset_version_tag"):
            query = parse_qs(urlparse(request.full_url).query)
            assert query["tag"] == ["eq.latest"]
            return io.BytesIO(
                json.dumps([{"dataset_version": {"id": "version-id"}}]).encode()
            )
        if path.endswith("/dataset_version_task"):
            query = parse_qs(urlparse(request.full_url).query)
            assert "archive_path" in query["select"][0]
            offset = int(query["offset"][0])
            assert offset in (0, 1000)
            name = "one" if offset == 0 else "two"
            rows = [
                {
                    "task_version": {
                        "archive_path": f"{name}.tgz",
                        "package": {"name": name},
                    }
                }
            ]
            response = io.BytesIO(json.dumps(rows).encode())
            response.headers = {"Content-Range": "0-999/1001"}
            return response
        name = path.rsplit("/", 1)[-1].removesuffix(".tgz")
        if name in archives:
            return io.BytesIO(archives[name])
        raise AssertionError(request.full_url)

    monkeypatch.setattr(harbor, "CACHE", tmp_path)
    monkeypatch.setattr(harbor, "urlopen", fake_urlopen)

    output = harbor.dataset_dir("acme/suite")
    assert (output / "suite" / "one" / "instruction.md").is_file()
    assert (output / "suite" / "two" / "instruction.md").is_file()

    taskset = harbor.HarborTaskset(harbor.HarborConfig(dataset="acme/suite"))
    assert [task.name for task in taskset.load_tasks()] == ["one", "two"]
    membership_calls = [
        call for call in calls if "/dataset_version_task?" in call.full_url
    ]
    assert len(membership_calls) == 2
    assert membership_calls[0].get_header("Prefer") == "count=exact"
    assert not any("resolve_task_version" in call.full_url for call in calls)


@pytest.mark.parametrize(
    ("ref", "field", "value"),
    [("7", "revision", "eq.7"), ("sha256:abc", "content_hash", "eq.abc")],
)
def test_resolves_dataset_revision_and_digest(
    ref, field, value, tmp_path, monkeypatch
) -> None:
    version_queries = []

    def fake_urlopen(request):
        path = urlparse(request.full_url).path
        if path.endswith("/rpc/resolve_task_version"):
            return io.BytesIO(b"null")
        if path.endswith("/dataset_version"):
            version_queries.append(parse_qs(urlparse(request.full_url).query))
            return io.BytesIO(json.dumps([{"id": "version-id"}]).encode())
        if path.endswith("/dataset_version_task"):
            response = io.BytesIO(b"[]")
            response.headers = {"Content-Range": "*/0"}
            return response
        raise AssertionError(request.full_url)

    monkeypatch.setattr(harbor, "CACHE", tmp_path)
    monkeypatch.setattr(harbor, "urlopen", fake_urlopen)

    harbor.dataset_dir(f"acme/suite@{ref}")
    assert version_queries[0][field] == [value]
