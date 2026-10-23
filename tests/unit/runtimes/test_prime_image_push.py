"""`push_image` builds a context once per tag, waits until its VM image is bootable,
and pushes a failed build again."""

import io
import tarfile
from types import SimpleNamespace

import httpx
import pytest
from prime_sandboxes.models import ImageListItem, ImageListResponse

from verifiers.v1.runtimes import prime_images
from verifiers.v1.runtimes.prime_images import (
    ImageBuildError,
    context_digest,
    push_image,
)


class FakeRegistry:
    """A registry whose builds run through `outcomes` (one status sequence per build)."""

    def __init__(self, *outcomes, existing=None):
        self.outcomes = list(outcomes)
        self.rows = list(existing or [])
        self.builds = []
        self.uploads = []
        self.started = []
        self._pending = []

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def list(self, team_id, search, offset, limit):
        if self._pending:
            status = self._pending.pop(0)
            self.rows = [_row(status)]
        return ImageListResponse(
            data=self.rows, totalCount=len(self.rows), offset=offset, limit=limit
        )

    async def initiate_build(self, request):
        self.builds.append(request)
        return SimpleNamespace(build_id=f"b{len(self.builds)}", upload_url="https://up")

    async def start_build(self, build_id):
        self.started.append(build_id)
        self._pending = list(self.outcomes.pop(0))
        self.rows = []


def _row(status, error=None):
    return ImageListItem.model_validate(
        {
            "id": "row",
            "artifactType": "VM_SANDBOX",
            "imageName": "world",
            "imageTag": "v1",
            "status": status,
            "errorMessage": error,
            "createdAt": "2026-10-23T00:00:00Z",
            "displayRef": "prime/team/world:v1",
        }
    )


@pytest.fixture
def registry(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PRIME_TEAM_ID", "team-1")
    monkeypatch.setattr(prime_images, "POLL", 0)

    def install(fake):
        import prime_sandboxes

        monkeypatch.setattr(prime_sandboxes, "AsyncImageClient", fake)

        def upload(request):
            fake.uploads.append(request.read())
            return httpx.Response(200)

        transport = httpx.MockTransport(upload)
        real = httpx.AsyncClient
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw)
        )
        return fake

    return install


@pytest.fixture
def context(tmp_path):
    root = tmp_path / "ctx"
    (root / "src").mkdir(parents=True)
    (root / "Dockerfile").write_text("FROM python:3.12-slim\n")
    (root / "src" / "app.py").write_text("print(1)\n")
    return root


async def test_a_new_tag_is_built_and_waited_for(registry, context):
    fake = registry(FakeRegistry(["PENDING", "BUILDING", "COMPLETED"]))
    image = await push_image(context, "world", "v1")
    assert image.status == "COMPLETED" and image.ref == "prime/team/world:v1"
    assert fake.started == ["b1"]
    request = fake.builds[0]
    assert (request.image_name, request.image_tag, request.team_id) == (
        "world",
        "v1",
        "team-1",
    )
    assert request.visibility.value == "PRIVATE"
    with tarfile.open(fileobj=io.BytesIO(fake.uploads[0])) as archive:
        names = {n.removeprefix("./") for n in archive.getnames()}
    assert {"Dockerfile", "src/app.py"} <= names


async def test_a_built_tag_is_reused(registry, context):
    fake = registry(FakeRegistry(existing=[_row("COMPLETED")]))
    assert (await push_image(context, "world", "v1")).status == "COMPLETED"
    assert fake.builds == []


async def test_a_failed_build_is_pushed_again(registry, context):
    fake = registry(FakeRegistry(["BUILDING", "FAILED"], ["COMPLETED"]))
    assert (await push_image(context, "world", "v1")).status == "COMPLETED"
    assert fake.started == ["b1", "b2"]


async def test_repeated_failure_reports_the_registry_cause(registry, context):
    fake = FakeRegistry(["FAILED"], ["FAILED"])
    registry(fake)
    original = fake.list

    async def with_cause(**kw):
        page = await original(**kw)
        page.data = [_row(r.status.value, "apt-get: 404") for r in page.data]
        return page

    fake.list = with_cause
    with pytest.raises(ImageBuildError, match="2 times: apt-get: 404"):
        await push_image(context, "world", "v1")


async def test_a_build_that_never_finishes_times_out(registry, context):
    registry(FakeRegistry(["BUILDING"] * 1000))
    with pytest.raises(ImageBuildError, match="not bootable"):
        await push_image(context, "world", "v1", timeout=0)


async def test_bad_arguments_are_refused_before_any_build(registry, context):
    fake = registry(FakeRegistry())
    with pytest.raises(ValueError, match="longer than"):
        await push_image(context, "world", "x" * 49)
    with pytest.raises(ValueError, match="not a file"):
        await push_image(context, "world", "v1", dockerfile="Missing")
    assert fake.builds == []


def test_the_digest_follows_the_context(context):
    before = context_digest(context)
    assert context_digest(context) == before
    (context / "src" / "app.py").write_text("print(2)\n")
    assert context_digest(context) != before
