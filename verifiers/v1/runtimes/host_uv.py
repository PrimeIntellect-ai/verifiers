"""A static uv binary from the host, for boxes that cannot install one themselves.

`_ENSURE_UV` installs uv in the box through pip, curl, wget, apt-get or apk. A hardened
task image strips all of them (`rm -f /usr/bin/apt /usr/bin/apt-get /usr/bin/dpkg`, no
fetchers), so the box has no way to get uv however open its network is. The host then
uploads uv itself: the pinned release's statically linked (musl) build for the box's
architecture, downloaded once, checked against its pinned digest and cached on the host.

Only uv comes from the host. The box's network is still open while a uv script is
prepared (every backend applies the task's egress policy after setup), so `uv sync
--script` resolves the script's dependencies from the index as usual, and uses the
image's own Python when it satisfies the script's `requires-python`.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import tarfile
import urllib.request
from pathlib import Path

UV_VERSION = "0.13.1"
UV_RELEASES = "https://github.com/astral-sh/uv/releases/download"
UV_SHA256 = {
    "x86_64-unknown-linux-musl": "6c86040a8f045ab9adc9e3ec4cdafa03fd00e9a988ed2bf28971c7fc2d0debd0",
    "aarch64-unknown-linux-musl": "345a8308e4d1835cdfaadf692836fd3137a6d45864773f05ad7bf59442b1ba16",
}
"""Digests of the pinned release archives, from the release's `.sha256` files."""

UV_UNAVAILABLE = 97
"""Exit status of a uv script's preparation when the box has no script-capable uv and
could not install one: the cue to upload the host's."""

_ARCHES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}
_lock = asyncio.Lock()


def uv_target(machine: str) -> str:
    """The uv release target for a box's `uname -m`."""
    arch = _ARCHES.get(machine.strip().lower())
    if arch is None:
        raise RuntimeError(f"no static uv build for box architecture {machine!r}")
    return f"{arch}-unknown-linux-musl"


def needs_host_uv(exit_code: int, uploaded: bool) -> bool:
    """Whether a failed preparation should upload the host's uv and try again: only when
    the box could not get uv itself, and only once per preparation. Not remembered per
    box: an agent may delete ~/.local/bin/uv from a box a later run borrows."""
    return exit_code == UV_UNAVAILABLE and not uploaded


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "verifiers" / "uv" / UV_VERSION


def _download(target: str) -> bytes:
    url = f"{UV_RELEASES}/{UV_VERSION}/uv-{target}.tar.gz"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != UV_SHA256[target]:
        raise RuntimeError(f"{url}: sha256 {digest}, expected {UV_SHA256[target]}")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member = tar.getmember(f"uv-{target}/uv")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"{url}: no uv binary in the archive")
        return extracted.read()


async def host_uv_binary(machine: str) -> bytes:
    """The static uv binary for a box whose `uname -m` is `machine`, from the host cache,
    downloading it on first use."""
    target = uv_target(machine)
    path = cache_dir() / target / "uv"
    async with _lock:
        if not path.is_file():
            data = await asyncio.to_thread(_download, target)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"uv.{os.getpid()}.partial")
            partial.write_bytes(data)
            partial.replace(path)
        return await asyncio.to_thread(path.read_bytes)
