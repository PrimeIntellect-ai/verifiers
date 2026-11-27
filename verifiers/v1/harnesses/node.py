"""The shared Node.js runtime of the ACP adapter harnesses.

The host downloads Node's official build for the runtime's platform once, checks it against
the release's SHASUMS256.txt, and writes it into the runtime, so a task image needs neither
curl nor a package manager, nor package mirrors that still serve its release. Node's musl
build needs GCC 13's libstdc++; on an image whose own is older or missing, Node gets Alpine
3.20's libstdc++ and libgcc beside it. Node publishes no musl build for arm64, so Alpine on
arm64 installs Alpine's Node with apk.
"""

import asyncio
import gzip
import hashlib
import io
import os
import tarfile
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import quote

import httpx

from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.runtimes import Runtime
from verifiers.v1.utils.format import format_time
from verifiers.v1.utils.logging import clip, get_logger
from verifiers.v1.utils.paths import CACHE_DIR
from verifiers.v1.utils.retries import retrying

logger = get_logger(__name__)

NODE_DIR = "/var/tmp/vf-node"
NODE_BIN_DIR = f"{NODE_DIR}/bin"
NODE_VERSION = "24.22.3"
NODE_DIST = f"https://nodejs.org/dist/v{NODE_VERSION}"

# GCC 13's libstdc++ and its libgcc, for musl images whose own libstdc++ is older or
# missing: (package, its file, the name Node loads it by, the package's sha256). Alpine
# 3.20's are the oldest that Node's musl build loads; the release is end-of-life, so these
# files no longer change.
ALPINE_PACKAGES = "https://dl-cdn.alpinelinux.org/alpine/v3.20/main/x86_64"
MUSL_LIBS = (
    (
        "libstdc++-13.2.1_git20240309-r1.apk",
        "usr/lib/libstdc++.so.6.0.32",
        "libstdc++.so.6",
        "af0fe894ef5051116e321bf4753a10fffa85abc2b71f30b2e949467775421ace",
    ),
    (
        "libgcc-13.2.1_git20240309-r1.apk",
        "usr/lib/libgcc_s.so.1",
        "libgcc_s.so.1",
        "f348d99e10b5267566afe6f80861661b08cb5aa43a6d4d1c8f1792b5001d0995",
    ),
)

CACHE = CACHE_DIR / "node"
_CACHE_LOCKS: dict[str, asyncio.Lock] = {}

READY = (
    f'[ -x {NODE_DIR}/bin/node ] && [ -x {NODE_DIR}/bin/npm ] && [ "$({NODE_DIR}/bin/node '
    '--version 2>/dev/null)" = "v$AGENT_NODE_VERSION" ]'
)

# The runtime's platform as Node names its builds, and its C library.
PLATFORM = r"""
case "$(uname -s)" in Linux) os=linux ;; Darwin) os=darwin ;; *) os=$(uname -s) ;; esac
case "$(uname -m)" in x86_64|amd64) arch=x64 ;; aarch64|arm64) arch=arm64 ;; *) arch=$(uname -m) ;; esac
libc=glibc
for loader in /lib/ld-musl-*.so.1; do [ -e "$loader" ] && libc=musl; done
echo "$os $arch $libc"
"""

INSTALL = r"""
set -e
node=/var/tmp/vf-node
# Node bundles Brotli; a newer major alone does not prove CVE-2026-111497 is fixed.
node_ok() { "$1" -e 'const [a,b,c]=process.versions.node.split(".").map(Number); const f={24:[22,1],26:[12,1]}[a]; process.exit(f && (b>f[0] || b===f[0] && c>=f[1]) ? 0 : 1)'; }

if [ -n "$AGENT_NODE_ARCHIVE" ]; then
    # Unpack and check the build the host wrote before replacing the shared installation.
    work=$(mktemp -d "$node.XXXXXX")
    trap 'rm -rf "$work"' EXIT
    mkdir "$work/node"
    tar -xzf "$AGENT_NODE_ARCHIVE" -C "$work/node" --strip-components=1
    if [ -n "$AGENT_NODE_LIBS" ] && ! "$work/node/bin/node" --version >/dev/null 2>&1; then
        # The image's libstdc++ is older than GCC 13 or missing: Node loads its own, from
        # beside it. What Node starts inherits LD_LIBRARY_PATH; libstdc++ keeps its ABI.
        mkdir "$work/node/lib/vf-libs"
        tar -xzf "$AGENT_NODE_LIBS" -C "$work/node/lib/vf-libs"
        mv "$work/node/bin/node" "$work/node/lib/vf-libs/node"
        cat > "$work/node/bin/node" <<'EOF'
#!/bin/sh
lib=$(dirname "$0")/../lib/vf-libs
LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$lib/node" "$@"
EOF
        chmod +x "$work/node/bin/node"
    fi
    node_ok "$work/node/bin/node"
    rm -rf "$node"
    mv "$work/node" "$node"
else
    # Node publishes no musl build for arm64; Alpine's own patched Node instead.
    apk add --upgrade --no-cache nodejs-current npm >/dev/null
    mkdir -p "$node/bin"
    # An image-provided /usr/local/bin/node may appear first on PATH.
    ln -sf /usr/bin/node "$node/bin/node"
    ln -sf /usr/bin/npm "$node/bin/npm"
fi
node_ok "$node/bin/node" || { echo "ACP adapters require patched Node.js (24.22.1 or 26.12.1 minimum)" >&2; exit 1; }
"""


async def ensure_node(runtime: Runtime) -> None:
    """Install the shared Node runtime used by ACP adapter harnesses."""
    env = {"AGENT_NODE_VERSION": NODE_VERSION}
    if (await runtime.run(["sh", "-c", READY], env)).exit_code == 0:
        return
    probe = await runtime.run(["sh", "-c", PLATFORM], {})
    if probe.exit_code:
        raise RuntimeError(
            f"Node.js platform probe failed: {clip(probe.stderr or probe.stdout)}"
        )
    os_name, arch, libc = probe.stdout.split()
    staged: dict[str, str] = {"AGENT_NODE_ARCHIVE": "", "AGENT_NODE_LIBS": ""}
    if (arch, libc) != ("arm64", "musl"):
        build = f"{os_name}-{arch}" + ("-musl" if libc == "musl" else "")
        staged["AGENT_NODE_ARCHIVE"] = await _stage(runtime, await node_archive(build))
        if libc == "musl":
            staged["AGENT_NODE_LIBS"] = await _stage(runtime, await musl_libs())
    try:
        # The install replaces NODE_DIR wholesale, so the lock lives beside it.
        await ensure_installed(
            runtime,
            directory=NODE_DIR,
            lock=f"{NODE_DIR}.install.lock",
            ready=READY,
            install=INSTALL,
            env={**env, **staged},
            label="Node.js",
        )
    finally:
        if paths := [path for path in staged.values() if path]:
            await runtime.run(["rm", "-f", *paths], {})


async def node_archive(build: str) -> Path:
    """Node's official build for `build` (e.g. `linux-x64-musl`), downloaded once into the
    host's cache and checked against the release's SHASUMS256.txt."""
    name = f"node-v{NODE_VERSION}-{build}.tar.gz"

    async def fetch(client: httpx.AsyncClient) -> bytes:
        sums = (await _get(client, f"{NODE_DIST}/SHASUMS256.txt")).decode()
        expected = [
            line.split()[0] for line in sums.splitlines() if line.endswith(f"  {name}")
        ]
        if not expected:
            raise ValueError(f"Node.js {NODE_VERSION} publishes no {build} build")
        data = await _get(client, f"{NODE_DIST}/{name}")
        _check(name, data, expected[0])
        return data

    return await _cached(name, fetch)


async def musl_libs() -> Path:
    """GCC 13's libstdc++ and libgcc from Alpine 3.20, as a tar.gz of the two libraries."""

    async def fetch(client: httpx.AsyncClient) -> bytes:
        bundle = io.BytesIO()
        with tarfile.open(fileobj=bundle, mode="w:gz") as out:
            for package, member, soname, sha256 in MUSL_LIBS:
                data = await _get(client, f"{ALPINE_PACKAGES}/{quote(package)}")
                _check(package, data, sha256)
                # An apk is three gzip members (signature, control, data), each a tar
                # stream with its own end marker.
                with tarfile.open(
                    fileobj=io.BytesIO(gzip.decompress(data)), ignore_zeros=True
                ) as apk:
                    content = apk.extractfile(member)
                    if content is None:
                        raise ValueError(f"{package} holds no file {member}")
                    library = content.read()
                info = tarfile.TarInfo(soname)
                info.size, info.mode = len(library), 0o755
                out.addfile(info, io.BytesIO(library))
        return bundle.getvalue()

    return await _cached("musl-libs-alpine3.20.tar.gz", fetch)


async def _stage(runtime: Runtime, path: Path) -> str:
    """Write a cached file into the runtime, beside NODE_DIR, and return its path there."""
    target = f"{NODE_DIR}.{uuid.uuid4().hex}.{path.name}"
    await runtime.write(target, path.read_bytes())
    return target


async def _cached(
    name: str, fetch: Callable[[httpx.AsyncClient], Awaitable[bytes]]
) -> Path:
    """`CACHE/name`, fetched once per host. Concurrent callers in a process wait for one
    download; processes that race each publish a complete file."""
    path = CACHE / name
    async with _CACHE_LOCKS.setdefault(name, asyncio.Lock()):
        if path.is_file():
            return path
        start = time.monotonic()
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=httpx.Timeout(300.0, connect=30.0)
        ) as client:
            async for attempt in retrying(
                retries=3, give_up=ValueError, label=f"download of {name}"
            ):
                with attempt:
                    data = await fetch(client)
        CACHE.mkdir(parents=True, exist_ok=True)
        partial = CACHE / f".{name}.{os.getpid()}.{uuid.uuid4().hex}"
        partial.write_bytes(data)
        partial.replace(path)
        logger.info("Downloaded %s in %s", name, format_time(time.monotonic() - start))
    return path


async def _get(client: httpx.AsyncClient, url: str) -> bytes:
    response = await client.get(url)
    response.raise_for_status()
    return response.content


def _check(name: str, data: bytes, sha256: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise ValueError(f"{name} has sha256 {actual}, expected {sha256}")
