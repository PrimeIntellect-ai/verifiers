import asyncio
import hashlib
import os
import urllib.request
import uuid
from pathlib import Path

from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.runtimes import Runtime

NODE_DIR = "/var/tmp/vf-node"
NODE_BIN_DIR = f"{NODE_DIR}/bin"
NODE_VERSION = "22.19.0"
NODE_SHA256 = {
    "x64": "d36e56998220085782c0ca965f9d51b7726335aed2f5fc7321c6c0ad233aa96d",
    "arm64": "d32817b937219b8f131a28546035183d79e7fd17a86e38ccb8772901a7cd9009",
}
"""Digests of the pinned linux tarballs, from nodejs.org's SHASUMS256.txt."""
NPM_ENV = r"""
export PATH="/var/tmp/vf-node/bin:$PATH"
# A task image keeps its solver off npm with its own npm settings (ENV
# NPM_CONFIG_OFFLINE=true, NPM_CONFIG_REGISTRY=http://127.0.0.1:9, an .npmrc), and a
# harness install inherited them: `ENOTCACHED ... cache mode is 'only-if-cached'`. The
# install runs before networking is restricted, so it drops them and uses npm's public
# registry and a cache of its own; nothing the solver runs afterwards sees this. The
# user and global config files are two paths that do not exist (npm refuses one file
# loaded as both: "double-loading config").
for var in $(env | sed -n 's/^\([Nn][Pp][Mm]_[Cc][Oo][Nn][Ff][Ii][Gg]_[A-Za-z0-9_]*\)=.*/\1/p'); do
    unset "$var"
done
export npm_config_userconfig=/var/tmp/vf-npm/no-user-npmrc \
    npm_config_globalconfig=/var/tmp/vf-npm/no-global-npmrc \
    npm_config_registry=https://registry.npmjs.org/ npm_config_offline=false \
    npm_config_prefer_offline=false npm_config_cache=/var/tmp/vf-npm/cache
"""
"""The start of every harness's npm install script: the harness's Node on PATH, and npm
free of the task image's npm configuration."""
_ARCHES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
_lock = asyncio.Lock()

# A hardened image strips every way to fetch Node (no curl, no apt-get, not Alpine): the
# host uploads the pinned tarball and the box only unpacks it.
CAN_FETCH = (
    "[ -x /var/tmp/vf-node/bin/node ] || [ -f /etc/alpine-release ] "
    "|| command -v curl >/dev/null 2>&1 || command -v apt-get >/dev/null 2>&1"
)
UNPACK = r"""
set -e
node=/var/tmp/vf-node
rm -rf "$node"
mkdir -p "$node"
tar -xzf "$VF_NODE_TARBALL" -C "$node" --strip-components=1
rm -f "$VF_NODE_TARBALL"
"$node/bin/node" -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=19 ? 0 : 1)'
"""

INSTALL = r"""
set -e
node=/var/tmp/vf-node
node_ok() { "$node/bin/node" -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=19 ? 0 : 1)'; }

if [ -f /etc/alpine-release ]; then
    apk add --no-cache curl ca-certificates nodejs-current npm >/dev/null
    if ! node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=19 ? 0 : 1)'; then
        sed -E -i 's/v[0-9]+\.[0-9]+/v3.22/g' /etc/apk/repositories
        apk upgrade --available --no-cache >/dev/null
        apk add --no-cache nodejs-current npm >/dev/null
    fi
    mkdir -p "$node/bin"
    ln -sf "$(command -v node)" "$node/bin/node"
    ln -sf "$(command -v npm)" "$node/bin/npm"
else
    if ! command -v curl >/dev/null 2>&1 \
        && ! (apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null); then
        (
            # Task images can have source-only repositories. Bootstrap from signed
            # distro repositories with temporary sources and indexes, leaving the task's intact.
            . /etc/os-release
            case "$ID" in
                debian)
                    mirror=http://deb.debian.org/debian
                    security_mirror=http://security.debian.org/debian-security ;;
                ubuntu)
                    case "$(dpkg --print-architecture)" in
                        amd64|i386)
                            mirror=http://archive.ubuntu.com/ubuntu
                            security_mirror=http://security.ubuntu.com/ubuntu ;;
                        *)
                            mirror=http://ports.ubuntu.com/ubuntu-ports
                            security_mirror=$mirror ;;
                    esac ;;
                *) echo "cannot bootstrap curl on $ID" >&2; exit 1 ;;
            esac
            apt_dir=$(mktemp -d)
            trap 'rm -rf "$apt_dir"' EXIT
            chmod 755 "$apt_dir"
            mkdir -p "$apt_dir/lists/partial"
            set -- -o "Dir::Etc::sourcelist=$apt_dir/sources.list" -o Dir::Etc::sourceparts=- \
                -o "Dir::State::lists=$apt_dir/lists" -o Dir::Cache::pkgcache= -o Dir::Cache::srcpkgcache=
            suites=${VERSION_CODENAME:?}
            # Testing and unstable share os-release; sid may be needed to match installed libcurl.
            if [ "$ID" = debian ] && [ -z "${VERSION_ID:-}" ]; then suites="$suites sid"; fi
            for suite in $suites; do
                printf 'deb %s %s main\n' "$mirror" "$suite" > "$apt_dir/sources.list"
                # Release images can already have libcurl from updates/security.
                if [ -n "${VERSION_ID:-}" ]; then
                    printf 'deb %s %s main\n' "$mirror" "$suite-updates" \
                        "$security_mirror" "$suite-security" >> "$apt_dir/sources.list"
                fi
                apt-get "$@" update -qq \
                    && apt-get "$@" install -y -qq --no-install-recommends curl ca-certificates >/dev/null \
                    && exit 0
            done
            exit 1
        )
    fi
    case "$(uname -s)" in Linux) node_os=linux ;; Darwin) node_os=darwin ;; *) echo "unsupported os: $(uname -s)" >&2; exit 1 ;; esac
    if [ ! -x "$node/bin/node" ] || [ "$("$node/bin/node" --version 2>/dev/null)" != "v$VF_NODE_VERSION" ]; then
        case "$(uname -m)" in aarch64|arm64) node_arch=arm64 ;; *) node_arch=x64 ;; esac
        rm -rf "$node"
        mkdir -p "$node"
        curl -fsSL "https://nodejs.org/dist/v$VF_NODE_VERSION/node-v$VF_NODE_VERSION-${node_os}-${node_arch}.tar.gz" \
            | tar -xz -C "$node" --strip-components=1
    fi
fi
node_ok || { echo "ACP adapters require Node.js 22.19 or newer" >&2; exit 1; }
"""


def _cache_path(arch: str) -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return (
        Path(base) / "verifiers" / "node" / NODE_VERSION / f"node-linux-{arch}.tar.gz"
    )


def _download(arch: str) -> bytes:
    url = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-{arch}.tar.gz"
    with urllib.request.urlopen(url, timeout=300) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != NODE_SHA256[arch]:
        raise RuntimeError(f"{url}: sha256 {digest}, expected {NODE_SHA256[arch]}")
    return data


async def host_node_tarball(machine: str) -> bytes:
    """The pinned Node tarball for a box whose `uname -m` is `machine`, from the host
    cache, downloading it on first use."""
    arch = _ARCHES.get(machine.strip().lower())
    if arch is None:
        raise RuntimeError(f"no Node.js build for box architecture {machine!r}")
    path = _cache_path(arch)
    async with _lock:
        if not path.is_file():
            data = await asyncio.to_thread(_download, arch)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
            partial.write_bytes(data)
            partial.replace(path)
        return await asyncio.to_thread(path.read_bytes)


async def _from_host(runtime: Runtime) -> None:
    machine = await runtime.run(["uname", "-m"], {})
    if machine.exit_code:
        raise RuntimeError(f"failed to read box architecture: {machine.stderr}")
    tarball = f"{NODE_DIR}.{uuid.uuid4().hex}.tar.gz"
    await runtime.write(tarball, await host_node_tarball(machine.stdout))
    await ensure_installed(
        runtime,
        directory=NODE_DIR,
        lock=f"{NODE_DIR}.install.lock",
        install=UNPACK,
        env={"VF_NODE_TARBALL": tarball},
        label="Node.js (from the host)",
    )


async def ensure_node(runtime: Runtime) -> None:
    """Install the shared Node runtime used by ACP adapter harnesses: fetched by the box,
    or uploaded from the host when the box has no way to fetch it or its own install
    fails (an image whose apt-get is there but its dpkg is broken)."""
    probe = await runtime.run(["sh", "-c", CAN_FETCH], {})
    if probe.exit_code != 0:
        await _from_host(runtime)
        return
    # The install replaces NODE_DIR wholesale, so the lock lives beside it.
    try:
        await ensure_installed(
            runtime,
            directory=NODE_DIR,
            lock=f"{NODE_DIR}.install.lock",
            install=INSTALL,
            env={"VF_NODE_VERSION": NODE_VERSION},
            label="Node.js",
        )
    except RuntimeError:
        await _from_host(runtime)
