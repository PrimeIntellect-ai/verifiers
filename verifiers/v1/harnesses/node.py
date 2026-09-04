from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.runtimes import Runtime

NODE_DIR = "/var/tmp/vf-node"
NODE_BIN_DIR = f"{NODE_DIR}/bin"
# This is the first 22.x release whose global fetch honors environment proxies.
NODE_VERSION = "22.21.0"

INSTALL = r"""
set -e
node=/var/tmp/vf-node
node_ok() { "$node/bin/node" -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=21 ? 0 : 1)'; }

if [ -f /etc/alpine-release ]; then
    apk add --no-cache curl ca-certificates nodejs-current npm >/dev/null
    if ! node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=21 ? 0 : 1)'; then
        sed -E -i 's/v[0-9]+\.[0-9]+/v3.22/g' /etc/apk/repositories
        apk upgrade --available --no-cache >/dev/null
        apk add --no-cache nodejs-current npm >/dev/null
    fi
    mkdir -p "$node/bin"
    ln -sf "$(command -v node)" "$node/bin/node"
    ln -sf "$(command -v npm)" "$node/bin/npm"
else
    command -v curl >/dev/null 2>&1 \
        || { apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; }
    case "$(uname -s)" in Linux) node_os=linux ;; Darwin) node_os=darwin ;; *) echo "unsupported os: $(uname -s)" >&2; exit 1 ;; esac
    if [ ! -x "$node/bin/node" ] || [ "$("$node/bin/node" --version 2>/dev/null)" != "v$VF_NODE_VERSION" ]; then
        case "$(uname -m)" in aarch64|arm64) node_arch=arm64 ;; *) node_arch=x64 ;; esac
        rm -rf "$node"
        mkdir -p "$node"
        curl -fsSL "https://nodejs.org/dist/v$VF_NODE_VERSION/node-v$VF_NODE_VERSION-${node_os}-${node_arch}.tar.gz" \
            | tar -xz -C "$node" --strip-components=1
    fi
fi
node_ok || { echo "ACP adapters require Node.js 22.21 or newer" >&2; exit 1; }
"""


async def ensure_node(runtime: Runtime) -> None:
    """Install the shared Node runtime used by ACP adapter harnesses."""
    # The install replaces NODE_DIR wholesale, so the lock lives beside it.
    await ensure_installed(
        runtime,
        directory=NODE_DIR,
        lock=f"{NODE_DIR}.install.lock",
        install=INSTALL,
        env={"VF_NODE_VERSION": NODE_VERSION},
        label="Node.js",
    )
