from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.runtimes import Runtime

NODE_DIR = "/var/tmp/vf-node"
NODE_BIN_DIR = f"{NODE_DIR}/bin"
NODE_VERSION = "22.19.0"

INSTALL = r"""
set -e
node=/var/tmp/vf-node
node_ok() { "$node/bin/node" -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=19 ? 0 : 1)'; }

if [ -f /etc/alpine-release ]; then
    # apk installs into /usr/bin. Images such as node:*-alpine put their own node in
    # /usr/local/bin, ahead on PATH, so check and link apk's by path.
    apk add --no-cache curl ca-certificates nodejs-current npm >/dev/null
    if ! /usr/bin/node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22 && b>=19 ? 0 : 1)'; then
        sed -E -i 's/v[0-9]+\.[0-9]+/v3.22/g' /etc/apk/repositories
        apk upgrade --available --no-cache >/dev/null
        apk add --no-cache nodejs-current npm >/dev/null
    fi
    mkdir -p "$node/bin"
    ln -sf /usr/bin/node "$node/bin/node"
    ln -sf /usr/bin/npm "$node/bin/npm"
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
        url="https://nodejs.org/dist/v$VF_NODE_VERSION/node-v$VF_NODE_VERSION-${node_os}-${node_arch}.tar.gz"
        # Fetch the whole archive before unpacking it, so that curl's exit code catches
        # a dropped connection, and retry: one failed download should not end the rollout.
        attempt=1
        until curl -fsSL "$url" -o "$node.tar.gz"; do
            if [ "$attempt" -ge 5 ]; then
                rm -f "$node.tar.gz"
                echo "Node.js download failed after $attempt attempts: $url" >&2
                exit 1
            fi
            attempt=$((attempt + 1))
            sleep "$attempt"
        done
        rm -rf "$node"
        mkdir -p "$node"
        tar -xzf "$node.tar.gz" -C "$node" --strip-components=1
        rm -f "$node.tar.gz"
    fi
fi
node_ok || { echo "ACP adapters require Node.js 22.19 or newer" >&2; exit 1; }
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
