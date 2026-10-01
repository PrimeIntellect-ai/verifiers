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


async def ensure_node(runtime: Runtime) -> None:
    """Install the shared Node runtime used by ACP adapter harnesses."""
    # The install replaces NODE_DIR wholesale, so the lock lives beside it.
    await ensure_installed(
        await runtime.with_user("root") if runtime.user is not None else runtime,
        directory=NODE_DIR,
        lock=f"{NODE_DIR}.install.lock",
        ready=f"{NODE_BIN_DIR}/node -e 'const [a,b]=process.versions.node.split(\".\").map(Number); process.exit(a>22 || a===22 && b>=19 ? 0 : 1)'",
        install=INSTALL,
        env={"VF_NODE_VERSION": NODE_VERSION},
        label="Node.js",
    )
