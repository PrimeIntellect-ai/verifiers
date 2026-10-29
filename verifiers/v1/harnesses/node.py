from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.runtimes import Runtime

NODE_DIR = "/var/tmp/vf-node"
NODE_BIN_DIR = f"{NODE_DIR}/bin"
NODE_VERSION = "22.23.4"

INSTALL = r"""
set -e
node=/var/tmp/vf-node
# Node bundles Brotli; a newer major alone does not prove CVE-2026-111497 is fixed.
node_ok() { "$1" -e 'const [a,b,c]=process.versions.node.split(".").map(Number); const f={22:[23,4],24:[22,1],26:[12,1]}[a]; process.exit(f && (b>f[0] || b===f[0] && c>=f[1]) ? 0 : 1)'; }

if [ -f /etc/alpine-release ]; then
    apk add --upgrade --no-cache curl ca-certificates nodejs-current npm >/dev/null
    mkdir -p "$node/bin"
    # apk installs into /usr/bin. An image-provided /usr/local/bin/node may
    # appear first on PATH, so link the binaries that apk just installed.
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
        # Finish downloading and extracting before replacing the shared installation.
        download_dir=$(mktemp -d "$node.XXXXXX")
        trap 'rm -rf "$download_dir"' EXIT
        curl -fsSL --retry 5 --retry-all-errors "https://nodejs.org/dist/v$VF_NODE_VERSION/node-v$VF_NODE_VERSION-${node_os}-${node_arch}.tar.gz" \
            -o "$download_dir/node.tar.gz"
        mkdir "$download_dir/node"
        tar -xzf "$download_dir/node.tar.gz" -C "$download_dir/node" --strip-components=1
        rm -rf "$node"
        mv "$download_dir/node" "$node"
    fi
fi
node_ok "$node/bin/node" || { echo "ACP adapters require patched Node.js (22.23.4, 24.22.1 or 26.12.1 minimum)" >&2; exit 1; }
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
