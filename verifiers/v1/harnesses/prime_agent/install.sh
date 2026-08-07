
set -eu
root="$VF_PA_INSTALL_DIR"
digest="$VF_PA_TARBALL_SHA256"
stamp="$root/.installed"
node_root="$VF_PA_NODE_ROOT"
uv_root="${root}.uv"
uv_bin="$uv_root/bin/uv"
# Pin the installer input so all rollouts resolve the same uv artifact.
uv_version="${VF_PA_UV_VERSION:-0.8.17}"
export PATH="$uv_root/bin:$node_root/bin:$PATH"

ensure_base_tools() {
    # curl fetches Node/uv/the tarball. git is not needed by prime-agent itself,
    # but a coding taskset that clones or diffs fails deep inside a rollout
    # without it, which reads as a bad score rather than a missing dependency.
    missing=""
    command -v curl >/dev/null 2>&1 || missing="$missing curl"
    command -v git >/dev/null 2>&1 || missing="$missing git"
    [ -z "$missing" ] && return 0
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y --no-install-recommends ca-certificates $missing
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache ca-certificates $missing
    else
        echo "prime-agent install needs$missing; neither apt-get nor apk is available" >&2
        exit 1
    fi
    command -v curl >/dev/null 2>&1 || {
        echo "prime-agent install requires curl, but installation did not provide it" >&2
        exit 1
    }
}

node_ok() {
    command -v node >/dev/null 2>&1 || return 1
    command -v npm >/dev/null 2>&1 || return 1
    node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=8)?0:1)'
}

bundled_node_ok() {
    [ -x "$node_root/bin/node" ] || return 1
    [ -x "$node_root/bin/npm" ] || return 1
    [ "$("$node_root/bin/node" --version 2>/dev/null)" = "v$VF_PA_NODE_VERSION" ] || return 1
    "$node_root/bin/npm" --version >/dev/null 2>&1
}

ensure_base_tools

# The Node.js release archives are glibc-linked and cannot execute on Alpine/musl.
# Prefer the image's distro Node rather than downloading an unusable archive.
if [ "$(uname -s)" = Linux ] && ldd --version 2>&1 | grep -qi musl; then
    if ! node_ok; then
        if command -v apk >/dev/null 2>&1; then
            apk add --no-cache nodejs-current npm
        fi
    fi
    node_ok || {
        echo "prime-agent: Alpine/musl requires nodejs-current and npm; install them in the image" >&2
        exit 1
    }
fi

install_uv() {
    if [ -x "$uv_bin" ] && "$uv_bin" --version 2>/dev/null | grep -F "uv $uv_version" >/dev/null 2>&1; then
        return 0
    fi
    tmp="${uv_root}.staging.$$"
    rm -rf "$tmp"
    mkdir -p "$tmp/bin"
    # The official installer honors UV_VERSION and UV_INSTALL_DIR. Download it
    # while setup still has network, then publish the verified executable atomically.
    # Pin through the versioned installer URL: the generic installer ignores
    # UV_VERSION (verified -- it installed 0.12.2 when asked for 0.8.17), which
    # both defeats the pin and makes the cache check below never match, so every
    # setup re-downloads uv. UV_NO_MODIFY_PATH keeps it from writing shell rc
    # files into the per-trace HOME.
    if ! curl -fsSL "https://astral.sh/uv/${uv_version}/install.sh" \
        | UV_INSTALL_DIR="$tmp/bin" UV_NO_MODIFY_PATH=1 sh; then
        echo "prime-agent: official uv installer failed for uv $uv_version" >&2
        exit 1
    fi
    if [ ! -x "$tmp/bin/uv" ]; then
        echo "prime-agent: official uv installer did not provide $tmp/bin/uv" >&2
        exit 1
    fi
    rm -rf "$uv_root"
    mv "$tmp" "$uv_root"
}

verify_uv() {
    [ -x "$uv_bin" ] || { echo "prime-agent: uv missing at $uv_bin" >&2; exit 1; }
    actual_uv="$($uv_bin --version 2>&1)" || { echo "prime-agent: uv --version failed after installation: $actual_uv" >&2; exit 1; }
    printf '%s\n' "$actual_uv" | grep -F "uv $uv_version" >/dev/null 2>&1 || { echo "prime-agent: uv version mismatch; expected uv $uv_version, got $actual_uv" >&2; exit 1; }
}

install_uv() {
    if [ -x "$uv_bin" ] && "$uv_bin" --version 2>/dev/null | grep -F "uv $uv_version" >/dev/null 2>&1; then
        return 0
    fi
    tmp="${uv_root}.staging.$$"
    rm -rf "$tmp"
    mkdir -p "$tmp/bin"
    # The official installer honors UV_VERSION and UV_INSTALL_DIR. Download it
    # while setup still has network, then publish the verified executable atomically.
    if ! curl -fsSL https://astral.sh/uv/install.sh | UV_VERSION="$uv_version" UV_INSTALL_DIR="$tmp/bin" sh; then
        echo "prime-agent: official uv installer failed for uv $uv_version" >&2
        exit 1
    fi
    if [ ! -x "$tmp/bin/uv" ]; then
        echo "prime-agent: official uv installer did not provide $tmp/bin/uv" >&2
        exit 1
    fi
    rm -rf "$uv_root"
    mv "$tmp" "$uv_root"
}

verify_uv() {
    [ -x "$uv_bin" ] || { echo "prime-agent: uv missing at $uv_bin" >&2; exit 1; }
    actual_uv="$($uv_bin --version 2>&1)" || { echo "prime-agent: uv --version failed after installation: $actual_uv" >&2; exit 1; }
    printf '%s\n' "$actual_uv" | grep -F "uv $uv_version" >/dev/null 2>&1 || { echo "prime-agent: uv version mismatch; expected uv $uv_version, got $actual_uv" >&2; exit 1; }
}

if ! node_ok; then
    case "$(uname -s)" in
        Linux) node_os=linux ;;
        Darwin) node_os=darwin ;;
        *) echo "prime-agent: unsupported OS $(uname -s)" >&2; exit 1 ;;
    esac
    # Reject unknown machines instead of guessing x64: a wrong archive yields a
    # node binary that cannot exec, failing much later and far less clearly.
    case "$(uname -m)" in
        aarch64|arm64) node_arch=arm64 ;;
        x86_64|amd64) node_arch=x64 ;;
        *) echo "prime-agent: unsupported architecture $(uname -m)" >&2; exit 1 ;;
    esac
    if ! bundled_node_ok; then
        rm -rf "$node_root"
        mkdir -p "$node_root"
        curl -fsSL "https://nodejs.org/dist/v$VF_PA_NODE_VERSION/node-v$VF_PA_NODE_VERSION-${node_os}-${node_arch}.tar.gz" \
            | tar -xz -C "$node_root" --strip-components=1
    fi
    node_ok || { echo "prime-agent requires Node.js 22.8 or newer with npm" >&2; exit 1; }
fi

install_uv
verify_uv

# The install is shared, so key the stamp on the verified digest: a changed
# version or tarball must reinstall rather than reuse another rollout's tree.
if [ -x "$root/node_modules/.bin/prime-agent" ] \
    && [ "$(cat "$stamp" 2>/dev/null)" = "$digest" ] \
    && [ -x "$uv_bin" ] \
    && "$uv_bin" --version 2>/dev/null | grep -F "uv $uv_version" >/dev/null 2>&1; then
    exit 0
fi

staging="${root}.staging.$$"
rm -rf "$staging"
mkdir -p "$staging"
cleanup() { rm -rf "$staging"; }
trap cleanup EXIT

curl -fsSL "$VF_PA_TARBALL_URL" -o "$staging/prime-agent.tgz"
if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$staging/prime-agent.tgz" | cut -d' ' -f1)"
else
    actual="$(shasum -a 256 "$staging/prime-agent.tgz" | cut -d' ' -f1)"
fi
if [ "$actual" != "$digest" ]; then
    echo "prime-agent tarball digest mismatch: expected $digest, got $actual" >&2
    exit 1
fi

npm install --no-audit --no-fund --prefix "$staging" "$staging/prime-agent.tgz" >/dev/null
rm -f "$staging/prime-agent.tgz"
printf %s "$digest" > "$staging/.installed"

# Publish atomically: a partially installed tree must never be observable, and
# a concurrent rollout either sees the old tree or the complete new one.
rm -rf "${root}.prev"
if [ -d "$root" ]; then
    mv "$root" "${root}.prev"
fi
if ! mv "$staging" "$root"; then
    if [ -d "${root}.prev" ]; then mv "${root}.prev" "$root"; fi
    echo "prime-agent: failed to publish install" >&2
    exit 1
fi
rm -rf "${root}.prev"

