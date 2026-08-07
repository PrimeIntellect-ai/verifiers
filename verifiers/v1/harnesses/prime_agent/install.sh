
set -eu
root="$VF_PA_INSTALL_DIR"
digest="$VF_PA_TARBALL_SHA256"
stamp="$root/.installed"
node_root="$VF_PA_NODE_ROOT"
export PATH="$node_root/bin:$PATH"

ensure_curl() {
    if command -v curl >/dev/null 2>&1; then
        return 0
    fi
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y --no-install-recommends curl
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl
    else
        echo "prime-agent install requires curl; neither apt-get nor apk is available" >&2
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

ensure_curl

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

# The install is shared, so key the stamp on the verified digest: a changed
# version or tarball must reinstall rather than reuse another rollout's tree.
if [ -x "$root/node_modules/.bin/prime-agent" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$digest" ]; then
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
