import json
import shlex
from hashlib import sha256

from verifiers.v1.runtimes import Runtime

NODE_DIR = "/var/tmp/vf-node"
NODE_BIN_DIR = f"{NODE_DIR}/bin"
NODE_VERSION = "22.19.0"
PATCH_SOURCE = r"""
const fs = require("fs");
const [target] = process.argv.slice(2);
let source = fs.readFileSync(target, "utf8");
for (const [before, after] of __REPLACEMENTS__) {
    const start = source.indexOf(before);
    if (start < 0 || source.indexOf(before, start + 1) >= 0) {
        throw new Error(`patch source mismatch: ${target}`);
    }
    source = source.slice(0, start) + after + source.slice(start + before.length);
}
fs.writeFileSync(target, source);
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
node_ok || { echo "ACP adapters require Node.js 22.19 or newer" >&2; exit 1; }
"""


async def ensure_node(runtime: Runtime) -> None:
    """Install the shared Node runtime used by ACP adapter harnesses."""
    lock = f"{NODE_DIR}.install.lock"
    guarded = (
        f'until ln -s "$$" {lock} 2>/dev/null; do '
        f"owner=$(readlink {lock}); "
        f'if ! kill -0 "$owner" 2>/dev/null; then '
        f'[ "$(readlink {lock})" != "$owner" ] || rm -f {lock}; fi; '
        f"sleep 0.1; done; "
        f'trap \'[ "$(readlink {lock})" != "$$" ] || rm -f {lock}\' EXIT; '
        f"sh -c {shlex.quote(INSTALL)}"
    )
    result = await runtime.run(["sh", "-c", guarded], {"VF_NODE_VERSION": NODE_VERSION})
    if result.exit_code != 0:
        raise RuntimeError(f"Node.js install failed: {result.stderr.strip()[-500:]}")


def node_patch_id(patch: bytes) -> str:
    return sha256(patch).hexdigest()[:12]


def _patch_replacements(patch: bytes) -> list[tuple[str, str]]:
    replacements = []
    before: list[str] | None = None
    after: list[str] | None = None
    for line in patch.decode().splitlines(keepends=True):
        if line.startswith("@@"):
            if before is not None and after is not None:
                replacements.append(("".join(before), "".join(after)))
            before, after = [], []
        elif before is not None and after is not None:
            if line.startswith("\\"):
                continue
            if line[:1] in (" ", "-"):
                before.append(line[1:])
            if line[:1] in (" ", "+"):
                after.append(line[1:])
    if before is not None and after is not None:
        replacements.append(("".join(before), "".join(after)))
    if not replacements:
        raise ValueError("Node package patch contains no hunks")
    return replacements


async def prepare_node_patch(
    runtime: Runtime, directory: str, target: str, patch: bytes
) -> str:
    """Upload a source-checked Node patch command and return its shell prefix."""
    revision = sha256(patch).hexdigest()
    script = f"{directory}/apply-patch-{revision}.js"
    await runtime.write(
        script,
        PATCH_SOURCE.replace(
            "__REPLACEMENTS__", json.dumps(_patch_replacements(patch))
        ).encode(),
    )
    return " ".join(map(shlex.quote, (f"{NODE_BIN_DIR}/node", script, target)))
