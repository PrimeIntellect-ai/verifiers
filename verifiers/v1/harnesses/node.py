import shlex

from verifiers.v1.harnesses.utils.install import ensure_installed
from verifiers.v1.runtimes import Runtime

NODE_DIR = "/var/tmp/vf-node"
NODE_BIN_DIR = f"{NODE_DIR}/bin"
NODE_VERSION = "22.19.0"
READY = (
    f'[ "$({NODE_BIN_DIR}/node --version 2>/dev/null)" = "v{NODE_VERSION}" ]'
    f" && [ -x {NODE_BIN_DIR}/npm ] && [ -x {NODE_BIN_DIR}/npx ]"
)
INSTALL = f"""# /// script
# requires-python = ">=3.11"
# dependencies = ["nodejs-wheel-binaries=={NODE_VERSION}"]
# ///
from pathlib import Path
import nodejs_wheel

root = Path(nodejs_wheel.__file__).parent
bin_dir = Path("{NODE_BIN_DIR}")
bin_dir.mkdir(parents=True, exist_ok=True)
for name, target in {{
    "node": root / "bin/node",
    "npm": root / "lib/node_modules/npm/bin/npm-cli.js",
    "npx": root / "lib/node_modules/npm/bin/npx-cli.js",
}}.items():
    link = bin_dir / name
    link.unlink(missing_ok=True)
    link.symlink_to(target)
"""


async def ensure_node(runtime: Runtime) -> None:
    """Install pinned Node/npm wheels for the runtime's platform, including musl."""
    if (await runtime.run(["sh", "-c", READY], {})).exit_code == 0:
        return
    program = await runtime.prepare_uv_script(
        INSTALL, {"UV_FROZEN": "false", "UV_OFFLINE": "false"}, activate=False
    )
    await ensure_installed(
        runtime,
        directory=NODE_DIR,
        lock=f"{NODE_DIR}.install.lock",
        ready=READY,
        install=f"{shlex.join(program)} && {READY}",
        env={},
        label="Node.js",
    )
