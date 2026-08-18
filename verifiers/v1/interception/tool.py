"""Bootstrap for native harness tool hooks: private one-shot credential delivery."""

import json
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime


async def install_tool_hook(
    runtime: "Runtime",
    path: str,
    source: bytes,
    url: str,
    secret: str,
) -> dict[str, str]:
    """Write a native-hook bridge at `path` plus a sibling one-shot credentials file,
    and return the env pointer the hook consumes (it deletes the file on first read).

    The credentials never cross argv or the harness env: the random file name cannot
    be pre-seeded, shell noclobber makes creation exclusive, and the payload arrives
    over stdin."""
    await runtime.write(path, source)
    credentials_path = f"{path}.{uuid.uuid4().hex}.credentials"
    payload = json.dumps({"url": url, "secret": secret}).encode()
    result = await runtime.run_with_input(
        [
            "sh",
            "-c",
            'umask 077; set -C; head -c "$1" > "$2"',
            "write-tool-credentials",
            str(len(payload)),
            credentials_path,
        ],
        {},
        payload,
    )
    if result.exit_code != 0:
        raise RuntimeError(
            "failed to write tool interception credentials privately: "
            f"{result.stderr.strip()[-500:]}"
        )
    return {"VF_TOOL_INTERCEPTION_CONFIG": credentials_path}
