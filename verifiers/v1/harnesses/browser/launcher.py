"""Browser process lifecycle for the self-launch fallback, kept out of the way.

`harness.py` and `program.py` stay readable by delegating everything about
starting, finding, reusing, and stopping a browser to this module. It is
exercised only in fallback mode -- when no `cdp_url` is supplied; attach mode
never calls in here.

`ensure` runs one idempotent shell step: reuse the browser recorded for this
state dir if it is still alive, otherwise find a Chromium, start it on a
kernel-chosen port, and read the port back from Chrome's own `DevTools listening
on ...` stderr line. Reading the port from Chrome closes the window between
choosing a port and binding it, and needs no `DevToolsActivePort` file headless
Chrome omits. The endpoint and PID are written into the state dir so a `resume`
reuses the same browser and `teardown` stops it by recorded PID -- no state is
held across calls in this process.
"""

from __future__ import annotations

import re

from verifiers.v1.runtimes import Runtime

# The same wait the bash/null family gives a slow external dependency: the
# OpenAI SDK read timeout its `MCP_TIMEOUT` uses. A cold container's first
# Chromium launch off the ~2GB Playwright image can take far longer than a
# handful of seconds, and the failure is asymmetric -- a premature readiness
# timeout kills a good rollout, while a generous one only delays the error on a
# launch that was never going to work.
READY_TIMEOUT = 600

# POSIX sh, run as `sh -c SCRIPT sh <state-dir> <timeout>`. Reuse-or-launch in
# one step; the endpoint and PID live in the state dir so nothing is cached in
# the calling process. Discovery order is BH_CHROME_PATH / CHROME_PATH, then a
# Playwright registry, then PATH -- an image that ships browsers through
# Playwright puts nothing on PATH.
ENSURE_SCRIPT = r"""
state="$1"; timeout="$2"
profile="$state/profile"; log="$state/browser.log"
epf="$state/cdp-endpoint"; pidf="$state/browser.pid"
if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null && [ -f "$epf" ]; then
  echo "ENDPOINT=$(cat "$epf")"
  exit 0
fi
mkdir -p "$profile"
bin=""
for c in "${BH_CHROME_PATH:-}" "${CHROME_PATH:-}"; do
  if [ -n "$c" ] && [ -x "$c" ]; then bin="$c"; break; fi
done
if [ -z "$bin" ] && [ -n "${PLAYWRIGHT_BROWSERS_PATH:-}" ]; then
  bin="$(ls -1 "$PLAYWRIGHT_BROWSERS_PATH"/chromium-*/chrome-linux*/chrome 2>/dev/null | sort | tail -1)"
fi
if [ -z "$bin" ]; then
  for n in google-chrome-stable google-chrome chromium chromium-browser; do
    p="$(command -v "$n" 2>/dev/null)" && { bin="$p"; break; }
  done
fi
if [ -z "$bin" ]; then
  echo "no Chromium/Chrome found; run on a browser-capable image (e.g. mcr.microsoft.com/playwright/python) or set BH_CHROME_PATH" >&2
  exit 3
fi
nohup "$bin" --remote-debugging-port=0 --user-data-dir="$profile" --headless \
  --no-first-run --no-default-browser-check --disable-dev-shm-usage --no-sandbox \
  >"$log" 2>&1 &
pid=$!
echo "$pid" >"$pidf"
i=0
while [ "$i" -lt "$timeout" ]; do
  port="$(sed -n 's#.*DevTools listening on ws://127.0.0.1:\([0-9][0-9]*\)/.*#\1#p' "$log" | head -1)"
  if [ -n "$port" ]; then
    endpoint="http://127.0.0.1:$port"
    echo "$endpoint" >"$epf"
    echo "ENDPOINT=$endpoint"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "browser exited before announcing a port; log tail:" >&2
    tail -c 2000 "$log" >&2
    exit 4
  fi
  sleep 1
  i=$((i + 1))
done
kill "$pid" 2>/dev/null || true
echo "browser did not announce a DevTools port within ${timeout}s; log tail:" >&2
tail -c 2000 "$log" >&2
exit 5
"""

# The daemon records its own PID under BH_HOME=<state>/bh-home (default name),
# so teardown stops it by that PID rather than pattern-matching.
_DAEMON_PID = "bh-home/runtime/bu-default.pid"

_ENDPOINT = re.compile(r"^ENDPOINT=(.+)$", re.MULTILINE)


class BrowserLaunchError(RuntimeError):
    pass


def ensure_argv(state_dir: str, timeout: int = READY_TIMEOUT) -> list[str]:
    return ["sh", "-c", ENSURE_SCRIPT, "sh", state_dir, str(timeout)]


def teardown_argv(state_dir: str) -> list[str]:
    """Stop what this state dir owns and drop it, by recorded PID: the launched
    browser if there was one and the browser-harness daemon, then the dir. In
    attach mode no browser PID was recorded, so the browser is never touched."""
    pids = f'"{state_dir}/browser.pid" "{state_dir}/{_DAEMON_PID}"'
    script = (
        f'for f in {pids}; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; '
        f'rm -rf "{state_dir}" 2>/dev/null || true'
    )
    return ["sh", "-c", script]


async def ensure(runtime: Runtime, state_dir: str, timeout: int = READY_TIMEOUT) -> str:
    """Reuse or start a browser in `runtime`; return its DevTools endpoint."""
    result = await runtime.run(ensure_argv(state_dir, timeout), {})
    match = _ENDPOINT.search(result.stdout)
    if result.exit_code != 0 or not match:
        raise BrowserLaunchError(
            (result.stderr or result.stdout or "browser launch failed").strip()[-2000:]
        )
    return match.group(1).strip()
