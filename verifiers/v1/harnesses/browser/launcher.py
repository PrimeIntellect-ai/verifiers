"""Browser process lifecycle for the self-launch fallback, kept out of the way.

`harness.py` and `program.py` stay readable by delegating everything about
starting, discovering, and stopping a browser to this module. It is exercised
only in fallback mode -- when no `cdp_url` is supplied and the harness starts a
browser of its own; attach mode never calls `launch`.

The launch command finds a Chromium, starts it on a kernel-chosen port, and
reads the port back from Chrome's own `DevTools listening on ...` stderr line --
there is no window between choosing a port and binding it in which another
process could take it, and no dependence on a `DevToolsActivePort` file headless
Chrome omits. It reports the PID so teardown targets it exactly rather than
pattern-matching, and kills the browser if it never announces a port.
"""

from __future__ import annotations

import re

from verifiers.v1.runtimes import ProgramResult, Runtime

READY_TIMEOUT = 30
"""Seconds a launched Chromium gets to announce its DevTools port."""

# POSIX sh, run as `sh -c SCRIPT sh <state-dir> <timeout>`. Discovery order is
# BH_CHROME_PATH / CHROME_PATH, then a Playwright registry, then PATH -- an
# image that ships browsers through Playwright puts nothing on PATH.
LAUNCH_SCRIPT = r"""
state="$1"; timeout="$2"
profile="$state/profile"; log="$state/browser.log"
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
  echo "no Chromium/Chrome found; set BH_CHROME_PATH or install one on PATH / under PLAYWRIGHT_BROWSERS_PATH" >&2
  exit 3
fi
nohup "$bin" --remote-debugging-port=0 --user-data-dir="$profile" --headless \
  --no-first-run --no-default-browser-check --disable-dev-shm-usage --no-sandbox \
  >"$log" 2>&1 &
pid=$!
i=0
while [ "$i" -lt "$timeout" ]; do
  port="$(sed -n 's#.*DevTools listening on ws://127.0.0.1:\([0-9][0-9]*\)/.*#\1#p' "$log" | head -1)"
  if [ -n "$port" ]; then
    echo "PID=$pid"
    echo "ENDPOINT=http://127.0.0.1:$port"
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

# The daemon records its own PID here (BH_HOME=<state>/bh-home, default name),
# so teardown kills it by that PID rather than pattern-matching.
_DAEMON_PID = "bh-home/runtime/bu-default.pid"

_ANNOUNCE = re.compile(r"^(PID|ENDPOINT)=(.+)$", re.MULTILINE)


class BrowserLaunchError(RuntimeError):
    pass


def launch_argv(state_dir: str, timeout: int = READY_TIMEOUT) -> list[str]:
    return ["sh", "-c", LAUNCH_SCRIPT, "sh", state_dir, str(timeout)]


def alive_argv(pid: str) -> list[str]:
    return ["sh", "-c", f"kill -0 {int(pid)} 2>/dev/null"]


def teardown_argv(state_dir: str, browser_pid: str | None) -> list[str]:
    """Stop what this harness owns and drop its state, by recorded PID.

    Always the browser-harness daemon (this harness's only process in attach
    mode) and the state dir; the launched browser too when there was one. In
    attach mode `browser_pid` is None, so the browser is never touched.
    """
    parts = []
    if browser_pid is not None:
        parts.append(f"kill {int(browser_pid)} 2>/dev/null || true")
    pid_file = f"{state_dir}/{_DAEMON_PID}"
    parts.append(
        f'[ -f "{pid_file}" ] && kill "$(cat "{pid_file}")" 2>/dev/null || true'
    )
    parts.append(f'rm -rf "{state_dir}" 2>/dev/null || true')
    return ["sh", "-c", "; ".join(parts)]


def parse_launch(result: ProgramResult) -> tuple[str, str]:
    """(endpoint, pid) from a `launch_argv` run, or raise with the browser's
    own diagnostics."""
    fields = {m.group(1): m.group(2).strip() for m in _ANNOUNCE.finditer(result.stdout)}
    endpoint, pid = fields.get("ENDPOINT"), fields.get("PID")
    if result.exit_code != 0 or not (endpoint and pid):
        raise BrowserLaunchError(
            (result.stderr or result.stdout or "browser launch failed").strip()[-2000:]
        )
    return endpoint, pid


async def launch(
    runtime: Runtime, state_dir: str, timeout: int = READY_TIMEOUT
) -> tuple[str, str]:
    """Start a Chromium in `runtime` and return (endpoint, pid)."""
    return parse_launch(await runtime.run(launch_argv(state_dir, timeout), {}))


async def is_alive(runtime: Runtime, pid: str) -> bool:
    return (await runtime.run(alive_argv(pid), {})).exit_code == 0
