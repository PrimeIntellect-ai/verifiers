"""fetch: run a probe that requests URLs from inside the runtime and report each outcome.

The smallest task that shows what the execution network policy does: `setup` writes a
python probe into the box (before the policy is applied), the model runs it once with the
bash tool, and each URL's line in the tool output says whether the request got through
(`<url> -> 200`) or the policy stopped it (`<url> -> blocked <error>`). Python's urllib
rather than curl, so the runtimes' default `python:3.11-slim` images need nothing extra.
A fixture taskset for the v1 e2e suite (tests/v1/fixtures, resolved by id `fetch-v1`).
"""

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.runtimes import Runtime

PROBE = "/tmp/fetch.py"
SCRIPT = """\
import sys
import urllib.request

for url in sys.argv[1:]:
    try:
        print(url, "->", urllib.request.urlopen(url, timeout=20).status)
    except Exception as error:  # noqa: BLE001 - the policy's refusal arrives as any of several
        print(url, "->", "blocked", type(error).__name__)
"""


def outcomes(trace: vf.Trace) -> dict[str, str]:
    """Each probed URL's result token (`200`, `blocked`, ...) from the tool output."""
    found: dict[str, str] = {}
    for message in trace.tool_messages:
        content = message.content
        text = (
            content
            if isinstance(content, str)
            else "".join(getattr(part, "text", "") or "" for part in content)
        )
        for line in text.splitlines():
            url, sep, result = line.partition(" -> ")
            if sep and url.startswith("http"):
                found[url.strip()] = result.split()[0] if result.split() else ""
    return found


class FetchData(vf.TaskData):
    urls: list[str]


class FetchTask(vf.Task[FetchData]):
    async def setup(self, runtime: Runtime) -> None:
        await runtime.write(PROBE, SCRIPT.encode())

    @vf.reward(weight=1.0)
    async def probed(self, trace: vf.Trace) -> float:
        """The probe ran and reported every URL, whatever the policy did with them."""
        return float(set(self.data.urls) <= set(outcomes(trace)))


class FetchConfig(vf.TasksetConfig):
    urls: list[str] = Field(
        default_factory=lambda: ["https://example.com/", "https://pypi.org/"]
    )


class FetchTaskset(vf.Taskset[FetchTask, FetchConfig]):
    def load(self) -> list[FetchTask]:
        urls = self.config.urls
        return [
            FetchTask(
                FetchData(
                    idx=0,
                    prompt=(
                        "Use the bash tool to run exactly this command once, then finish: "
                        f"`python3 {PROBE} {' '.join(urls)}`"
                    ),
                    urls=urls,
                ),
                self.config.task,
            )
        ]


__all__ = ["FetchTaskset"]
