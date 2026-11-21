"""network-policy: the agent runs an egress probe under the task's network policy.

One task per policy the task data can declare: `open` (the default), `off` (an empty
allowlist), `block` (blocks `example.com`) and `allow` (allows only `pypi.org`). Setup
stages a probe script while egress is still open; the prompt asks the model to run it.
The probe fetches both hosts and records which ones it reached, so the reward is 1 only
if the policy held: each host answered exactly when the policy permits it. A harness
without a shell writes no report and scores 0. Needs a runtime with network policy
support; the integration test runs it on prime.
"""

import json

import verifiers.v1 as vf
from verifiers.v1.runtimes import Runtime

__all__ = ["NetworkPolicyTaskset"]

BLOCKED = "example.com"
ALLOWED = "pypi.org"
PROBE_PATH = "/tmp/network-probe.py"
REPORT = "/tmp/network-probe.json"

# What each task's policy lets through, keyed by task name.
EXPECTED = {
    "open": {"blocked": "ok", "allowed": "ok"},
    "off": {"blocked": "denied", "allowed": "denied"},
    "block": {"blocked": "denied", "allowed": "ok"},
    "allow": {"blocked": "denied", "allowed": "ok"},
}

# Any HTTP answer, error status included, means the destination was reachable; a
# refused connection, a proxy refusal, a DNS failure, or a timeout means it was not.
PROBE = f"""
import json, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def probe(url):
    try:
        with urllib.request.urlopen(url, timeout=15):
            return "ok"
    except urllib.error.HTTPError:
        return "ok"
    except OSError:
        return "denied"

pending = Path({REPORT!r} + ".pending")
pending.touch()
urls = {{"blocked": "https://{BLOCKED}", "allowed": "https://{ALLOWED}/simple/"}}
with ThreadPoolExecutor(max_workers=len(urls)) as pool:
    report = dict(zip(urls, pool.map(probe, urls.values())))
pending.write_text(json.dumps(report))
pending.replace({REPORT!r})
print(json.dumps(report))
"""


class NetworkPolicyTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        await runtime.write(PROBE_PATH, PROBE.encode())

    @vf.reward(weight=1.0)
    async def policy_held(self, trace: vf.Trace, runtime: Runtime) -> float:
        # A shell can return a job handle while its two requests are still in flight.
        collect = 'while [ -e "$1.pending" ]; do sleep 0.2; done; cat "$1"'
        result = await runtime.run(
            ["timeout", "45", "sh", "-c", collect, "sh", REPORT], {}
        )
        report = json.loads(result.stdout) if result.exit_code == 0 else None
        trace.info["network_report"] = report
        return float(report == EXPECTED[self.data.name])


class NetworkPolicyTaskset(vf.Taskset[NetworkPolicyTask, vf.TasksetConfig]):
    def load(self) -> list[NetworkPolicyTask]:
        policies = {
            "open": {},
            "off": {"network": vf.NetworkPolicyConfig(allow=[])},
            "block": {"network": vf.NetworkPolicyConfig(block=[BLOCKED])},
            "allow": {"network": vf.NetworkPolicyConfig(allow=[ALLOWED])},
        }
        prompt = (
            "Verify the active network policy with the supplied read-only test. It makes "
            "ordinary HTTPS requests and records denied connections as test results. "
            "Leave the policy in place and run the test once in the foreground. "
            f"If you have a shell, run `python3 {PROBE_PATH}`. If your tool is a Python REPL, run "
            f"`import subprocess; print(subprocess.check_output(['python3', '{PROBE_PATH}'], text=True))`. "
            "If your tool returns a running job, wait for that same job to exit. "
            "Reply with the printed JSON once the command has finished."
        )
        return [
            NetworkPolicyTask(
                vf.TaskData(idx=idx, name=name, prompt=prompt, **policy),
                self.config.task,
            )
            for idx, (name, policy) in enumerate(policies.items())
        ]
