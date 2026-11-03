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
report = {{}}
for name, url in {{"blocked": "https://{BLOCKED}", "allowed": "https://{ALLOWED}/simple/"}}.items():
    try:
        urllib.request.urlopen(url, timeout=15)
        report[name] = "ok"
    except urllib.error.HTTPError:
        report[name] = "ok"
    except Exception:
        report[name] = "denied"
open({REPORT!r}, "w").write(json.dumps(report))
print(json.dumps(report))
"""


class NetworkPolicyTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        await runtime.write(PROBE_PATH, PROBE.encode())

    @vf.reward(weight=1.0)
    async def policy_held(self, trace: vf.Trace, runtime: Runtime) -> float:
        result = await runtime.run(["cat", REPORT], {})
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
            f"Your only task: run the command `python3 {PROBE_PATH}` once in your "
            "shell, then reply with exactly what it printed. Any network policy is "
            "already enforced, so do not inspect, configure, or change the network or "
            "firewall, and do not run any other command."
        )
        return [
            NetworkPolicyTask(
                vf.TaskData(idx=idx, name=name, prompt=prompt, **policy),
                self.config.task,
            )
            for idx, (name, policy) in enumerate(policies.items())
        ]
