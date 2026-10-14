"""network-policy-v1: a scripted, model-free agent probes egress under the task's policy.

One task per policy the task data can declare: `open` (the default), `off` (an empty
allowlist), `block` (blocks `example.com`) and `allow` (allows only `pypi.org`). During
execution the scripted harness fetches both hosts and the framework's own interception
endpoint from inside the runtime and records which ones it reached. It also records the
system prompt it was handed, so a test can check the network notice. The reward is 1
only if the policy held: each host answered exactly when the policy permits it, and the
framework route stayed reachable. Needs a runtime with network policy support.
"""

import json

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime

__all__ = ["NetworkPolicyTaskset", "ScriptedNetworkProbeHarness"]

BLOCKED = "example.com"
ALLOWED = "pypi.org"
REPORT = "/tmp/network-policy-report.json"

# What each task's policy lets through, keyed by task name.
EXPECTED = {
    "open": {"blocked": "ok", "allowed": "ok", "framework": "ok"},
    "off": {"blocked": "denied", "allowed": "denied", "framework": "ok"},
    "block": {"blocked": "denied", "allowed": "ok", "framework": "ok"},
    "allow": {"blocked": "denied", "allowed": "ok", "framework": "ok"},
}

# Any HTTP answer, error status included, means the destination was reachable; a
# refused connection, a proxy refusal, a DNS failure, or a timeout means it was not.
PROBE = """
import json, sys, urllib.error, urllib.request
report = {}
for name, url in json.loads(sys.argv[1]).items():
    try:
        urllib.request.urlopen(url, timeout=15)
        report[name] = "ok"
    except urllib.error.HTTPError:
        report[name] = "ok"
    except Exception:
        report[name] = "denied"
open(sys.argv[2], "w").write(json.dumps(report))
"""


class ScriptedNetworkProbeHarness(Harness[HarnessConfig]):
    async def launch(
        self,
        ctx: ModelContext,
        trace: vf.Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        trace.info["system_prompt"] = data.system_prompt
        urls = {
            "blocked": f"https://{BLOCKED}",
            "allowed": f"https://{ALLOWED}/simple/",
            "framework": endpoint,
        }
        return await runtime.run_program(
            ["python3", "-c", PROBE, json.dumps(urls), REPORT], {}
        )


class NetworkPolicyTask(vf.Task):
    @vf.reward(weight=1.0)
    async def policy_held(self, trace: vf.Trace, runtime: Runtime) -> float:
        report = json.loads(await runtime.read(REPORT))
        trace.info["network_report"] = report
        return float(report == EXPECTED[self.data.name])


class NetworkPolicyTaskset(vf.Taskset[NetworkPolicyTask, vf.TasksetConfig]):
    def load(self) -> list[NetworkPolicyTask]:
        policies = {
            "open": {},
            "off": {"network_allow": []},
            "block": {"network_block": [BLOCKED]},
            "allow": {"network_allow": [ALLOWED]},
        }
        return [
            NetworkPolicyTask(
                vf.TaskData(idx=idx, name=name, prompt="probe", **policy),
                self.config.task,
            )
            for idx, (name, policy) in enumerate(policies.items())
        ]
