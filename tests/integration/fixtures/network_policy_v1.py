"""network-policy-v1: a scripted, model-free agent probes egress under the task's policy.

Task 0 blocks `example.com`; task 1 allows only `pypi.org`. During execution the scripted
harness fetches both hosts and the framework's own interception endpoint from inside the
runtime and records which ones it reached. The reward is 1 only if the policy held: the
denied host failed, the permitted host answered, and the framework route stayed reachable.
Needs a runtime with network policy support (docker or prime).
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
        return float(
            report == {"blocked": "denied", "allowed": "ok", "framework": "ok"}
        )


class NetworkPolicyTaskset(vf.Taskset[NetworkPolicyTask, vf.TasksetConfig]):
    def load(self) -> list[NetworkPolicyTask]:
        return [
            NetworkPolicyTask(
                vf.TaskData(
                    idx=0, name="block", prompt="probe", network_block=[BLOCKED]
                ),
                self.config.task,
            ),
            NetworkPolicyTask(
                vf.TaskData(
                    idx=1, name="allow", prompt="probe", network_allow=[ALLOWED]
                ),
                self.config.task,
            ),
        ]
