"""A tool server on a Prime runtime of its own runs on a VM of its own, reachable from
the restricted workspace over the rollout's private network, under the egress policy of
its own runtime: unrestricted, or none at all."""

import json
import os

import pytest
from fetch_tool import FetchToolTask, FetchToolTaskConfig

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY"
)

FETCH = """
import sys, urllib.error, urllib.request
try:
    print(urllib.request.urlopen(sys.argv[1], timeout=15).status)
except urllib.error.HTTPError as error:
    print(error.code)
except Exception as error:
    print(type(error).__name__)
"""

# The workspace reaches the tool server (any HTTP status) but not the internet.
PROBE = """
import json, sys, urllib.error, urllib.request

def fetch(url):
    try:
        return urllib.request.urlopen(url, timeout=15).status
    except urllib.error.HTTPError as error:
        return error.code
    except Exception as error:
        return type(error).__name__

json.dump(
    {"tool": fetch(sys.argv[1]), "public": fetch("https://example.com")},
    open("/tmp/probe.json", "w"),
)
"""


class ProbeHarness(vf.Harness):
    SUPPORTS_MCP = True

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        trace.info["url"] = mcp_urls["fetch"]
        return await runtime.run(["python", "-c", PROBE, mcp_urls["fetch"]], {})


class ProbeTask(FetchToolTask):
    async def finalize(self, trace, runtime, deployment):
        trace.info["workspace"] = json.loads(await runtime.read("/tmp/probe.json"))
        tool = await deployment.target("fetch").run(
            ["python", "-c", FETCH, "https://example.com"], {}
        )
        trace.info["tool"] = tool.stdout.strip()


@pytest.mark.parametrize(("allow", "egress"), [(["*"], "200"), ([], "URLError")])
async def test_a_tool_vm_keeps_its_own_egress_beside_a_restricted_workspace(
    allow, egress
):
    tools = vf.ToolsetConfig(
        runtime=vf.PrimeConfig(labels=["vf-integration"], allow=allow)
    )
    task = ProbeTask(
        vf.TaskData(
            prompt="Probe the network", network=vf.NetworkPolicyConfig(allow=[])
        ),
        FetchToolTaskConfig(tools=tools),
    )
    config = vf.AgentConfig(
        model="unused",
        runtime=vf.PrimeConfig(labels=["vf-integration"]),
        timeout=TimeoutConfig(setup=900, finalize=120, scoring=60),
    )
    async with vf.Agent(config) as agent:
        agent.harness = ProbeHarness(vf.HarnessConfig())
        trace = await agent.run(task)
    assert trace.ok, trace.errors
    assert ".sandbox.internal:" in trace.info["url"]
    workspace = trace.info["workspace"]
    assert isinstance(workspace["tool"], int), workspace
    assert workspace["public"] == "URLError"
    assert trace.info["tool"] == egress
