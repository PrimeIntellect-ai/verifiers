"""A service VM runs on a Prime VM of its own, reachable from the restricted workspace
over the rollout's LAN, with unrestricted egress of its own."""

import json
import os

import pytest

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY"
)

# The working directory does not exist in the image, and the probe passes only with the
# service's environment and working directory.
READY = (
    'test "$(pwd)" = /srv/sidecar && python -c "import os, urllib.request; '
    "urllib.request.urlopen('http://localhost:' + os.environ['PORT'])\""
)
SIDECAR = vf.ServiceVM(
    image="python:3.11-slim",
    command=["python", "-m", "http.server", "8000"],
    environment={"PORT": "8000"},
    workdir="/srv/sidecar",
    readiness=vf.Probe(["sh", "-c", READY], interval=2, attempts=60),
)

FETCH = """
import sys, urllib.request
try:
    print(urllib.request.urlopen(sys.argv[1], timeout=15).status)
except Exception as error:
    print(type(error).__name__)
"""

PROBE = """
import json, urllib.request

def fetch(url):
    try:
        return urllib.request.urlopen(url, timeout=15).status
    except Exception as error:
        return type(error).__name__

host = open("/tmp/sidecar-host").read()
json.dump(
    {"sidecar": fetch(f"http://{host}:8000"), "public": fetch("https://example.com")},
    open("/tmp/probe.json", "w"),
)
"""


class ProbeHarness(vf.Harness):
    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        return await runtime.run_program(["python", "-c", PROBE], {})


class ServiceVMTask(vf.Task[vf.TaskData, vf.State, vf.TaskConfig]):
    def deployment_spec(self) -> vf.ServiceVMSpec:
        return vf.ServiceVMSpec({"sidecar": SIDECAR})

    async def setup(self, trace, runtime, deployment):
        await runtime.write("/tmp/sidecar-host", deployment.hosts["sidecar"].encode())

    async def finalize(self, trace, runtime, deployment):
        trace.info["workspace"] = json.loads(await runtime.read("/tmp/probe.json"))
        sidecar = await deployment.target("sidecar").run(
            ["python", "-c", FETCH, "https://example.com"], {}
        )
        trace.info["sidecar"] = sidecar.stdout.strip()


async def test_service_vm_keeps_its_egress_beside_a_restricted_workspace():
    task = ServiceVMTask(vf.TaskData(prompt="Probe the network", network_allow=[]))
    config = vf.AgentConfig(
        model="unused",
        runtime=vf.PrimeConfig(labels=["vf-integration"]),
        timeout=TimeoutConfig(setup=900, finalize=120, scoring=60),
    )
    async with vf.Agent(config) as agent:
        agent.harness = ProbeHarness(vf.HarnessConfig())
        trace = await agent.run(task)
    assert trace.ok, trace.errors
    assert trace.info["workspace"] == {"sidecar": 200, "public": "URLError"}
    assert trace.info["sidecar"] == "200"
