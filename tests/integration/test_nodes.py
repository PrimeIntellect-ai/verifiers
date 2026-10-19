"""Nodes a task declares (`TaskData.nodes`), on Prime: one with a policy of its own
gets a machine of its own on the rollout's private network, runs its process there, and
can host a tool server (`ToolsetConfig.on`); one under the workspace's policy shares the
workspace's machine as a Compose service."""

import json
import os
from urllib.parse import urlsplit

import pytest
from fetch_tool import FetchToolTask, FetchToolTaskConfig

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("PRIME_API_KEY"), reason="needs PRIME_API_KEY"
)

LABELS = ["vf-integration"]
# Containers on a Prime VM pull from Docker Hub through the VMs' one shared egress
# address, which its anonymous rate limit soon catches; this mirror has none.
MIRRORED = "mirror.gcr.io/library/python:3.11-slim"
SERVE = ["python", "-m", "http.server", "8000", "--directory", "/"]
READY = [
    "python",
    "-c",
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')",
]
FETCH = """
import json, sys, urllib.error, urllib.request

def fetch(url):
    try:
        return urllib.request.urlopen(url, timeout=15).status
    except urllib.error.HTTPError as error:
        return error.code
    except Exception as error:
        return type(error).__name__

json.dump({url: fetch(url) for url in sys.argv[1:]}, sys.stdout)
"""


async def fetch(runtime: vf.Runtime, *urls: str) -> dict:
    result = await runtime.run(["python", "-c", FETCH, *urls], {})
    assert result.exit_code == 0, result
    return json.loads(result.stdout)


class ProbeHarness(vf.Harness):
    """Fetches `trace.info["urls"]` from the workspace, plus the tool server's URL."""

    SUPPORTS_MCP = True

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        urls = [*trace.info["urls"], *mcp_urls.values()]
        trace.info["mcp"] = dict(mcp_urls)
        result = await runtime.run(["python", "-c", FETCH, *urls], {})
        trace.info["workspace"] = json.loads(result.stdout)
        return result


class NodeTask(FetchToolTask):
    async def setup(self, trace, runtime, deployment):
        trace.info["urls"] = [
            *(
                f"http://{deployment.host(name)}:8000/"
                for name in self.data.nodes or {}
            ),
            "https://example.com",
        ]

    async def finalize(self, trace, runtime, deployment):
        trace.info["nodes"] = {
            name: await fetch(deployment.target(name), "https://example.com")
            for name in self.data.nodes or {}
        }


async def run(task: vf.Task, runtime: vf.PrimeConfig) -> vf.Trace:
    config = vf.AgentConfig(
        model="unused",
        runtime=runtime,
        timeout=TimeoutConfig(setup=1200, finalize=120, scoring=60),
    )
    async with vf.Agent(config) as agent:
        agent.harness = ProbeHarness(vf.HarnessConfig())
        trace = await agent.run(task)
    assert trace.ok, trace.errors
    return trace


async def test_a_node_with_a_policy_of_its_own_runs_on_a_machine_of_its_own():
    web = vf.NodeConfig(
        runtime={"allow": ["*"]},
        process=vf.ProcessConfig(command=SERVE, readiness=READY),
    )
    task = NodeTask(
        vf.TaskData(
            prompt="Probe the network",
            network=vf.NetworkPolicyConfig(allow=[]),
            nodes={"web": web},
        ),
        FetchToolTaskConfig(tools=vf.ToolsetConfig(on="web")),
    )
    trace = await run(task, vf.PrimeConfig(labels=LABELS))
    web_url, public, tool = trace.info["urls"] + [trace.info["mcp"]["fetch"]]
    # The node's machine, at its private network name, serving its process and the
    # tool server placed on it.
    assert urlsplit(web_url).hostname.endswith(".sandbox.internal")
    assert urlsplit(tool).hostname == urlsplit(web_url).hostname
    workspace = trace.info["workspace"]
    assert workspace[web_url] == 200, workspace
    assert isinstance(workspace[tool], int), workspace
    # The workspace stays offline; the node keeps its own egress.
    assert workspace[public] == "URLError", workspace
    assert trace.info["nodes"]["web"]["https://example.com"] == 200


async def test_nodes_under_the_workspace_s_policy_share_its_machine():
    # `$PORT` is the process's own variable, kept literal through Compose.
    serve = ["sh", "-c", 'exec python -m http.server "$PORT" --directory /']
    db = vf.NodeConfig(
        process=vf.ProcessConfig(
            command=serve, environment={"PORT": "8000"}, readiness=READY
        )
    )
    task = NodeTask(
        vf.TaskData(prompt="Probe the network", nodes={"db": db}),
        FetchToolTaskConfig(tools=vf.ToolsetConfig(colocated=True)),
    )
    trace = await run(task, vf.PrimeConfig(labels=LABELS, image=MIRRORED))
    # A Compose service on the workspace's machine, at its own name.
    db_url = trace.info["urls"][0]
    assert db_url == "http://db:8000/"
    assert trace.info["workspace"][db_url] == 200, trace.info["workspace"]
