"""Split harness execution and the proc, file, and control-plane boundary."""

import asyncio
import os
import secrets
import shlex
import shutil
from contextlib import asynccontextmanager

import pytest
from conftest import _eval_config, scripted_chat_reply, serve_scripted_model

from verifiers.v1.cli.run import run_episodes
from verifiers.v1.dialects.chat import ChatDialect
from verifiers.v1.harnesses.bash.harness import BashHarness, BashHarnessConfig
from verifiers.v1.interception.tunnel.base import HEALTH_PATH
from verifiers.v1.runtimes.deployment import Deployment
from verifiers.v1.types import ToolMessage
from verifiers.v1.utils.prime import load_prime_config

# The boot ID identifies a VM, and the PID namespace identifies containers on one kernel.
_NAMESPACE = "cat /proc/sys/kernel/random/boot_id; readlink /proc/self/ns/pid"


@pytest.mark.parametrize("provider", ["docker", "prime"])
async def test_split_bash_boundary(tmp_path, monkeypatch, provider):
    if provider == "docker":
        if shutil.which("docker") is None:
            pytest.skip("Docker CLI or daemon unavailable")
        probe = await asyncio.create_subprocess_exec(
            "docker",
            "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await probe.wait():
            pytest.skip("Docker CLI or daemon unavailable")
    elif not (os.environ.get("PRIME_API_KEY") or load_prime_config().get("api_key")):
        pytest.skip("Prime sandbox credentials unavailable")

    env_sentinel = f"VF_SPLIT_ENV_SENTINEL_{secrets.token_hex(16)}"
    marker = f"/tmp/vf-harness-{secrets.token_hex(12)}"
    control_urls: list[str] = []
    backends = []
    harness_namespaces = []
    dialect = ChatDialect()

    original_start = Deployment.start

    async def capture_start(self):
        await original_start(self)
        backends.extend([self.harness_backend, self.execution_backend])

    monkeypatch.setattr(Deployment, "start", capture_start)

    original_setup = BashHarness.setup

    async def setup_with_marker(self, runtime):
        await original_setup(self, runtime)
        await runtime.write(marker, b"harness marker")
        namespace = await runtime.run(["sh", "-ec", _NAMESPACE], {})
        assert namespace.exit_code == 0
        harness_namespaces.append(namespace.stdout.strip())

    monkeypatch.setattr(BashHarness, "setup", setup_with_marker)

    import verifiers.v1.rollout as rollout_module

    original_serve = rollout_module.serve_interception

    @asynccontextmanager
    async def capture_interception(*args, **kwargs):
        async with original_serve(*args, **kwargs) as slot:
            control_urls.append(args[1].host_url(slot[0]).rstrip("/") + HEALTH_PATH)
            yield slot

    monkeypatch.setattr(rollout_module, "serve_interception", capture_interception)

    def answer(path, body):
        turn = sum(message.get("role") == "tool" for message in body["messages"])
        if turn == 0:
            name, arguments = "bash", {"command": "touch blocked.txt"}
        elif turn == 1:
            name, arguments = "bash", {"command": "printf 'before\\n' > work.txt"}
        elif turn == 2:
            name, arguments = (
                "edit",
                {"path": "work.txt", "old_str": "before", "new_str": "after"},
            )
        elif turn == 3:
            # Only request the evaluator's public health endpoint. The workspace
            # must be unable to reach it through the harness's callback route.
            probe = (
                "import sys, urllib.error, urllib.request\n"
                "try:\n"
                "    urllib.request.urlopen(sys.argv[1], timeout=3)\n"
                "except urllib.error.HTTPError as error:\n"
                "    if error.code != 403: raise\n"
                "except OSError:\n"
                "    pass\n"
                "else:\n"
                "    raise SystemExit('unexpected access to harness callback')"
            )
            name, arguments = (
                "bash",
                {
                    "command": (
                        f'set -e; test ! -e {shlex.quote(marker)}; test -z "${{VF_SPLIT_ENV_SENTINEL:-}}"; '
                        f"python3 -c {shlex.quote(probe)} {shlex.quote(control_urls[-1])}; "
                        f"echo WORKSPACE_NAMESPACE; {_NAMESPACE}"
                    )
                },
            )
        else:
            name = ""
            arguments = {}
        return dialect, scripted_chat_reply(
            body["model"],
            content="done",
            tool=name,
            arguments=arguments,
            call_id=f"split-{turn}",
        )

    async with serve_scripted_model(answer) as client:
        config = _eval_config(
            "split-sandbox",
            harness=BashHarnessConfig(
                id="bash", env={"VF_SPLIT_ENV_SENTINEL": env_sentinel}
            ),
            runtime={
                "type": provider,
                **({"image": "python:3.13-slim"} if provider == "docker" else {}),
            },
            client=client,
            max_turns=6,
        )
        config.output_dir = tmp_path
        (episode,) = await run_episodes(config)

    (trace,) = episode.traces
    assert trace.ok, [error.message for error in trace.errors]
    assert trace.reward == 1.0
    results = [
        message for message in trace.messages if isinstance(message, ToolMessage)
    ]
    assert [message.tool_call_id for message in results] == [
        f"split-{i}" for i in range(4)
    ]
    assert results[0].content == "POLICY_BLOCKED"
    assert "WORKSPACE_NAMESPACE" in results[-1].content
    workspace_namespace = results[-1].content.split("WORKSPACE_NAMESPACE", 1)[1].strip()
    assert len(workspace_namespace.splitlines()) == 2
    assert len(harness_namespaces[-1].splitlines()) == 2
    assert workspace_namespace != harness_namespaces[-1]
    assert env_sentinel not in str(results)
    harness_backend, workspace_backend = backends[-2:]
    assert trace.agent.harness_runtime.id != trace.agent.runtime.id
    if provider == "docker":
        from verifiers.v1.configs.deployment import DOCKER_HARNESS_IMAGE

        assert harness_backend.info.image == DOCKER_HARNESS_IMAGE
        assert workspace_backend.info.image == "python:3.13-slim"
    for backend in backends:
        await backend.verify_stopped()
