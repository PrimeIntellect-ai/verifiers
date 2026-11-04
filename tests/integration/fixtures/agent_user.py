"""agent-user: a scripted, model-free agent that runs as a non-root user and tries root actions.

Root setup creates the task's `agent_user`, plants a root-only secret, and prepares a uv
script; the workdir stays root-owned, like a typical image WORKDIR. The scripted harness runs
as that user and records whether root-only actions are denied, and whether its own uv script
and its runtime file I/O act as the user. The reward is 1 only if every check holds. Needs
docker or prime.
"""

import json

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime

__all__ = ["AgentUserTaskset", "ScriptedAgentUserHarness"]

AGENT_USER = "vf-agent"
SECRET = "/opt/vf-agent-user-secret"
REPORT = "/tmp/agent-user-report.json"
WRITTEN = "/home/vf-agent/agent-user-written.txt"
ROOT_ACTIONS = ("read_secret", "write_etc", "chmod_secret", "signal_init")

SETUP = f"useradd -m {AGENT_USER} && echo secret > {SECRET} && chmod 600 {SECRET}"

UV_SCRIPT = """# /// script
# dependencies = []
# ///
import os, pwd, sys
print(pwd.getpwuid(os.getuid()).pw_name, sys.executable)
"""

PROBE = f"""
r() {{ "$@" >/dev/null 2>&1 && echo ok || echo denied; }}
printf '{{"user":"%s","secret_exists":"%s","read_secret":"%s","write_etc":"%s","chmod_secret":"%s","signal_init":"%s","write_home":"%s","written_owner":"%s"}}' \\
  "$(id -un)" "$(r test -e {SECRET})" "$(r cat {SECRET})" "$(r touch /etc/vf-agent-user-probe)" \\
  "$(r chmod 644 {SECRET})" "$(r kill -0 1)" "$(r touch "$HOME/probe")" "$(stat -c %U {WRITTEN})" \\
  > {REPORT}
"""


class ScriptedAgentUserHarness(Harness[HarnessConfig]):
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
        script = await runtime.prepare_uv_script(UV_SCRIPT)
        uv_user, uv_python = (await runtime.run(script, {})).stdout.split()
        await runtime.write(WRITTEN, b"written by the agent user")
        try:
            await runtime.read(SECRET)
            read_api = "ok"
        except SandboxError:
            read_api = "denied"
        trace.info["agent_user_harness"] = {
            "uv_user": uv_user,
            "uv_python": uv_python,
            "uv_script": script[-1],
            "read_secret_api": read_api,
        }
        return await runtime.run(["sh", "-c", PROBE], {})


class AgentUserTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        result = await runtime.run(["sh", "-c", SETUP], {})
        if result.exit_code != 0:
            raise RuntimeError(f"root setup failed: {result.stderr}")
        trace.info["root_uv_script"] = (await runtime.prepare_uv_script(UV_SCRIPT))[-1]

    @vf.reward(weight=1.0)
    async def agent_user_contained(self, trace: vf.Trace, runtime: Runtime) -> float:
        report = {
            **json.loads(await runtime.read(REPORT)),
            **trace.info["agent_user_harness"],
            "root_uv_script": trace.info["root_uv_script"],
        }
        trace.info["agent_user_report"] = report
        users = ("user", "uv_user", "written_owner")
        return float(
            all(report[key] == AGENT_USER for key in users)
            and report["secret_exists"] == report["write_home"] == "ok"
            and all(report[action] == "denied" for action in ROOT_ACTIONS)
            and report["read_secret_api"] == "denied"
            and not report["uv_python"].startswith("/root/")
            and report["uv_script"] != report["root_uv_script"]
        )


class AgentUserTaskset(vf.Taskset[AgentUserTask]):
    def load(self) -> list[AgentUserTask]:
        return [
            AgentUserTask(
                vf.TaskData(
                    idx=0,
                    prompt="Scripted: try root-only actions as a non-root user.",
                    agent_user=AGENT_USER,
                )
            )
        ]
