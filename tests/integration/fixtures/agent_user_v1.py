"""agent-user-v1: a scripted, model-free agent that runs as a non-root user and tries root actions.

Root setup creates the task's `agent_user`, plants a root-only secret, and prepares a uv
script; the workdir stays root-owned, like a typical image WORKDIR. The scripted harness runs
as that user and records whether root-only actions are denied, whether its own uv script, its
runtime file I/O, and its colocated tool server all act as the user, and whether that server
can reach uv. The reward is 1 only if every check holds. Needs docker or prime.
"""

import json
import os
import pwd
import shutil
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.harness import Harness
from verifiers.v1.mcp.launch import serve
from verifiers.v1.runtimes import ProgramResult, Runtime

__all__ = ["AgentUserTaskset", "ScriptedAgentUserHarness"]

AGENT_USER = "vf-agent"
SECRET = "/opt/vf-agent-user-secret"
REPORT = "/tmp/agent-user-report.json"
WRITTEN = "/home/vf-agent/agent-user-written.txt"
TOOL_USER = "/tmp/vf-agent-user-tool-user"
TOOL_UV = "/tmp/vf-agent-user-tool-uv"
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
printf '{{"user":"%s","secret_exists":"%s","read_secret":"%s","write_etc":"%s","chmod_secret":"%s","signal_init":"%s","write_home":"%s","written_owner":"%s","tool_user":"%s","tool_uv":"%s"}}' \\
  "$(id -un)" "$(r test -e {SECRET})" "$(r cat {SECRET})" "$(r touch /etc/vf-agent-user-probe)" \\
  "$(r chmod 644 {SECRET})" "$(r kill -0 1)" "$(r touch "$HOME/probe")" "$(stat -c %U {WRITTEN})" \\
  "$(cat {TOOL_USER})" "$(cat {TOOL_UV})" > {REPORT}
"""


def current_user() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


class AgentUserToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "agentuser"

    @vf.tool
    def whoami(self) -> str:
        """Return the user this tool server runs as."""
        return current_user()


class ScriptedAgentUserHarness(Harness[HarnessConfig]):
    SUPPORTS_MCP = True

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
        return await runtime.run_program(["sh", "-c", PROBE], {})


class AgentUserData(vf.TaskData):
    preinstalled_tools: bool = False


class AgentUserTask(vf.Task[AgentUserData]):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [AgentUserToolset(vf.ToolsetConfig(colocated=True))]

    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        result = await runtime.run(["sh", "-c", SETUP], {})
        if result.exit_code != 0:
            raise RuntimeError(f"root setup failed: {result.stderr}")
        trace.info["root_uv_script"] = (await runtime.prepare_uv_script(UV_SCRIPT))[-1]
        if self.data.preinstalled_tools:
            # As an earlier rollout on a reused runtime would: the agent's server then
            # finds its package already installed by the default user.
            async with serve(
                AgentUserToolset(vf.ToolsetConfig(colocated=True)), runtime
            ):
                pass
            await runtime.run(["rm", "-f", TOOL_USER, TOOL_UV], {})

    @vf.reward(weight=1.0)
    async def agent_user_contained(self, trace: vf.Trace, runtime: Runtime) -> float:
        report = {
            **json.loads(await runtime.read(REPORT)),
            **trace.info["agent_user_harness"],
            "root_uv_script": trace.info["root_uv_script"],
        }
        trace.info["agent_user_report"] = report
        users = ("user", "uv_user", "written_owner", "tool_user")
        return float(
            all(report[key] == AGENT_USER for key in users)
            and report["secret_exists"] == report["write_home"] == "ok"
            and all(report[action] == "denied" for action in ROOT_ACTIONS)
            and report["read_secret_api"] == "denied"
            and report["tool_uv"] == "ok"
            and not report["uv_python"].startswith("/root/")
            and report["uv_script"] != report["root_uv_script"]
        )


class AgentUserConfig(vf.TasksetConfig):
    preinstalled_tools: bool = False
    """Bring the colocated tool up once as the default user before the agent's run."""


class AgentUserTaskset(vf.Taskset[AgentUserTask, AgentUserConfig]):
    def load(self) -> list[AgentUserTask]:
        return [
            AgentUserTask(
                AgentUserData(
                    idx=0,
                    prompt="Scripted: try root-only actions as a non-root user.",
                    agent_user=AGENT_USER,
                    preinstalled_tools=self.config.preinstalled_tools,
                )
            )
        ]


if __name__ == "__main__":
    Path(TOOL_USER).write_text(current_user())
    Path(TOOL_UV).write_text("ok" if shutil.which("uv") else "missing")
    AgentUserToolset.run()
