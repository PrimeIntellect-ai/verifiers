"""agent-user-v1: a scripted, model-free agent that runs as a non-root user and tries root actions.

Task setup runs as root: it creates the task's `agent_user` (images normally ship one),
plants a root-only secret, hands the workdir to that user, and prepares a uv script. The
scripted harness then runs as that user and records whether:
- four actions only root may take are denied and two the user may take still work;
- the same uv script, prepared through the harness's runtime, runs as the user from its own
  environment rather than root's;
- a file written through the runtime is the user's, and a root-only file can't be read;
- its colocated tool server runs as the user too.
The reward is 1 only if every check holds. Needs a runtime that can switch users (docker or
prime).
"""

import json
import os
import pwd
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime

__all__ = ["AgentUserTaskset", "ScriptedAgentUserHarness"]

AGENT_USER = "vf-agent"
SECRET = "/opt/vf-agent-user-secret"
REPORT = "agent-user-report.json"
WRITTEN = "agent-user-written.txt"
TOOL_USER = "/tmp/vf-agent-user-tool-user"
ROOT_ACTIONS = ("read_secret", "write_etc", "chmod_secret", "signal_init")

SETUP = (
    f"useradd -m {AGENT_USER} && echo secret > {SECRET} && chmod 600 {SECRET} "
    f"&& chown -R {AGENT_USER} ."
)

UV_SCRIPT = """# /// script
# dependencies = []
# ///
import os, pwd, sys
print(pwd.getpwuid(os.getuid()).pw_name, sys.executable)
"""

PROBE = f"""
r() {{ "$@" >/dev/null 2>&1 && echo ok || echo denied; }}
printf '{{"user":"%s","secret_exists":"%s","read_secret":"%s","write_etc":"%s","chmod_secret":"%s","signal_init":"%s","write_workdir":"%s","written_owner":"%s","tool_user":"%s"}}' \\
  "$(id -un)" "$(r test -e {SECRET})" "$(r cat {SECRET})" "$(r touch /etc/vf-agent-user-probe)" \\
  "$(r chmod 644 {SECRET})" "$(r kill -0 1)" "$(r touch probe)" "$(stat -c %U {WRITTEN})" \\
  "$(cat {TOOL_USER})" > {REPORT}
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


class AgentUserTask(vf.Task):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [AgentUserToolset(vf.ToolsetConfig(colocated=True))]

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
        users = ("user", "uv_user", "written_owner", "tool_user")
        return float(
            all(report[key] == AGENT_USER for key in users)
            and report["secret_exists"] == report["write_workdir"] == "ok"
            and all(report[action] == "denied" for action in ROOT_ACTIONS)
            and report["read_secret_api"] == "denied"
            and not report["uv_python"].startswith("/root/")
            and report["uv_script"] != report["root_uv_script"]
        )


class AgentUserTaskset(vf.Taskset[AgentUserTask, vf.TasksetConfig]):
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


if __name__ == "__main__":
    Path(TOOL_USER).write_text(current_user())
    AgentUserToolset.run()
