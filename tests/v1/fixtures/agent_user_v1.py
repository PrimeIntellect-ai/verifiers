"""agent-user-v1: a scripted, model-free agent that runs as a non-root user and tries root actions.

Task setup runs as root: it creates the task's `agent_user` (images normally ship one),
plants a root-only secret, and hands the workdir to that user. The scripted harness then runs
as that user, attempts four actions only root may take, and records each outcome in the
workdir. The reward is 1 only if every root action was denied while the agent still saw the
secret and could write its workspace.
Needs a runtime that can switch users (docker or prime).
"""

import json

import verifiers.v1 as vf
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime

__all__ = ["AgentUserTaskset", "ScriptedAgentUserHarness"]

AGENT_USER = "vf-agent"
SECRET = "/opt/vf-agent-user-secret"
REPORT = "agent-user-report.json"
ROOT_ACTIONS = ("read_secret", "write_etc", "chmod_secret", "signal_init")

SETUP = (
    f"useradd -m {AGENT_USER} && echo secret > {SECRET} && chmod 600 {SECRET} "
    f"&& chown -R {AGENT_USER} ."
)

PROBE = f"""
r() {{ "$@" >/dev/null 2>&1 && echo ok || echo denied; }}
printf '{{"user":"%s","secret_exists":"%s","read_secret":"%s","write_etc":"%s","chmod_secret":"%s","signal_init":"%s","write_workdir":"%s"}}' \\
  "$(id -un)" "$(r test -e {SECRET})" "$(r cat {SECRET})" "$(r touch /etc/vf-agent-user-probe)" \\
  "$(r chmod 644 {SECRET})" "$(r kill -0 1)" "$(r touch probe)" > {REPORT}
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
        return await runtime.run_program(["sh", "-c", PROBE], {})


class AgentUserTask(vf.Task):
    async def setup(self, trace: vf.Trace, runtime: Runtime) -> None:
        result = await runtime.run(["sh", "-c", SETUP], {})
        if result.exit_code != 0:
            raise RuntimeError(f"root setup failed: {result.stderr}")

    @vf.reward(weight=1.0)
    async def root_actions_denied(self, trace: vf.Trace, runtime: Runtime) -> float:
        report = json.loads(await runtime.read(REPORT))
        trace.info["agent_user_report"] = report
        return float(
            report["user"] == AGENT_USER
            and report["secret_exists"] == "ok"
            and report["write_workdir"] == "ok"
            and all(report[action] == "denied" for action in ROOT_ACTIONS)
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
