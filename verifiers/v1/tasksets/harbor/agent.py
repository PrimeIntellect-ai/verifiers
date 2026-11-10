"""Bind a RewardKit agent attempt to the environment's native judge."""

import json

from pydantic import Field

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.envs.agentic_judge.env import TRACE_FILE, VERDICT_FILE, JudgeTask
from verifiers.v1.errors import TaskError
from verifiers.v1.grading import scoring_network
from verifiers.v1.judges.agent import AgentJudgeRequest
from verifiers.v1.task import TaskData, TaskTimeout
from verifiers.v1.tasksets.harbor.toolset import HarborMCPConfig, HarborMCPToolset
from verifiers.v1.types import SystemMessage, UserMessage

from .judge import HarborJudgeConfig


class HarborAgentJudgeConfig(TaskConfig):
    mcp_servers: list[dict] = Field(default_factory=list)


class HarborAgentJudgeTask(JudgeTask[HarborAgentJudgeConfig]):
    @classmethod
    def toolsets(cls, config: HarborAgentJudgeConfig):
        return [
            HarborMCPToolset(HarborMCPConfig(colocated=True, server=server))
            for server in config.mcp_servers
        ]


async def prepare_agent_judge(task, runtime, trace, reward, authored, workspace):
    from rewardkit.agents import AgentAttempt

    if task.judge_executor is None:
        raise TaskError("Agent judge TOMLs require the agentic-judge environment")
    if (
        task.config.judge is not None
        and task.config.judge.model_copy(
            update={"view": "last_reply", "include_hidden_reasoning": False}
        )
        != HarborJudgeConfig()
    ):
        raise TaskError("Configure agent judge overrides on env.judge")
    exclude = {"provider_state"}
    if not (task.config.judge and task.config.judge.include_hidden_reasoning):
        exclude.add("reasoning_content")
    evidence = trace.to_json(
        exclude={"ok": True, "nodes": {"__all__": {"message": exclude}}}
    )
    network = None
    if task.scoring.mode == "shared":
        network = (
            scoring_network(trace.agent.config.runtime, task.scoring.network)
            if task.scoring.network is not None
            else NetworkPolicyConfig()
        )

    async def call(prompt, schema):
        # RewardKit constructs criterion prompts and validates the returned JSON.
        # VF owns only execution and the trusted evidence/verdict file locations.
        judge_task = HarborAgentJudgeTask(
            TaskData(
                idx=trace.task.data.idx,
                prompt=[
                    SystemMessage(content=prompt),
                    UserMessage(
                        content=(
                            f"Task:\n{task.data.prompt_text}\n\n"
                            f"The native solver trace is at {TRACE_FILE}. "
                            f"Write the JSON answer to {VERDICT_FILE}. Required schema:\n{json.dumps(schema)}"
                        )
                    ),
                ],
                timeout=TaskTimeout(agent=reward.judge.timeout),
            ),
            files={TRACE_FILE: evidence},
            parse_verdict=str,
            workspace=workspace,
            env=dict(runtime.env),
            config=HarborAgentJudgeConfig(
                mcp_servers=[
                    server.model_dump(mode="json")
                    for server in reward.judge.mcp_servers
                ]
            ),
        )
        verdict = await task.judge_executor(
            AgentJudgeRequest(
                judge_task,
                model=reward.judge.model,
                sampling=SamplingConfig(reasoning_effort=reward.judge.reasoning_effort)
                if reward.judge.reasoning_effort is not None
                else None,
                isolated=True,
                network=network,
            ),
            runtime,
        )
        if verdict.is_timeout:
            return AgentAttempt(timed_out=True)
        if not verdict.ok:
            return AgentAttempt(
                error=RuntimeError("; ".join(e.message for e in verdict.errors))
            )
        return AgentAttempt(output=verdict.info["verdict_raw"])

    reward.call = call
