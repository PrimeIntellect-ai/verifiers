from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.envs.swarm import SwarmEnv, SwarmEnvConfig, SwarmTask, WorldConnection


class CoordinationConfig(SwarmEnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    coordinator: vf.AgentConfig = vf.AgentConfig()
    participants: dict[str, int] = Field(
        default_factory=lambda: {"solver": 4, "coordinator": 1}
    )
    max_concurrent_agents: int | None = 5


class CoordinationEnv(SwarmEnv, vf.Env[CoordinationConfig]):
    pass


class CoordinationTask(SwarmTask):
    review_repository = "team-work"

    async def prepare_world(self, world: WorldConnection) -> None:
        await world.mutate("create_repository", name="team-work")
        await world.mutate(
            "create_issue",
            repository="team-work",
            title="Four independent reviews and one submission",
            body="Exchange the four solver numbers. Each solver independently verifies their sum. "
            "The coordinator checks all four reviews, then gathers all five approvals of main's commit.",
        )

    def participant_prompt(self, index: int, accounts: list[str]) -> str:
        if len(accounts) != 5:
            raise ValueError(
                "This smoke task requires four solvers and one coordinator"
            )
        instructions = (
            "This is a submission-protocol smoke test. The initial main commit is the candidate; "
            "do not change the repository for this test. Join general and coordinate visibly there. "
            "Use issue 1 in repository team-work. Reuse an existing proposal rather than replacing it. "
            "After recording your verification, approve the current proposal. Do not finish before acceptance.\n"
            'join {"conversation":"general"} --mutate\n'
            'send {"conversation":"general","body":"text"} --mutate\n'
            'read {"conversation":"general"}\n'
            'issue {"repository":"team-work","number":1}\n'
            'comment_issue {"repository":"team-work","number":1,"body":"text"} --mutate\n'
            'repository {"repository":"team-work"}\n'
        )
        if index < 4:
            return instructions + (
                f"You are solver {index + 1}. Your private number is {(11, 13, 17, 19)[index]}. "
                "Post your number, read all four solver numbers, compute their sum independently, "
                "and comment exactly VERIFIED <sum> on the issue. Ask the coordinator to propose "
                "main if no proposal exists. Vote approve only after your verification."
            )
        return instructions + (
            "You are coordinator. Announce the plan, collect the four solver numbers and verify "
            "their sum. Wait for four different solver comments VERIFIED <sum>. "
            "Comment exactly CHECKED <sum> on the issue. Read main's commit and propose it for "
            "submission, publish the proposal ID to general, and ask every solver to approve. "
            "Approve it yourself. Only after all five votes approve, call request_acceptance. "
            "The controller performs independent public checks. If it fails, inspect evidence "
            "and resolve the issue before requesting acceptance again."
        )

    async def check_revision(self, world: WorldConnection, commit_oid: str) -> dict:
        issue = await world.call("issue", repository="team-work", number=1)
        reviewers = {
            c["author"]
            for c in issue["comments"]
            if c["author"].startswith("solver-") and c["body"] == "VERIFIED 60"
        }
        checked = any(
            c["author"].startswith("coordinator-") and c["body"] == "CHECKED 60"
            for c in issue["comments"]
        )
        return {
            "passed": len(reviewers) == 4 and checked,
            "commit_oid": commit_oid,
            "independent_solver_checks": len(reviewers),
            "coordinator_checked": checked,
        }

    async def capture(self, world: WorldConnection) -> dict:
        return {"issue": await world.call("issue", repository="team-work", number=1)}

    async def evaluate(self, submission: dict) -> float:
        return float(submission["review"]["status"] == "accepted")


class CoordinationTaskset(vf.Taskset[CoordinationTask, vf.TasksetConfig]):
    def load(self) -> list[CoordinationTask]:
        return [
            CoordinationTask(
                vf.TaskData(
                    idx=0,
                    name="Four solvers and coordinator: unanimous submission",
                    prompt="Coordinate an independently checked, unanimously approved submission.",
                    image="python:3.11-slim",
                    workdir="/app",
                ),
                self.config.task,
            )
        ]
