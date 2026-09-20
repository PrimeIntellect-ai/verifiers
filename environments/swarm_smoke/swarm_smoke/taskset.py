import verifiers.v1 as vf
from verifiers.v1.envs.swarm import SwarmTask, WorldConnection


class SwarmSmokeTask(SwarmTask):
    async def prepare_world(self, world: WorldConnection) -> None:
        await world.mutate("create_repository", name="team-work")
        await world.mutate(
            "create_issue",
            repository="team-work",
            title="Exchange and verify numbers",
            body="Exchange numbers in general, compute the product, and each leave an "
            "independent issue comment. Close this issue after both confirm.",
        )

    def participant_prompt(self, index: int, accounts: list[str]) -> str:
        if len(accounts) != 2:
            raise ValueError(
                "This connection smoke task requires exactly two participants"
            )
        number = (17, 23)[index]
        return (
            f"Your private number is {number}. Your colleague has the other number. "
            "Exchange numbers through the world general channel, compute their product, "
            "and each post exactly FINAL <product> in general. Each add an comment_issue "
            "to team-work issue 1 with exactly VERIFIED <product>. After both comments "
            "exist, close the issue. You may finish when both comments and the closed issue exist. "
            "Read the colleague's actual messages; do not invent their number. "
            "Available operations and JSON arguments:\n"
            'join {"conversation":"general"} --mutate\n'
            'send {"conversation":"general","body":"text"} --mutate\n'
            'read {"conversation":"general"}\n'
            'issue {"repository":"team-work","number":1}\n'
            'comment_issue {"repository":"team-work","number":1,"body":"text"} --mutate\n'
            'update_issue {"repository":"team-work","number":1,"status":"closed"} --mutate\n'
            "Use bash sleep 3 between checks when waiting. Never print environment credentials."
        )

    async def capture(self, world: WorldConnection) -> dict:
        return {
            "messages": await world.call("read", conversation="general"),
            "issue": await world.call("issue", repository="team-work", number=1),
        }

    async def evaluate(self, submission: dict) -> float:
        messages = submission["messages"]["messages"]
        issue = submission["issue"]
        confirmations = {m["author"] for m in messages if m["body"] == "FINAL 391"}
        reviews = {
            c["author"] for c in issue["comments"] if c["body"] == "VERIFIED 391"
        }
        return float(len(confirmations & reviews) == 2 and issue["status"] == "closed")


class SwarmSmokeTaskset(vf.Taskset[SwarmSmokeTask, vf.TasksetConfig]):
    def load(self) -> list[SwarmSmokeTask]:
        return [
            SwarmSmokeTask(
                vf.TaskData(
                    idx=0,
                    name="Shared communication and issue verification",
                    prompt="Collaborate through the shared world to verify a joint result.",
                    image="python:3.11-slim",
                    workdir="/app",
                ),
                self.config.task,
            )
        ]
