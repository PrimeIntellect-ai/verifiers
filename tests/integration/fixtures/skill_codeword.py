"""skill-codeword: the answer lives only in a skill the task installs.

The task ships a `project-codeword` skill (fixtures/skills) whose SKILL.md holds a
codeword the prompt never reveals, so reward 1.0 proves the harness installed the skill
where its program discovers skills and the model read it. Needs a harness with native
skill support (`SUPPORTS_SKILLS`).
"""

from pathlib import Path

import verifiers.v1 as vf

__all__ = ["SkillCodewordTaskset"]

CODEWORD = "amber-falcon-482"
SKILL = Path(__file__).resolve().parent / "skills" / "project-codeword"


class SkillCodewordTask(vf.Task):
    @vf.reward(weight=1.0)
    async def found(self, trace: vf.Trace) -> float:
        return float(CODEWORD in (trace.last_reply or ""))


class SkillCodewordConfig(vf.TasksetConfig):
    finish_tool: bool = False
    """Request the answer through finish for harnesses that require it."""


class SkillCodewordTaskset(vf.Taskset[SkillCodewordTask, SkillCodewordConfig]):
    def load(self) -> list[SkillCodewordTask]:
        # A finish-tool harness replies with its finish reason, which models fill
        # with a summary of their work unless told to put the codeword there.
        reply = (
            "call your finish tool with reason set to only the codeword and paths=[]."
            if self.config.finish_tool
            else "reply with only the codeword."
        )
        return [
            SkillCodewordTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "What is the project codeword? Look it up with your "
                        f"`project-codeword` skill, then {reply}"
                    ),
                    skills=[SKILL],
                ),
                self.config.task,
            )
        ]
