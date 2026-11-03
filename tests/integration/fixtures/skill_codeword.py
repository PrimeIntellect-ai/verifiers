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


class SkillCodewordTaskset(vf.Taskset[SkillCodewordTask, vf.TasksetConfig]):
    def load(self) -> list[SkillCodewordTask]:
        return [
            SkillCodewordTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "What is the project codeword? Look it up with your "
                        "`project-codeword` skill, then reply with only the codeword."
                    ),
                    skills=[SKILL],
                ),
                self.config.task,
            )
        ]
