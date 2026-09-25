"""An audit requests repair; the fixer steers the waiting task to review. Runs offline."""

from verifiers.v1.flow import Flow, Job, JobData, Transition, stage


class Repair(JobData):
    task: str


class CrossJob(Flow):
    async def setup(self) -> None:
        self.create("task", stage="audit", data=JobData())

    @stage
    async def audit(self, job: Job[JobData]) -> Transition[JobData]:
        self.create("repair", stage="fix", data=Repair(task=job.id))
        return Transition(outcome="needs_repair", status="waiting")

    @stage
    async def fix(self, job: Job[Repair]) -> Transition[Repair]:
        # After performing the repair, choose where the waiting task resumes.
        self.apply(
            job.data.task,
            Transition(
                stage="review", status="ready", note="Review the repaired task."
            ),
        )
        return Transition(outcome="released", status="terminal")

    @stage
    async def review(self, job: Job[JobData]) -> Transition[JobData]:
        return Transition(outcome="reviewed", reason=job.notes, status="terminal")


__all__ = ["CrossJob"]
