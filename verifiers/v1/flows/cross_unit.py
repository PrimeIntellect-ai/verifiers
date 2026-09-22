"""An audit requests repair; the fixer steers the waiting task to review. Runs offline."""

from verifiers.v1.flow import Flow, Transition, Unit, UnitData, stage


class Repair(UnitData):
    task: str


class CrossUnit(Flow):
    async def setup(self) -> None:
        self.create_unit("task", stage="audit", data=UnitData())

    @stage
    async def audit(self, unit: Unit[UnitData]) -> Transition[UnitData]:
        self.create_unit("repair", stage="fix", data=Repair(task=unit.id))
        return Transition("needs_repair", status="waiting")

    @stage
    async def fix(self, unit: Unit[Repair]) -> Transition[Repair]:
        # After performing the repair, choose where the waiting task resumes.
        self.unit(unit.data.task).steer(
            stage="review", status="ready", note="Review the repaired task."
        )
        return Transition("released", status="terminal")

    @stage
    async def review(self, unit: Unit[UnitData]) -> Transition[UnitData]:
        return Transition("reviewed", unit.notes, status="terminal")


__all__ = ["CrossUnit"]
