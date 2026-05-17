from dataclasses import dataclass, field
from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonData: TypeAlias = dict[str, JsonValue]


REQUIRED_MANIFEST_FIELDS = {
    "environment_id",
    "module_name",
    "level",
    "slug",
    "display_name",
    "difficulty",
    "goal",
    "base_prompt",
    "requirements",
    "core_concepts",
    "public_verification",
    "hidden_verification",
    "success_criteria",
    "verification_cases",
}


@dataclass(frozen=True)
class VerificationCase:
    id: str
    stage: str
    kind: str
    name: str
    expected_signal: str
    source: str = "external"
    notes: str = ""

    @classmethod
    def from_mapping(cls, value: JsonData) -> "VerificationCase":
        missing = {
            key
            for key in ("id", "stage", "kind", "name", "expected_signal")
            if key not in value
        }
        if missing:
            raise ValueError(f"verification case is missing fields: {sorted(missing)}")
        return cls(
            id=str(value["id"]),
            stage=str(value["stage"]),
            kind=str(value["kind"]),
            name=str(value["name"]),
            expected_signal=str(value["expected_signal"]),
            source=str(value.get("source", "external")),
            notes=str(value.get("notes", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "stage": self.stage,
            "kind": self.kind,
            "name": self.name,
            "expected_signal": self.expected_signal,
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class EmulatorManifest:
    environment_id: str
    module_name: str
    level: int
    slug: str
    display_name: str
    difficulty: str
    goal: str
    base_prompt: str
    requirements: tuple[str, ...]
    core_concepts: tuple[str, ...]
    public_verification: tuple[str, ...]
    hidden_verification: tuple[str, ...]
    success_criteria: tuple[str, ...]
    verification_cases: tuple[VerificationCase, ...]
    exposed_api: tuple[str, ...] = ()
    framebuffer: dict[str, int] = field(default_factory=dict)
    runtime: dict[str, int] = field(default_factory=dict)
    scoring_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: JsonData) -> "EmulatorManifest":
        missing = REQUIRED_MANIFEST_FIELDS - set(value)
        if missing:
            raise ValueError(f"manifest is missing fields: {sorted(missing)}")
        cases = tuple(
            VerificationCase.from_mapping(case)
            for case in value.get("verification_cases", [])
        )
        if not cases:
            raise ValueError("manifest must define at least one verification case")
        return cls(
            environment_id=str(value["environment_id"]),
            module_name=str(value["module_name"]),
            level=int(value["level"]),
            slug=str(value["slug"]),
            display_name=str(value["display_name"]),
            difficulty=str(value["difficulty"]),
            goal=str(value["goal"]),
            base_prompt=str(value["base_prompt"]).strip(),
            requirements=tuple(map(str, value["requirements"])),
            core_concepts=tuple(map(str, value["core_concepts"])),
            public_verification=tuple(map(str, value["public_verification"])),
            hidden_verification=tuple(map(str, value["hidden_verification"])),
            success_criteria=tuple(map(str, value["success_criteria"])),
            verification_cases=cases,
            exposed_api=tuple(map(str, value.get("exposed_api", ()))),
            framebuffer={
                str(k): int(v) for k, v in value.get("framebuffer", {}).items()
            },
            runtime={str(k): int(v) for k, v in value.get("runtime", {}).items()},
            scoring_weights={
                str(k): float(v) for k, v in value.get("scoring_weights", {}).items()
            },
        )

    def public_cases(self) -> list[dict[str, str]]:
        return [
            case.to_dict() for case in self.verification_cases if case.stage == "public"
        ]

    def hidden_cases(self) -> list[dict[str, str]]:
        return [
            case.to_dict() for case in self.verification_cases if case.stage == "hidden"
        ]

    def to_info(self) -> JsonData:
        return {
            "environment_id": self.environment_id,
            "module_name": self.module_name,
            "level": self.level,
            "slug": self.slug,
            "display_name": self.display_name,
            "difficulty": self.difficulty,
            "goal": self.goal,
            "requirements": list(self.requirements),
            "core_concepts": list(self.core_concepts),
            "exposed_api": list(self.exposed_api),
            "public_verification": list(self.public_verification),
            "hidden_verification": list(self.hidden_verification),
            "success_criteria": list(self.success_criteria),
            "public_cases": self.public_cases(),
            "hidden_cases": self.hidden_cases(),
            "framebuffer": dict(self.framebuffer),
            "runtime": dict(self.runtime),
            "scoring_weights": dict(self.scoring_weights),
        }
