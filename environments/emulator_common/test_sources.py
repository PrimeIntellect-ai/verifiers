import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonData: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class TestSource:
    name: str
    kind: str
    url: str
    covers: tuple[str, ...]
    license: str = "unknown"
    ref: str | None = None
    sparse_paths: tuple[str, ...] = ()
    artifact_globs: tuple[str, ...] = ()
    artifact_extensions: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_mapping(cls, value: JsonData) -> "TestSource":
        missing = {"name", "kind", "url", "covers"} - set(value)
        if missing:
            raise ValueError(f"test source is missing fields: {sorted(missing)}")
        return cls(
            name=str(value["name"]),
            kind=str(value["kind"]),
            url=str(value["url"]),
            covers=tuple(map(str, value["covers"])),
            license=str(value.get("license", "unknown")),
            ref=str(value["ref"]) if value.get("ref") is not None else None,
            sparse_paths=tuple(map(str, value.get("sparse_paths", ()))),
            artifact_globs=tuple(map(str, value.get("artifact_globs", ()))),
            artifact_extensions=tuple(map(str, value.get("artifact_extensions", ()))),
            notes=str(value.get("notes", "")),
        )

    def to_dict(self) -> JsonData:
        payload: JsonData = {
            "name": self.name,
            "kind": self.kind,
            "url": self.url,
            "covers": list(self.covers),
            "license": self.license,
            "notes": self.notes,
        }
        if self.ref is not None:
            payload["ref"] = self.ref
        if self.sparse_paths:
            payload["sparse_paths"] = list(self.sparse_paths)
        if self.artifact_globs:
            payload["artifact_globs"] = list(self.artifact_globs)
        if self.artifact_extensions:
            payload["artifact_extensions"] = list(self.artifact_extensions)
        return payload


@dataclass(frozen=True)
class PublicTestManifest:
    environment_id: str
    sources: tuple[TestSource, ...]
    private_artifacts: tuple[JsonData, ...] = ()

    @classmethod
    def from_mapping(cls, value: JsonData) -> "PublicTestManifest":
        missing = {"environment_id", "sources"} - set(value)
        if missing:
            raise ValueError(
                f"public test manifest is missing fields: {sorted(missing)}"
            )
        sources = tuple(TestSource.from_mapping(item) for item in value["sources"])
        if not sources:
            raise ValueError("public test manifest must define at least one source")
        return cls(
            environment_id=str(value["environment_id"]),
            sources=sources,
            private_artifacts=tuple(
                dict(item) for item in value.get("private_artifacts", ())
            ),
        )

    def covered_names(self) -> set[str]:
        return {name for source in self.sources for name in source.covers}

    def to_dict(self) -> JsonData:
        return {
            "environment_id": self.environment_id,
            "sources": [source.to_dict() for source in self.sources],
            "private_artifacts": [dict(item) for item in self.private_artifacts],
        }


def load_public_test_manifest(path: str | Path) -> PublicTestManifest:
    with Path(path).open(encoding="utf-8") as f:
        return PublicTestManifest.from_mapping(json.load(f))


def verify_source_reachable(source: TestSource, timeout: int = 20) -> tuple[bool, str]:
    if source.kind == "git":
        command = ["git", "ls-remote", "--heads", source.url]
        if source.ref:
            command.append(source.ref)
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode == 0:
            if source.ref and not result.stdout.strip():
                return False, f"missing git branch/ref: {source.ref}"
            return True, result.stdout.strip().splitlines()[
                0
            ] if result.stdout.strip() else "ok"
        return False, (result.stderr or result.stdout).strip()

    if source.kind in {"http", "private-artifact"}:
        headers = {"User-Agent": "verifiers-emulator-source-check/0.1"}
        request = urllib.request.Request(source.url, method="HEAD", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= response.status < 400, f"HTTP {response.status}"
        except Exception:
            try:
                fallback = urllib.request.Request(source.url, headers=headers)
                with urllib.request.urlopen(fallback, timeout=timeout) as response:
                    return 200 <= response.status < 400, f"HTTP {response.status}"
            except Exception as exc:
                return False, str(exc)

    return False, f"unsupported source kind: {source.kind}"
