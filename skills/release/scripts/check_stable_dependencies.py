"""Reject prerelease and direct-source dependencies before a stable release."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SPECIFIER = re.compile(r"(?:===|==|~=|!=|<=|>=|<|>)\s*([^,;\s]+)")
DIRECT_REFERENCE = re.compile(r"\s*@\s*\S+")
PRERELEASE = re.compile(
    r"(?:\.dev\d*|(?<=\d)(?:a|b|rc)\d+|[-_.](?:alpha|beta|pre|preview|rc)\d*)",
    re.IGNORECASE,
)


def dependency_lists(config: dict[str, Any]) -> Iterable[tuple[str, list[Any]]]:
    build_requires = config.get("build-system", {}).get("requires", [])
    yield "build-system.requires", build_requires

    project = config.get("project", {})
    yield "project.dependencies", project.get("dependencies", [])
    for group, requirements in project.get("optional-dependencies", {}).items():
        yield f"project.optional-dependencies.{group}", requirements

    for group, requirements in config.get("dependency-groups", {}).items():
        yield f"dependency-groups.{group}", requirements

    uv_config = config.get("tool", {}).get("uv", {})
    for key in ("constraint-dependencies", "override-dependencies"):
        yield f"tool.uv.{key}", uv_config.get(key, [])


def requirement_issues(location: str, requirement: str) -> list[str]:
    issues: list[str] = []
    requirement_without_marker = requirement.partition(";")[0].strip()

    if DIRECT_REFERENCE.search(requirement_without_marker):
        issues.append(f"{location}: direct dependency reference: {requirement}")

    for version in SPECIFIER.findall(requirement_without_marker):
        if PRERELEASE.search(version):
            issues.append(
                f"{location}: prerelease version '{version}' in: {requirement}"
            )

    return issues


def source_issues(config: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    sources = config.get("tool", {}).get("uv", {}).get("sources", {})

    for package, value in sources.items():
        variants = value if isinstance(value, list) else [value]
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            if "git" in variant or "url" in variant:
                issues.append(
                    f"tool.uv.sources.{package}: direct Git or URL source: {variant}"
                )

    return issues


def check(config: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    for location, requirements in dependency_lists(config):
        for requirement in requirements:
            if isinstance(requirement, str):
                issues.extend(requirement_issues(location, requirement))

    issues.extend(source_issues(config))
    return issues


def load_config(path: str) -> dict[str, Any]:
    if path == "-":
        return tomllib.load(sys.stdin.buffer)
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="pyproject.toml")
    args = parser.parse_args()

    issues = check(load_config(args.path))
    if issues:
        print("Stable release dependency check failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print(f"{args.path} uses stable registry dependency versions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
