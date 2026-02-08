#!/usr/bin/env python3
"""Compile AGENTS.md / CLAUDE.md files from modular docs sources."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent

REPO_GENERATED_NOTE = (
    "<!-- Generated for repository development workflows. Do not edit directly. -->"
)
LAB_GENERATED_NOTE = "<!-- Generated for lab workspaces. -->"


def read_without_title(path: Path) -> str:
    """Read a markdown file, skipping its first title line and leading blank lines."""
    lines = path.read_text().splitlines()
    start = 1
    while start < len(lines) and lines[start].strip() == "":
        start += 1
    return "\n".join(lines[start:]).strip()


def read_markdown(path: Path) -> str:
    """Read markdown and trim outer whitespace."""
    return path.read_text().strip()


def combine_sections(sections: list[str]) -> str:
    """Combine markdown sections with blank lines."""
    return (
        "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"
    )


def write_if_changed(path: Path, content: str) -> None:
    """Write file only when content changed."""
    if path.exists() and path.read_text() == content:
        print(f"{path} is up to date")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"Updated {path}")


def compile_agents() -> None:
    common = read_markdown(ROOT / "assets" / "agents" / "common_best_practices.md")
    repo_only = read_markdown(
        ROOT / "assets" / "agents" / "repo_development_best_practices.md"
    )
    end_user = read_markdown(ROOT / "assets" / "agents" / "end_user_best_practices.md")

    repo_agents = combine_sections(
        [
            "# AGENTS.md",
            REPO_GENERATED_NOTE,
            common,
            repo_only,
        ]
    )
    write_if_changed(ROOT / "AGENTS.md", repo_agents)

    lab_agents = combine_sections(
        [
            "# AGENTS.md",
            LAB_GENERATED_NOTE,
            "This AGENTS guide is intended for end users working in a `prime lab setup` workspace.",
            common,
            end_user,
        ]
    )
    write_if_changed(ROOT / "assets" / "lab" / "AGENTS.md", lab_agents)

    repo_claude = combine_sections(
        [
            "# CLAUDE.md",
            REPO_GENERATED_NOTE,
            "Before beginning work in this repository, read `AGENTS.md` and follow all scoped AGENTS guidance.",
        ]
    )
    write_if_changed(ROOT / "CLAUDE.md", repo_claude)

    lab_claude = combine_sections(
        [
            "# CLAUDE.md",
            LAB_GENERATED_NOTE,
            "Before beginning any task, read `AGENTS.md` and `environments/AGENTS.md` in this workspace.",
            "Treat all `AGENTS.md` files as equivalent to `CLAUDE.md` files.",
        ]
    )
    write_if_changed(ROOT / "assets" / "lab" / "CLAUDE.md", lab_claude)


def compile_environment_guides() -> None:
    envs_body = read_without_title(ROOT / "docs" / "environments.md")
    repo_envs_agents = combine_sections(
        [
            "# environments/AGENTS.md",
            REPO_GENERATED_NOTE,
            'This file mirrors the "Environments" documentation page.',
            "---",
            envs_body,
        ]
    )
    lab_envs_agents = combine_sections(
        [
            "# environments/AGENTS.md",
            LAB_GENERATED_NOTE,
            'This file mirrors the "Environments" documentation page.',
            "---",
            envs_body,
        ]
    )

    write_if_changed(ROOT / "environments" / "AGENTS.md", repo_envs_agents)
    write_if_changed(
        ROOT / "assets" / "lab" / "environments" / "AGENTS.md", lab_envs_agents
    )


def main() -> None:
    compile_agents()
    compile_environment_guides()


if __name__ == "__main__":
    main()
