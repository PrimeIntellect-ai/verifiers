#!/usr/bin/env python3
"""Sync AGENTS.md files from docs/ sources.

This script generates AGENTS.md files from the docs/ folder:
- AGENTS.md <- docs/overview.md (minus title)
- environments/AGENTS.md <- docs/environments.md (minus title)

Run manually or via pre-commit hook.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

AGENTS_MD_HEADER = """\
# AGENTS.md

This guide covers best practices for building environments with `verifiers` and using them to train and evaluate LLMs. It is downloaded automatically using the setup script below (which has likely already been run if you're reading this). See `environments/AGENTS.md` for more details.

---

"""

ENVS_AGENTS_MD_HEADER = """\
# environments/AGENTS.md

This file mirrors the "Environments" section from the Verifiers documentation, and is downloaded automatically using the setup script.

---

"""


def read_without_title(path: Path) -> str:
    """Read a markdown file, skipping the first line (title)."""
    content = path.read_text()
    lines = content.split("\n")
    # Skip title line and any immediately following blank lines
    start = 1
    while start < len(lines) and lines[start].strip() == "":
        start += 1
    return "\n".join(lines[start:])


def main():
    # Generate AGENTS.md from docs/overview.md
    overview_content = read_without_title(ROOT / "docs" / "overview.md")
    agents_md = AGENTS_MD_HEADER + overview_content
    agents_path = ROOT / "AGENTS.md"
    if agents_path.read_text() != agents_md:
        agents_path.write_text(agents_md)
        print(f"Updated {agents_path}")
    else:
        print(f"{agents_path} is up to date")

    # Generate environments/AGENTS.md from docs/environments.md
    envs_content = read_without_title(ROOT / "docs" / "environments.md")
    envs_agents_md = ENVS_AGENTS_MD_HEADER + envs_content
    envs_agents_path = ROOT / "environments" / "AGENTS.md"
    if envs_agents_path.read_text() != envs_agents_md:
        envs_agents_path.write_text(envs_agents_md)
        print(f"Updated {envs_agents_path}")
    else:
        print(f"{envs_agents_path} is up to date")


if __name__ == "__main__":
    main()
