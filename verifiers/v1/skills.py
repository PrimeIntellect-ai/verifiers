"""SKILL.md-based agent skills, installed into the harness runtime."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    root: Path
    """Local skill directory; every file under it ships to the runtime."""

    def files(self) -> Iterator[Path]:
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            hidden = any(
                part.startswith(".") or part == "__pycache__" for part in relative.parts
            )
            if path.is_file() and not hidden:
                yield path


def load_skills(paths: list[Path]) -> list[Skill]:
    """Resolve each path — one skill (a directory with a `SKILL.md` manifest, or the
    manifest itself) or a directory of skill directories — into `Skill`s."""
    return [_load(root) for path in paths for root in _skill_dirs(Path(path))]


def _skill_dirs(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"skill path {path} does not exist")
    if path.is_file():
        return [path.parent]
    if (path / MANIFEST).is_file():
        return [path]
    roots = sorted(d for d in path.iterdir() if (d / MANIFEST).is_file())
    if not roots:
        raise ValueError(f"no {MANIFEST} under {path} or its immediate subdirectories")
    return roots


def _load(root: Path) -> Skill:
    meta = _frontmatter((root / MANIFEST).read_text())
    return Skill(
        name=meta.get("name", root.name),
        description=meta.get("description", ""),
        root=root,
    )


def _frontmatter(text: str) -> dict[str, str]:
    """Top-level `key: value` pairs of the manifest's `---` frontmatter block;
    nested or multiline YAML values are ignored (only name/description are read)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key[0].isspace():
            meta[key.strip()] = value.strip().strip("'\"")
    return meta


def skills_prompt(skills: list[Skill], dest: str) -> str:
    """The prompt section announcing installed skills to a program without native
    skill discovery."""
    listing = "\n".join(
        f"- {skill.name}: {skill.description} ({dest}/{skill.name}/{MANIFEST})"
        for skill in skills
    )
    return (
        f"# Skills\n\n"
        f"Skills are directories of instructions for specific kinds of tasks, "
        f"installed under `{dest}/`:\n\n{listing}\n\n"
        f"When a task matches a skill's description, read its {MANIFEST} first "
        f"and follow its instructions."
    )
