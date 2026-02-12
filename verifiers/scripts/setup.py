import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import wget

VERIFIERS_REPO = "primeintellect-ai/verifiers"
PRIME_RL_REPO = "primeintellect-ai/prime-rl"
VERIFIERS_COMMIT = "main"
PRIME_RL_COMMIT = (
    "main"  # Commit hash, branch name, or tag to use for installed prime-rl version
)
PRIME_RL_INSTALL_SCRIPT_REF = (
    "main"  # Ref to use for fetching the install script itself
)

ENDPOINTS_SRC = f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_COMMIT}/configs/endpoints.toml"
ENDPOINTS_DST = "configs/endpoints.toml"

AGENTS_MD_SRC = f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_COMMIT}/assets/lab/AGENTS.md"
AGENTS_MD_DST = "AGENTS.md"

CLAUDE_MD_SRC = f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_COMMIT}/assets/lab/CLAUDE.md"
CLAUDE_MD_DST = "CLAUDE.md"

ENVS_AGENTS_MD_SRC = f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/{VERIFIERS_COMMIT}/assets/lab/environments/AGENTS.md"
ENVS_AGENTS_MD_DST = "environments/AGENTS.md"

LAB_SKILLS = [
    "create-environments",
    "browse-environments",
    "review-environments",
    "evaluate-environments",
    "optimize-with-environments",
    "train-with-environments",
    "brainstorm",
]
AGENT_SKILLS_DIR_MAP: dict[str, str] = {
    "amp": ".agents/skills",
}
AGENT_SKILL_NAME_MAP: dict[str, dict[str, str]] = {}
SUPPORTED_AGENTS = ("codex", "claude", "cursor", "opencode", "amp")
PRIME_SKILLS_DIR = ".prime/skills"
PRIME_DIR = ".prime"
LAB_METADATA_PATH = os.path.join(PRIME_DIR, "lab.json")

ConfigSpec = tuple[str, str, str]


def _mirror_repo_configs(repo: str, source_paths: list[str]) -> list[ConfigSpec]:
    """Map repo paths to identical destination paths."""
    return [(repo, source_path, source_path) for source_path in source_paths]


PRIME_RL_CONFIGS: list[ConfigSpec] = [
    # (source_repo, source_path, dest_path)
    # Configs can come from either verifiers or prime-rl repo
    (
        VERIFIERS_REPO,
        "configs/local/prime-rl/wiki-search.toml",
        "configs/prime-rl/wiki-search.toml",
    ),
]

RL_CONFIGS = _mirror_repo_configs(
    VERIFIERS_REPO,
    [
        "configs/rl/alphabet-sort.toml",
        "configs/rl/gsm8k.toml",
        "configs/rl/math-python.toml",
        "configs/rl/reverse-text.toml",
        "configs/rl/wiki-search.toml",
        "configs/rl/wordle.toml",
    ],
)

GEPA_CONFIGS = _mirror_repo_configs(
    VERIFIERS_REPO,
    [
        "configs/gepa/base.toml",
        "configs/gepa/wordle.toml",
    ],
)

EVAL_CONFIGS = _mirror_repo_configs(
    VERIFIERS_REPO,
    [
        "configs/eval/minimal.toml",
        "configs/eval/multi-env.toml",
    ],
)


def install_prime_rl():
    """Install prime-rl by running its install script, then checkout the specified commit."""
    if os.path.exists("prime-rl"):
        print("prime-rl directory already exists, skipping installation")
    else:
        print(f"Installing prime-rl (commit ref: {PRIME_RL_COMMIT})...")
        install_url = f"https://raw.githubusercontent.com/{PRIME_RL_REPO}/{PRIME_RL_INSTALL_SCRIPT_REF}/scripts/install.sh"
        install_cmd = [
            "bash",
            "-c",
            f"curl -sSL {install_url} | bash",
        ]
        result = subprocess.run(install_cmd, check=False)
        if result.returncode != 0:
            print(
                f"Error: prime-rl installation failed with exit code {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Checking out prime-rl commit: {PRIME_RL_COMMIT}")
    checkout_cmd = [
        "bash",
        "-c",
        f"cd prime-rl && git checkout {PRIME_RL_COMMIT}",
    ]
    result = subprocess.run(checkout_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(
            f"Error: Failed to checkout prime-rl branch {PRIME_RL_COMMIT}",
            file=sys.stderr,
        )
        sys.stderr.write(result.stderr)
        sys.exit(1)

    print("Syncing prime-rl dependencies...")
    sync_cmd = [
        "bash",
        "-c",
        "cd prime-rl && uv sync && uv sync --all-extras",
    ]
    result = subprocess.run(sync_cmd, check=False)
    if result.returncode != 0:
        print(
            f"Error: Failed to sync prime-rl dependencies with exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(1)
    print("prime-rl setup completed")


def _dedupe_config_destinations(configs: list[ConfigSpec]) -> list[ConfigSpec]:
    """Drop duplicate destination paths while preserving the first occurrence."""
    deduped: list[ConfigSpec] = []
    seen_destinations: set[str] = set()
    for config in configs:
        dest_path = config[2]
        if dest_path in seen_destinations:
            continue
        seen_destinations.add(dest_path)
        deduped.append(config)
    return deduped


def download_configs(configs: list[ConfigSpec]):
    """Download configs from specified repos."""
    for repo, source_path, dest_path in configs:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        ref = PRIME_RL_COMMIT if repo == PRIME_RL_REPO else VERIFIERS_COMMIT
        src = f"https://raw.githubusercontent.com/{repo}/refs/heads/{ref}/{source_path}"
        dst = dest_path
        if not os.path.exists(dst):
            wget.download(src, dst)
            print(f"\nDownloaded {dst} from https://github.com/{repo}")
        else:
            print(f"{dst} already exists")


def sync_endpoints_config():
    """Ensure configs/endpoints.toml exists."""
    os.makedirs(os.path.dirname(ENDPOINTS_DST), exist_ok=True)
    if not os.path.exists(ENDPOINTS_DST):
        wget.download(ENDPOINTS_SRC, ENDPOINTS_DST)
        print(f"\nDownloaded {ENDPOINTS_DST} from https://github.com/{VERIFIERS_REPO}")
    else:
        print(f"{ENDPOINTS_DST} already exists")


def sync_prime_skills():
    """Ensure .prime/skills contains the lab skill set."""
    os.makedirs(PRIME_SKILLS_DIR, exist_ok=True)

    for skill_name in LAB_SKILLS:
        skill_src = (
            f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/"
            f"{VERIFIERS_COMMIT}/skills/{skill_name}/SKILL.md"
        )
        skill_dst = os.path.join(PRIME_SKILLS_DIR, skill_name, "SKILL.md")
        os.makedirs(os.path.dirname(skill_dst), exist_ok=True)

        if not os.path.exists(skill_dst):
            wget.download(skill_src, skill_dst)
            print(f"\nDownloaded {skill_dst} from https://github.com/{VERIFIERS_REPO}")
        else:
            print(f"{skill_dst} already exists")


def install_environments_to_prime_rl():
    """Install all environments from environments/ folder into prime-rl workspace."""
    envs_dir = "environments"
    if not os.path.exists(envs_dir):
        print(f"{envs_dir}/ not found, skipping environment installation")
        return

    if not os.path.exists("prime-rl"):
        print("prime-rl/ not found, skipping environment installation")
        return

    env_modules = []
    for entry in os.listdir(envs_dir):
        env_path = os.path.join(envs_dir, entry)
        if os.path.isdir(env_path) and os.path.exists(
            os.path.join(env_path, "pyproject.toml")
        ):
            env_modules.append(entry)

    if not env_modules:
        print(f"No installable environments found in {envs_dir}/")
        return

    print(f"Installing {len(env_modules)} environments into prime-rl workspace...")
    env_paths = [f"-e environments/{m}" for m in sorted(env_modules)]
    install_cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        "prime-rl/.venv/bin/python",
        *env_paths,
    ]
    result = subprocess.run(install_cmd, check=False)
    if result.returncode != 0:
        print(
            f"Error: Failed to install environments with exit code {result.returncode}",
            file=sys.stderr,
        )
    else:
        print(f"Installed {len(env_modules)} environments")


def ensure_uv_project():
    """Ensure we're in a uv project, initializing one if needed, and add verifiers."""
    if not os.path.exists("pyproject.toml"):
        print("No pyproject.toml found, initializing uv project...")
        print("Running: uv init")
        result = subprocess.run(["uv", "init"], check=False)
        if result.returncode != 0:
            print("Error: Failed to initialize uv project", file=sys.stderr)
            sys.exit(1)

        if os.path.exists("main.py"):
            os.remove("main.py")
        if os.path.exists(".python-version"):
            os.remove(".python-version")

        gitignore_section = """
# outputs from `prime eval run`
./outputs
./environments/*/outputs
"""
        with open(".gitignore", "a") as f:
            f.write(gitignore_section)
    else:
        print("Found existing pyproject.toml")

    print("Running: uv add verifiers")
    result = subprocess.run(["uv", "add", "verifiers"], check=False)
    if result.returncode != 0:
        print("Error: Failed to add verifiers", file=sys.stderr)
        sys.exit(1)


def run_setup(
    prime_rl: bool = False,
    skip_agents_md: bool = False,
    skip_install: bool = False,
    agents: str | None = None,
    no_interactive: bool = False,
) -> None:
    """Run verifiers setup with the specified options.

    Args:
        prime_rl: Install prime-rl and download prime-rl configs.
        skip_agents_md: Skip downloading AGENTS.md, CLAUDE.md, and environments/AGENTS.md.
        skip_install: Skip uv project initialization and verifiers installation.
        agents: Comma-separated coding agents to scaffold.
        no_interactive: Disable interactive coding agent prompts.
    """
    primary_agent, selected_agents, use_multiple_agents = _resolve_setup_agents(
        agents=agents,
        no_interactive=no_interactive,
    )

    if not skip_install:
        ensure_uv_project()

    os.makedirs("configs", exist_ok=True)
    os.makedirs("environments", exist_ok=True)
    sync_prime_skills()
    _prepare_agent_skill_dirs(selected_agents)
    _sync_lab_metadata(
        primary_agent=primary_agent,
        selected_agents=selected_agents,
        use_multiple_agents=use_multiple_agents,
    )

    if not skip_agents_md:
        if os.path.exists(AGENTS_MD_DST):
            os.remove(AGENTS_MD_DST)
        wget.download(AGENTS_MD_SRC, AGENTS_MD_DST)
        print(f"\nDownloaded {AGENTS_MD_DST} from https://github.com/{VERIFIERS_REPO}")

        if os.path.exists(CLAUDE_MD_DST):
            os.remove(CLAUDE_MD_DST)
        wget.download(CLAUDE_MD_SRC, CLAUDE_MD_DST)
        print(f"\nDownloaded {CLAUDE_MD_DST} from https://github.com/{VERIFIERS_REPO}")

        if os.path.exists(ENVS_AGENTS_MD_DST):
            os.remove(ENVS_AGENTS_MD_DST)
        wget.download(ENVS_AGENTS_MD_SRC, ENVS_AGENTS_MD_DST)
        print(
            f"\nDownloaded {ENVS_AGENTS_MD_DST} from https://github.com/{VERIFIERS_REPO}"
        )

    if prime_rl:
        install_prime_rl()
        install_environments_to_prime_rl()

    sync_endpoints_config()

    configs_to_download: list[ConfigSpec] = []
    if prime_rl:
        configs_to_download.extend(PRIME_RL_CONFIGS)
    configs_to_download.extend(GEPA_CONFIGS)
    configs_to_download.extend(EVAL_CONFIGS)
    if not prime_rl:
        configs_to_download.extend(RL_CONFIGS)

    download_configs(_dedupe_config_destinations(configs_to_download))
    print_post_setup_call_to_action(prime_rl=prime_rl, primary_agent=primary_agent)


def print_post_setup_call_to_action(prime_rl: bool, primary_agent: str) -> None:
    """Print practical next steps after setup."""
    prompt_heading = f"ask {primary_agent}"
    prompt_body = (
        "I want to train a model for <my task domain>. Propose an initial environment "
        "scaffold including relevant tools, and come up with a good method to generate "
        "a small sample synthetic dataset. Run a quick eval baseline, inspect the "
        "results, and then decide how we should iterate on refining the implementation."
    )
    prompt_text = Text(
        prompt_body,
        style="italic",
    )

    command_table = Table.grid(padding=(0, 1))
    command_table.add_row("[bold green]$[/bold green]", "prime env init my-env")
    command_table.add_row(
        "[bold green]$[/bold green]", "prime eval run my-env -m gpt-5-nano -n 5"
    )
    command_table.add_row("[bold green]$[/bold green]", "prime eval tui")
    if prime_rl:
        command_table.add_row(
            "[bold green]$[/bold green]",
            "uv run prime-rl configs/prime-rl/wiki-search.toml",
        )
    else:
        command_table.add_row(
            "[bold green]$[/bold green]", "prime rl run configs/rl/wiki-search.toml"
        )
    command_table.add_row(
        "[bold green]$[/bold green]", "prime gepa run my-env -m gpt-5-nano"
    )

    header_text = Text.assemble(
        ("idea -> environment -> eval -> training", "dim"),
    )

    content = Group(
        header_text,
        Panel(
            prompt_text,
            title=prompt_heading,
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
            expand=False,
        ),
        Panel(
            command_table,
            title="quick commands",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        ),
    )

    console = Console()
    console.print()
    console.print(
        Panel(
            content,
            title="[bold white]get started[/bold white]",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2),
            expand=False,
        )
    )


def _load_lab_metadata() -> dict[str, object]:
    """Load persisted lab metadata from .prime directory."""
    metadata_path = Path(LAB_METADATA_PATH)
    if not metadata_path.exists():
        return {}
    try:
        raw = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _normalize_agent(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_AGENTS:
        allowed = ", ".join(SUPPORTED_AGENTS)
        raise ValueError(
            f"Unsupported coding agent '{value}'. Supported values: {allowed}"
        )
    return normalized


def _parse_agents_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [token.strip() for token in value.split(",") if token.strip()]
    if not parsed:
        return None
    deduped: list[str] = []
    seen: set[str] = set()
    for token in parsed:
        normalized = _normalize_agent(token)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _prompt_for_agents() -> list[str]:
    print(f"Supported coding agents: {', '.join(SUPPORTED_AGENTS)}")
    while True:
        raw_primary = input("Primary coding agent [codex]: ").strip()
        primary = raw_primary if raw_primary else "codex"
        try:
            normalized_primary = _normalize_agent(primary)
            break
        except ValueError as exc:
            print(exc)

    selected = [normalized_primary]
    use_multiple_raw = input("Using multiple coding agents? [y/N]: ").strip().lower()
    if use_multiple_raw in {"y", "yes"}:
        while True:
            additional_raw = input("Additional agents (comma-separated): ").strip()
            try:
                additional = _parse_agents_csv(additional_raw) or []
            except ValueError as exc:
                print(exc)
                continue
            for agent in additional:
                if agent not in selected:
                    selected.append(agent)
            break
    return selected


def _resolve_setup_agents(
    agents: str | None, no_interactive: bool
) -> tuple[str, list[str], bool]:
    if agents is not None:
        selected = _parse_agents_csv(agents)
        if selected is None:
            raise RuntimeError(
                "No valid coding agents provided. Supported values: "
                + ", ".join(SUPPORTED_AGENTS)
            )
    elif not no_interactive and sys.stdin.isatty():
        selected = _prompt_for_agents()
    else:
        selected = ["codex"]

    primary_agent = selected[0]
    use_multiple_agents = len(selected) > 1
    return primary_agent, selected, use_multiple_agents


def _prepare_agent_skill_dirs(agents: list[str]) -> None:
    prime_skills_dir = Path(PRIME_SKILLS_DIR)
    for agent in agents:
        skills_dir = _resolve_agent_skills_dir(agent)
        skills_dir.mkdir(parents=True, exist_ok=True)
        for skill_name in LAB_SKILLS:
            source_skill_dir = prime_skills_dir / skill_name
            if not source_skill_dir.exists():
                continue
            target_skill_name = _resolve_agent_skill_name(agent, skill_name)
            target_skill_dir = skills_dir / target_skill_name
            _safe_link_or_copy_skill_dir(source_skill_dir, target_skill_dir)
        print(f"Prepared {skills_dir}")


def _safe_link_or_copy_skill_dir(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return

    try:
        relative_source = os.path.relpath(source, start=target.parent)
        target.symlink_to(relative_source, target_is_directory=True)
        return
    except OSError:
        shutil.copytree(source, target, dirs_exist_ok=True)


def _resolve_agent_skills_dir(agent: str) -> Path:
    mapped_dir = AGENT_SKILLS_DIR_MAP.get(agent)
    if mapped_dir is not None:
        return Path(mapped_dir)
    return Path(f".{agent}") / "skills"


def _resolve_agent_skill_name(agent: str, skill_name: str) -> str:
    return AGENT_SKILL_NAME_MAP.get(agent, {}).get(skill_name, skill_name)


def _sync_lab_metadata(
    *, primary_agent: str, selected_agents: list[str], use_multiple_agents: bool
) -> None:
    """Persist lab setup metadata in .prime/."""
    os.makedirs(PRIME_DIR, exist_ok=True)

    metadata = _load_lab_metadata()
    metadata["setup_source"] = "prime lab setup"
    metadata["choices"] = {
        "agents": selected_agents,
        "primary_agent": primary_agent,
        "use_multiple_agents": use_multiple_agents,
    }

    metadata_path = Path(LAB_METADATA_PATH)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Setup verifiers development workspace"
    )
    parser.add_argument(
        "--prime-rl",
        action="store_true",
        help="Install prime-rl and download prime-rl configs",
    )
    parser.add_argument(
        "--skip-agents-md",
        action="store_true",
        help="Skip downloading AGENTS.md, CLAUDE.md, and environments/AGENTS.md",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip uv project initialization and verifiers installation",
    )
    parser.add_argument(
        "--agents",
        help="Comma-separated coding agents to scaffold (codex,claude,cursor,opencode,amp)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Disable interactive coding agent prompts",
    )
    args = parser.parse_args()

    run_setup(
        prime_rl=args.prime_rl,
        skip_agents_md=args.skip_agents_md,
        skip_install=args.skip_install,
        agents=args.agents,
        no_interactive=args.no_interactive,
    )


if __name__ == "__main__":
    main()
