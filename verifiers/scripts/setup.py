import argparse
import os
import subprocess
import sys

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
) -> None:
    """Run verifiers setup with the specified options.

    Args:
        prime_rl: Install prime-rl and download prime-rl configs.
        skip_agents_md: Skip downloading AGENTS.md, CLAUDE.md, and environments/AGENTS.md.
        skip_install: Skip uv project initialization and verifiers installation.
    """
    if not skip_install:
        ensure_uv_project()

    os.makedirs("configs", exist_ok=True)
    os.makedirs("environments", exist_ok=True)

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

    if not os.path.exists(ENDPOINTS_DST):
        wget.download(ENDPOINTS_SRC, ENDPOINTS_DST)
        print(f"\nDownloaded {ENDPOINTS_DST} from https://github.com/{VERIFIERS_REPO}")
    else:
        print(f"{ENDPOINTS_DST} already exists")

    configs_to_download: list[ConfigSpec] = []
    if prime_rl:
        configs_to_download.extend(PRIME_RL_CONFIGS)
    configs_to_download.extend(GEPA_CONFIGS)
    configs_to_download.extend(EVAL_CONFIGS)
    if not prime_rl:
        configs_to_download.extend(RL_CONFIGS)

    download_configs(_dedupe_config_destinations(configs_to_download))


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
    args = parser.parse_args()

    run_setup(
        prime_rl=args.prime_rl,
        skip_agents_md=args.skip_agents_md,
        skip_install=args.skip_install,
    )


if __name__ == "__main__":
    main()
