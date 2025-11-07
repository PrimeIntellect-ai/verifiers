import os
import subprocess
import sys

import wget

VERIFIERS_REPO = "primeintellect-ai/verifiers"
PRIME_RL_REPO = "primeintellect-ai/prime-rl"
PRIME_RL_COMMIT = (
    "main"  # Commit hash, branch name, or tag to use for installed prime-rl version
)
PRIME_RL_INSTALL_SCRIPT_REF = (
    "main"  # Ref to use for fetching the install script itself
)

ENDPOINTS_SRC = f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/main/configs/endpoints.py"
ENDPOINTS_DST = "configs/endpoints.py"

ZERO3_SRC = f"https://raw.githubusercontent.com/{VERIFIERS_REPO}/refs/heads/main/configs/zero3.yaml"
ZERO3_DST = "configs/zero3.yaml"

VERIFIERS_CONFIGS = [
    # (source_repo, source_path, dest_path)
    (
        VERIFIERS_REPO,
        "configs/vf-rl/alphabet-sort.toml",
        "configs/vf-rl/alphabet-sort.toml",
    ),
    (
        VERIFIERS_REPO,
        "configs/vf-rl/gsm8k.toml",
        "configs/vf-rl/gsm8k.toml",
    ),
    (
        VERIFIERS_REPO,
        "configs/vf-rl/math-python.toml",
        "configs/vf-rl/math-python.toml",
    ),
    (
        VERIFIERS_REPO,
        "configs/vf-rl/reverse-text.toml",
        "configs/vf-rl/reverse-text.toml",
    ),
    (
        VERIFIERS_REPO,
        "configs/vf-rl/wordle.toml",
        "configs/vf-rl/wordle.toml",
    ),
    (
        VERIFIERS_REPO,
        "configs/vf-rl/tool-test.toml",
        "configs/vf-rl/tool-test.toml",
    ),
]

PRIME_RL_CONFIGS = [
    # (source_repo, source_path, dest_path)
    # Configs can come from either verifiers or prime-rl repo
    (
        VERIFIERS_REPO,
        "configs/prime-rl/wiki-search.toml",
        "configs/prime-rl/wiki-search.toml",
    ),
]


def install_prime_rl():
    """Install prime-rl by running its install script, then checkout the specified commit."""
    if os.path.exists("prime-rl"):
        print("prime-rl directory already exists, skipping installation")
    else:
        print(f"Installing prime-rl (will checkout commit: {PRIME_RL_COMMIT})...")
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


def download_configs(configs):
    """Download configs from specified repos."""
    for repo, source_path, dest_path in configs:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        ref = PRIME_RL_COMMIT if repo == PRIME_RL_REPO else "main"
        src = f"https://raw.githubusercontent.com/{repo}/{ref}/{source_path}"
        dst = dest_path
        if not os.path.exists(dst):
            wget.download(src, dst)
            print(f"\nDownloaded {dst} from https://github.com/{repo}")
        else:
            print(f"{dst} already exists")


def main():
    os.makedirs("configs", exist_ok=True)

    install_prime_rl()

    if not os.path.exists(ENDPOINTS_DST):
        wget.download(ENDPOINTS_SRC, ENDPOINTS_DST)
        print(f"\nDownloaded {ENDPOINTS_DST} from https://github.com/{VERIFIERS_REPO}")
    else:
        print(f"{ENDPOINTS_DST} already exists")

    if not os.path.exists(ZERO3_DST):
        wget.download(ZERO3_SRC, ZERO3_DST)
        print(f"\nDownloaded {ZERO3_DST} from https://github.com/{VERIFIERS_REPO}")
    else:
        print(f"{ZERO3_DST} already exists")

    download_configs(VERIFIERS_CONFIGS)
    download_configs(PRIME_RL_CONFIGS)


if __name__ == "__main__":
    main()
