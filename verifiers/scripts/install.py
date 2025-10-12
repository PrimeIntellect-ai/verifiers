import argparse
import subprocess
from pathlib import Path

"""
Install a local environment

Usage:
    vf-install <env_id> -p <path>

Options:
    -h, --help    Show this help message and exit
    -d, --local-dir-name <local_dir_name>    The name of the local directory to install the environment into.

"""


def install_environment(env: str, path: str, from_repo: bool, branch: str):
    env_folder = env.replace("-", "_")
    env_name = env_folder.replace("_", "-")
    if from_repo:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                f"{env_name} @ git+https://github.com/PrimeIntellect-ai/verifiers.git@{branch}#subdirectory=environments/{env_folder}",
            ]
        )
    else:
        env_path = Path(path) / env_folder
        subprocess.run(["uv", "pip", "install", "-e", env_path])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="The environment id to install")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path to environments directory (default: ./environments)",
        default="./environments",
    )
    parser.add_argument(
        "-r",
        "--from-repo",
        action="store_true",
        help="Install from the Verifiers repo (default: False)",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--branch",
        type=str,
        help="Branch to install from if --from-repo is True (default: main)",
        default="main",
    )
    args = parser.parse_args()

    install_environment(
        env=args.env,
        path=args.path,
        from_repo=args.from_repo,
        branch=args.branch,
    )


if __name__ == "__main__":
    main()
