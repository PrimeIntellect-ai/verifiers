import argparse

from verifiers.utils.install_utils import (
    install_from_hub,
    install_from_local,
    install_from_repo,
    is_hub_env,
)

"""
Install a verifiers environment

Usage:
    vf-install <env_id>                    # Install from Hub (owner/name) or local
    vf-install <env_id> -p <path>          # Install from local path
    vf-install <env_id> -r                 # Install from verifiers repo

Examples:
    vf-install primeintellect/gsm8k        # Install from Hub
    vf-install gsm8k                       # Install from ./environments/gsm8k
    vf-install gsm8k -p /path/to/envs      # Install from custom local path
    vf-install gsm8k -r                    # Install from GitHub repo
"""


def main():
    parser = argparse.ArgumentParser(
        description="Install a verifiers environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vf-install primeintellect/gsm8k        Install from Environments Hub
  vf-install gsm8k                       Install from ./environments/gsm8k
  vf-install gsm8k -p /path/to/envs      Install from custom local path
  vf-install gsm8k -r                    Install from GitHub repo
        """,
    )
    parser.add_argument(
        "env", type=str, help="Environment ID (owner/name for Hub, or local name)"
    )
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
        help="Install from the verifiers GitHub repo",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--branch",
        type=str,
        help="Branch to install from if --from-repo is set (default: main)",
        default="main",
    )
    args = parser.parse_args()

    if args.from_repo:
        success = install_from_repo(args.env, args.branch)
    elif is_hub_env(args.env):
        success = install_from_hub(args.env)
    else:
        success = install_from_local(args.env, args.path)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
