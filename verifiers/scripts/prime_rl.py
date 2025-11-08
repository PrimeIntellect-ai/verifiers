#!/usr/bin/env python3
"""
Wrapper script to run prime-rl rl command from the current working directory.

Usage:
    uv run prime-rl @ configs/prime-rl/config.toml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def tmux_exists() -> bool:
    try:
        subprocess.run(
            ["tmux", "-V"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def session_exists(session: str) -> bool:
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def find_available_session_name(base_name: str) -> str:
    if not session_exists(base_name):
        return base_name
    index = 2
    while True:
        candidate = f"{base_name}-{index}"
        if not session_exists(candidate):
            return candidate
        index += 1


def create_tmux_session(session: str, output_dir: str, cmd: str, cwd: Path) -> None:
    """Create tmux session matching scripts/tmux.sh layout."""
    # Start new tmux session detached
    run(["tmux", "new-session", "-d", "-s", session, "-n", "RL", "-c", str(cwd)])

    # Window 1: RL - 3 vertical panes
    run(["tmux", "split-window", "-v", "-t", f"{session}:RL.0", "-c", str(cwd)])
    run(["tmux", "split-window", "-v", "-t", f"{session}:RL.1", "-c", str(cwd)])
    run(["tmux", "select-layout", "-t", f"{session}:RL", "even-vertical"])

    # Pane titles
    run(["tmux", "select-pane", "-t", f"{session}:RL.0", "-T", "Trainer"])
    run(["tmux", "select-pane", "-t", f"{session}:RL.1", "-T", "Orchestrator"])
    run(["tmux", "select-pane", "-t", f"{session}:RL.2", "-T", "Inference"])

    # Logs: Orchestrator
    orchestrator_log_cmd = f"""while true; do
echo "Waiting for orchestrator log file..."
while [ ! -f {output_dir}/logs/orchestrator.stdout ]; do sleep 1; done
echo "Following orchestrator.stdout..."
tail -F {output_dir}/logs/orchestrator.stdout
done"""
    run(["tmux", "send-keys", "-t", f"{session}:RL.1", orchestrator_log_cmd, "C-m"])

    # Logs: Inference
    inference_log_cmd = f"""while true; do
echo "Waiting for inference log file..."
while [ ! -f {output_dir}/logs/inference.stdout ]; do sleep 1; done
echo "Following inference.stdout..."
tail -F {output_dir}/logs/inference.stdout
done"""
    run(["tmux", "send-keys", "-t", f"{session}:RL.2", inference_log_cmd, "C-m"])

    # Window 2: Monitor
    run(["tmux", "new-window", "-t", session, "-n", "Monitor", "-c", str(cwd)])
    run(["tmux", "split-window", "-h", "-t", f"{session}:Monitor.0", "-c", str(cwd)])
    run(["tmux", "select-layout", "-t", f"{session}:Monitor", "even-horizontal"])

    run(["tmux", "select-pane", "-t", f"{session}:Monitor.0", "-T", "GPU"])
    run(["tmux", "send-keys", "-t", f"{session}:Monitor.0", "nvtop", "C-m"])

    run(["tmux", "select-pane", "-t", f"{session}:Monitor.1", "-T", "CPU"])
    run(["tmux", "send-keys", "-t", f"{session}:Monitor.1", "htop", "C-m"])

    # Pane title styling
    run(["tmux", "set-option", "-t", session, "-g", "pane-border-status", "top"])
    run(
        [
            "tmux",
            "set-option",
            "-t",
            session,
            "-g",
            "pane-border-format",
            " #{pane_title} ",
        ]
    )
    run(
        [
            "tmux",
            "set-window-option",
            "-t",
            f"{session}:RL",
            "pane-border-status",
            "top",
        ]
    )
    run(
        [
            "tmux",
            "set-window-option",
            "-t",
            f"{session}:Monitor",
            "pane-border-status",
            "top",
        ]
    )

    # Send command to Trainer pane and focus it
    run(["tmux", "select-window", "-t", f"{session}:RL"])
    run(["tmux", "select-pane", "-t", f"{session}:RL.0"])
    run(["tmux", "send-keys", "-t", f"{session}:RL.0", cmd, "C-m"])

    # Attach to the session if running in an interactive terminal
    if sys.stdout.isatty():
        os.execvp("tmux", ["tmux", "attach-session", "-t", session])


def main():
    parser = argparse.ArgumentParser(
        description="Create a tmux session and run prime-rl rl command from a TOML config."
    )
    parser.add_argument("at", type=str)
    parser.add_argument("config_path", type=str)
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        default="prime-rl",
        help="tmux session name (default: prime-rl)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="Output directory for logs (default: outputs)",
    )
    args = parser.parse_args()

    if not tmux_exists():
        raise SystemExit("tmux not found in PATH. Please install tmux.")

    if args.at != "@":
        raise SystemExit("Usage: prime-rl @ path/to/file.toml")

    cwd = Path.cwd()
    prime_rl_dir = cwd / "prime-rl"
    if not prime_rl_dir.exists():
        raise SystemExit(
            "Error: prime-rl directory not found. Run 'uv run vf-setup' first."
        )

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = cwd / config_path
    if not config_path.exists():
        raise SystemExit(f"TOML config not found: {config_path}")

    config_path_abs = config_path.resolve()
    config_path_rel_to_prime_rl = os.path.relpath(
        config_path_abs, prime_rl_dir.resolve()
    )

    cmd = f"uv run rl @ {config_path_rel_to_prime_rl}"

    session = find_available_session_name(args.session)

    output_dir = str((prime_rl_dir / args.output_dir).resolve())
    create_tmux_session(session, output_dir, cmd, prime_rl_dir.resolve())


if __name__ == "__main__":
    main()
