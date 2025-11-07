import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # type: ignore[unresolved-import]
except ImportError:
    import tomli as tomllib  # type: ignore[unresolved-import]


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


def ensure_no_session(session: str) -> None:
    # kill any existing session with same name (fail fast)
    proc = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode == 0:
        # session exists -> kill it
        run(["tmux", "kill-session", "-t", session])


def create_tmux_with_commands(
    session: str, cmd_top: str, cmd_bottom: str, cwd: Path | None
) -> None:
    # new session detached
    new_session_cmd = ["tmux", "new-session", "-d", "-s", session]
    if cwd is not None:
        new_session_cmd += ["-c", str(cwd)]
    new_session_cmd += ["bash"]
    run(new_session_cmd)

    # Split horizontally (top/bottom)
    split_cmd = ["tmux", "split-window", "-v"]
    if cwd is not None:
        split_cmd += ["-c", str(cwd)]
    run(split_cmd)

    # Pane titles
    run(["tmux", "select-pane", "-t", f"{session}:0.0", "-T", "Inference"])
    run(["tmux", "select-pane", "-t", f"{session}:0.1", "-T", "Trainer"])

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
            f"{session}:0",
            "pane-border-status",
            "top",
        ]
    )

    # Send commands to top (pane 0) and bottom (pane 1)
    run(["tmux", "select-pane", "-t", f"{session}:0.0"])  # focus top
    run(["tmux", "send-keys", "-t", f"{session}:0.0", cmd_top, "C-m"])  # enter

    run(["tmux", "select-pane", "-t", f"{session}:0.1"])  # bottom
    run(["tmux", "send-keys", "-t", f"{session}:0.1", cmd_bottom, "C-m"])  # enter

    # Attach to the session if running in an interactive terminal
    if sys.stdout.isatty():
        os.execvp("tmux", ["tmux", "attach-session", "-t", session])


def load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def to_kebab_case(name: str) -> str:
    return name.replace("_", "-")


def build_vllm_command(model: str, inference_cfg: dict, inference_gpu_str: str) -> str:
    gpus = inference_cfg.get("gpus")
    args = inference_cfg.get("args", {}) or {}

    parts: list[str] = [inference_gpu_str, "uv run", "vf-vllm", "--model", str(model)]

    for key, value in args.items():
        flag = "--" + to_kebab_case(str(key))
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.extend([flag, str(value)])

    if isinstance(gpus, int) and gpus > 1:
        # divide by tensor_parallel_size if provided
        tensor_parallel_size = args.get("tensor_parallel_size")
        if tensor_parallel_size:
            gpus = gpus // tensor_parallel_size
        parts.extend(["--data-parallel-size", str(gpus)])

    return " ".join(parts)


def build_train_command(env_id: str, config_path_str: str, trainer_gpu_str: str) -> str:
    parts: list[str] = []
    if "/" in env_id:
        parts.extend(["prime env install", env_id, "&&"])
    else:
        parts.extend(["uv run", "vf-install", env_id, "&&"])

    parts.extend([trainer_gpu_str, "uv run", "vf-train", "@", str(config_path_str)])

    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a tmux session and run vf-vllm (top) and vf-train (bottom) from a TOML config."
        )
    )
    parser.add_argument("at", type=str)
    parser.add_argument("config_path", type=str)
    parser.add_argument(
        "--session",
        "-s",
        type=str,
        default="vf-rl",
        help="tmux session name (default: from TOML or filename)",
    )
    parser.add_argument(
        "--cwd", type=str, default=None, help="Working directory to run commands in"
    )
    args = parser.parse_args()

    if not tmux_exists():
        raise SystemExit("tmux not found in PATH. Please install tmux.")

    if args.at != "@":
        raise SystemExit("Usage: vf-rl @ path/to/file.toml")

    config_path_str = args.config_path
    config_path = Path(config_path_str)
    if not config_path.exists():
        raise SystemExit(f"TOML config not found: {config_path}")

    data = load_toml(config_path)
    session = args.session
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise SystemExit("Missing required 'model' at top level in TOML.")
    env_id = data.get("env", {}).get("id")

    num_inference_gpus: int = data.get("inference", {}).get("gpus")
    if not isinstance(num_inference_gpus, int) or num_inference_gpus <= 0:
        raise SystemExit("Missing required 'inference.gpus' at top level in TOML.")
    num_trainer_gpus: int = data.get("trainer", {}).get("gpus")
    if not isinstance(num_trainer_gpus, int) or num_trainer_gpus <= 0:
        raise SystemExit("Missing required 'trainer.gpus' at top level in TOML.")
    inference_gpu_str = "CUDA_VISIBLE_DEVICES=" + ",".join(
        str(i) for i in range(num_inference_gpus)
    )
    trainer_gpu_str = "CUDA_VISIBLE_DEVICES=" + ",".join(
        str(i) for i in range(num_inference_gpus, num_inference_gpus + num_trainer_gpus)
    )
    inference_cfg = data.get("inference", {}) or {}
    cmd_top = build_vllm_command(model, inference_cfg, inference_gpu_str)
    cmd_bottom = build_train_command(env_id, config_path_str, trainer_gpu_str)
    cwd = Path(args.cwd).resolve() if args.cwd else None
    ensure_no_session(session)
    create_tmux_with_commands(session, cmd_top, cmd_bottom, cwd)


if __name__ == "__main__":
    main()
