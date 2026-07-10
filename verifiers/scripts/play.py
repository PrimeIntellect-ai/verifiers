import argparse
import asyncio
import json
import sys
import tomllib
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.table import Table

import verifiers as vf
from verifiers.cli.interactive import HumanClient, InteractiveSessionExit
from verifiers.scripts.eval import DEFAULT_ENV_DIR_PATH
from verifiers.types import RolloutInput, SamplingArgs, State
from verifiers.utils.env_utils import env_module_name
from verifiers.utils.save_utils import make_serializable, state_to_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options] [env_id]",
        description="Play through one Verifiers rollout interactively as the model.",
        epilog=(
            "env_id is the environment module name to load. It is inferred from "
            "the current environment directory when omitted."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "env_id",
        nargs="?",
        default=None,
        help=(
            "Environment module name to load. Inferred from the current "
            "environment directory when omitted."
        ),
    )
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default=None,
        help='Environment loader arguments as JSON, e.g. \'{"mode": "text"}\'.',
    )
    parser.add_argument(
        "--env-dir-path",
        "-p",
        default=DEFAULT_ENV_DIR_PATH,
        help="Directory containing local environment packages.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval"],
        default="eval",
        help="Dataset split to draw the rollout input from.",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        default=0,
        help="Dataset row index to play.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the selected split before applying --index.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Seed for --shuffle. Defaults to 0 when --shuffle is set.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="human",
        help="Model name recorded in the rollout response metadata.",
    )
    parser.add_argument(
        "--sampling-args",
        "-S",
        type=json.loads,
        default=None,
        help="Sampling arguments recorded in rollout state as JSON.",
    )
    parser.add_argument(
        "--save",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save the played rollout as JSON.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip environment reward/metric scoring after the rollout.",
    )
    parser.add_argument(
        "--hide-prompt",
        action="store_true",
        help="Hide the model-visible prompt pane.",
    )
    parser.add_argument(
        "--hide-tools",
        action="store_true",
        help="Hide the tool schema pane.",
    )
    parser.add_argument(
        "--allow-external-images",
        action="store_true",
        help=(
            "Load image_url parts that reference external resources (http(s), "
            "file://, or local paths). Off by default; only self-contained "
            "data: images render, since environment-supplied URLs can trigger "
            "SSRF or read arbitrary local files."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def prepare_local_env_import(env_id: str, env_dir_path: str) -> None:
    """Make a local uninstalled environment importable when present."""
    env_dir = Path(env_dir_path).expanduser().resolve()
    module_name = env_module_name(env_id)
    candidates = [
        env_dir / module_name,
        env_dir / module_name.replace("_", "-"),
        env_dir,
    ]
    existing = [path for path in candidates if path.exists()]
    for path in reversed(existing):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def infer_current_env_id(cwd: Path | None = None) -> str | None:
    """Infer an environment id from the nearest project metadata."""
    start = (cwd or Path.cwd()).resolve()
    for directory in [start, *start.parents]:
        pyproject_path = directory / "pyproject.toml"
        if not pyproject_path.is_file():
            continue
        with pyproject_path.open("rb") as file:
            pyproject = tomllib.load(file)
        project = pyproject.get("project")
        if not isinstance(project, dict):
            continue
        # The verifiers library itself defines the ``vf-play`` entry point;
        # it is never a playable environment, so keep walking up (or fall back
        # to requiring an explicit env_id) instead of trying to load it.
        scripts = project.get("scripts")
        if isinstance(scripts, dict) and "vf-play" in scripts:
            continue
        name = project.get("name")
        if isinstance(name, str) and name:
            return name
        # Nameless/incomplete metadata: keep walking up rather than giving up.
    return None


def resolve_env_dir_path(env_dir_path: str, cwd: Path | None = None) -> str:
    """Resolve the default environment directory from inside nested env packages."""
    path = Path(env_dir_path).expanduser()
    if path.is_absolute() or env_dir_path != DEFAULT_ENV_DIR_PATH:
        return str(path)

    start = (cwd or Path.cwd()).resolve()
    direct = start / path
    if direct.is_dir():
        return str(direct)

    for directory in [start, *start.parents]:
        if directory.name == "environments" and directory.is_dir():
            return str(directory)
        candidate = directory / "environments"
        if candidate.is_dir():
            return str(candidate)
    return str(path)


def resolve_env_id(args: argparse.Namespace) -> str:
    env_id = args.env_id
    if isinstance(env_id, str) and env_id:
        return env_id
    inferred = infer_current_env_id()
    if inferred is not None:
        return inferred
    raise ValueError(
        "env_id is required unless vf-play is run from an environment directory."
    )


def select_rollout_input(
    env: vf.Environment,
    *,
    split: str,
    index: int,
    shuffle: bool,
    shuffle_seed: int | None,
) -> RolloutInput:
    seed = (0 if shuffle_seed is None else shuffle_seed) if shuffle else None
    dataset = (
        env.get_eval_dataset(seed=seed)
        if split == "eval"
        else env.get_dataset(seed=seed)
    )
    if index < 0 or index >= len(dataset):
        raise IndexError(
            f"--index {index} is out of range for {split} split with {len(dataset)} rows."
        )
    return cast(RolloutInput, dict(dataset[index]))


async def run_interactive_rollout(
    args: argparse.Namespace,
    *,
    console: Console | None = None,
) -> State:
    console = console or Console()
    if args.env_args is not None and not isinstance(args.env_args, dict):
        raise ValueError("--env-args must be a JSON object.")
    env_id = resolve_env_id(args)
    env_dir_path = resolve_env_dir_path(args.env_dir_path)
    prepare_local_env_import(env_id, env_dir_path)
    env_args = dict(args.env_args or {})
    env = vf.load_environment(env_id, **env_args)
    env.set_score_rollouts(not args.no_score)
    rollout_input = select_rollout_input(
        env,
        split=args.split,
        index=args.index,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
    )
    answer = rollout_input.get("answer")
    client = HumanClient(
        show_prompt=not args.hide_prompt,
        show_tools=not args.hide_tools,
        answer=str(answer) if answer is not None and str(answer) else None,
        allow_external_images=args.allow_external_images,
    )

    try:
        console.print(
            f"Playing [bold]{env_id}[/bold] "
            f"({args.split} row {args.index}) as model [bold]{args.model}[/bold]."
        )
        await client.start()
        state = await env._run_rollout_state(
            rollout_input,
            client,
            args.model,
            cast(SamplingArgs, args.sampling_args or {}),
        )
    finally:
        await client.close()
    return state


def render_summary(state: State, console: Console) -> None:
    table = Table(title="Rollout Summary")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("completed", str(state.get("is_completed", False)))
    table.add_row("truncated", str(state.get("is_truncated", False)))
    table.add_row("stop_condition", str(state.get("stop_condition")))
    table.add_row("reward", str(state.get("reward")))
    metrics = state.get("metrics") or {}
    if metrics:
        table.add_row(
            "metrics", json.dumps(metrics, indent=2, default=make_serializable)
        )
    console.print()
    console.print(table)


def save_state(state: State, path: Path) -> None:
    output = state_to_output(state, state_columns=["trajectory"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, default=make_serializable),
        encoding="utf-8",
    )


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    try:
        state = await run_interactive_rollout(args, console=console)
    except (ValueError, IndexError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except InteractiveSessionExit:
        console.print("\n[yellow]Interactive rollout aborted.[/yellow]")
        return 130
    if isinstance(state.get("error"), InteractiveSessionExit):
        console.print("\n[yellow]Interactive rollout aborted.[/yellow]")
        return 130
    render_summary(state, console)
    if args.save is not None:
        save_state(state, args.save)
        console.print(f"Saved rollout to [bold]{args.save}[/bold]")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
