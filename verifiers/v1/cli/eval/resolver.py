"""Side-effect-free resolution of eval argv into a typed invocation."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic_config import cli

from verifiers.v1.cli.eval.resume import load_resume_config, split_resume
from verifiers.v1.cli.output import load_manifest, output_path
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.eval import EvalConfig


@dataclass(frozen=True)
class EvalInvocation:
    """The resolved config and run location for one eval invocation."""

    config: EvalConfig
    run_id: str
    output_dir: Path
    resume: bool = False


def resolve_eval(argv: Sequence[str], *, prog: str = "eval run") -> EvalInvocation:
    """Resolve eval arguments without reading or mutating ``sys.argv``.

    Config files, taskset/harness narrowing, aliases, defaults, and validation all flow
    through ``prime-pydantic-config``'s explicit ``args`` and ``prog`` interface. Help is a
    presentation operation handled by the entrypoint rather than this resolver.
    """
    args = with_positional_taskset(list(argv))
    if any(arg in ("-h", "--help") for arg in args):
        raise ValueError("resolve_eval does not render help; use `eval run --help`")

    resume_dir, rest = split_resume(args)
    if resume_dir is not None:
        if rest:
            raise ValueError(
                "--resume re-runs a saved config and takes no other arguments"
            )
        manifest = load_manifest(resume_dir)
        config = load_resume_config(resume_dir)
        run_id = manifest["run_id"] if manifest else resume_dir.resolve().name
        config.uuid = run_id
        return EvalInvocation(
            config=config,
            run_id=run_id,
            output_dir=resume_dir,
            resume=True,
        )

    legacy_id = any(arg == "--id" or arg.startswith("--id=") for arg in args)
    if (
        not extract_id(args, "taskset")
        and not legacy_id
        and not references_config_file(args)
    ):
        raise ValueError("eval needs a taskset id, a legacy --id, or an @ config file")

    config_type = narrow_config(EvalConfig, args)
    # `prog` is supported by cli(), but is missing from its overload declarations.
    config = cli(config_type, args=args, prog=prog)  # type: ignore[no-matching-overload]
    return EvalInvocation(
        config=config,
        run_id=config.uuid,
        output_dir=output_path(config),
    )
