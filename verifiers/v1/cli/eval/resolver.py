"""Side-effect-free resolution of eval argv into a typed config."""

from pydantic_config import cli

from verifiers.v1.cli.eval.resume import load_resume_config, split_resume
from verifiers.v1.cli.resolve import (
    extract_id,
    narrow_config,
    references_config_file,
    with_positional_taskset,
)
from verifiers.v1.configs.eval import EvalConfig


def resolve_eval(argv: list[str], *, prog: str = "eval run") -> EvalConfig:
    """Resolve eval arguments without reading or mutating ``sys.argv``."""
    args = with_positional_taskset(list(argv))
    if any(arg in ("-h", "--help") for arg in args):
        raise ValueError("resolve_eval does not render help; use `eval run --help`")

    resume_dir, rest = split_resume(args)
    if resume_dir is not None and rest:
        raise ValueError("--resume takes no other arguments")
    if resume_dir is not None:
        config = load_resume_config(resume_dir)
        config.uuid = resume_dir.resolve().name
        return config

    if (
        not extract_id(args, "taskset")
        and not any(arg == "--id" or arg.startswith("--id=") for arg in args)
        and not references_config_file(args)
    ):
        raise ValueError("eval needs a taskset id, a legacy --id, or an @ config file")

    # `prog` is supported by cli(), but is missing from its overload declarations.
    return cli(  # type: ignore[no-matching-overload]
        narrow_config(EvalConfig, args), args=args, prog=prog
    )
