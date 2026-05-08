from __future__ import annotations

import warnings


def combine_filter_fns(*filter_fns: str | None) -> str | None:
    """Compose ``filter_fn`` expression strings into one post-processed filter."""
    active = [filter_fn for filter_fn in filter_fns if filter_fn is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    return "lambda x: " + " and ".join(f"({filter_fn})(x)" for filter_fn in active)


def deprecated_filter_repos_filter_fn(filter_repos: list[str] | None) -> str | None:
    if not filter_repos:
        return None
    warnings.warn(
        "filter_repos is deprecated; use filter_fn with a predicate over "
        "x['info']['repo'] instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    repos = tuple(filter_repos)
    return (
        "lambda x: "
        f"x['info'].get('repo') not in {repos!r} and "
        f"x['info'].get('repo_name') not in {repos!r}"
    )


def deprecated_swerebench_language_filter_fn(language: str | None) -> str | None:
    if language is None:
        return None
    warnings.warn(
        "SWERebenchV2TaskSet(language=...) is deprecated; use filter_fn with "
        "a predicate over x['info']['language'] instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return f"lambda x: x['info'].get('language') == {language!r}"
