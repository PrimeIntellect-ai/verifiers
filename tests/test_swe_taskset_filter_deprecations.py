import pytest

from verifiers.envs.experimental.composable._filter import _resolve_filter_fn
from verifiers.envs.experimental.composable.tasksets.swe._filters import (
    combine_filter_fns,
    deprecated_filter_repos_filter_fn,
    deprecated_swerebench_language_filter_fn,
)


def test_deprecated_filter_repos_filter_fn_matches_repo_aliases():
    with pytest.warns(DeprecationWarning, match="filter_repos is deprecated"):
        expr = deprecated_filter_repos_filter_fn(["owner/repo"])

    fn = _resolve_filter_fn(expr)

    assert not fn({"info": {"repo": "owner/repo"}})
    assert not fn({"info": {"repo_name": "owner/repo"}})
    assert fn({"info": {"repo": "other/repo", "repo_name": "other/repo"}})


def test_deprecated_swerebench_language_filter_fn_matches_language():
    with pytest.warns(DeprecationWarning, match="language=.* is deprecated"):
        expr = deprecated_swerebench_language_filter_fn("python")

    fn = _resolve_filter_fn(expr)

    assert fn({"info": {"language": "python"}})
    assert not fn({"info": {"language": "go"}})


def test_combine_filter_fns_requires_all_predicates():
    expr = combine_filter_fns(
        "lambda x: x['info'].get('language') == 'python'",
        "lambda x: x['info'].get('repo') != 'blocked/repo'",
    )
    fn = _resolve_filter_fn(expr)

    assert fn({"info": {"language": "python", "repo": "allowed/repo"}})
    assert not fn({"info": {"language": "python", "repo": "blocked/repo"}})
    assert not fn({"info": {"language": "go", "repo": "allowed/repo"}})
