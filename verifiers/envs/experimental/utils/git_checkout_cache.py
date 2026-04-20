from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote, urlsplit, urlunsplit

from verifiers.envs.experimental.utils.file_locks import (
    exclusive_file_lock,
    exclusive_path_lock,
    sibling_lock_path,
)

DEFAULT_GIT_CHECKOUT_CACHE_ROOT = Path.home() / ".cache" / "verifiers" / "git-checkouts"
_FULL_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

logger = logging.getLogger(__name__)


def validate_git_checkout(
    path: Path,
    *,
    required_files: tuple[str, ...] = (),
) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Git checkout is not a directory: {path}")
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Git checkout is missing required files ({missing_list}): {path}"
        )
    if not (path / ".git").exists():
        raise ValueError(f"Git checkout must be a git checkout or worktree: {path}")
    return path


def _slugify_cache_component(text: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in text)
    slug = slug.strip("-")
    return slug or "repo"


def _repo_cache_dir(cache_root: Path, repo_url: str) -> Path:
    repo_name = repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    fingerprint = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    return cache_root / f"{_slugify_cache_component(repo_name)}-{fingerprint}"


def _mirror_dir(repo_cache_dir: Path) -> Path:
    return repo_cache_dir / "mirror.git"


def _worktrees_dir(repo_cache_dir: Path) -> Path:
    return repo_cache_dir / "worktrees"


def _checkout_dir_for_commit(repo_cache_dir: Path, commit_sha: str) -> Path:
    return _worktrees_dir(repo_cache_dir) / commit_sha.lower()


def _build_clone_url(repo_url: str, gh_token: str | None = None) -> str:
    if repo_url.startswith("github.com/"):
        token_prefix = f"{quote(gh_token, safe='')}@" if gh_token else ""
        return f"https://{token_prefix}{repo_url}"

    if gh_token and repo_url.startswith(("https://github.com/", "http://github.com/")):
        parsed = urlsplit(repo_url)
        if "@" not in parsed.netloc:
            return urlunsplit(
                (
                    parsed.scheme,
                    f"{quote(gh_token, safe='')}@{parsed.netloc}",
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )

    return repo_url


def _redact_clone_error_detail(detail: str, gh_token: str | None = None) -> str:
    if not gh_token:
        return detail
    redacted = detail.replace(gh_token, "<redacted>")
    quoted_token = quote(gh_token, safe="")
    if quoted_token != gh_token:
        redacted = redacted.replace(quoted_token, "<redacted>")
    return redacted


def _git_env(gh_token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if gh_token:
        env.setdefault("GH_TOKEN", gh_token)
    return env


def _git_required_error() -> RuntimeError:
    return RuntimeError("git is required to populate the local checkout cache")


def _run_git(
    args: list[str],
    *,
    gh_token: str | None = None,
    cwd: Path | None = None,
    failure: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=_git_env(gh_token),
        )
    except FileNotFoundError as exc:
        raise _git_required_error() from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        detail = _redact_clone_error_detail(detail, gh_token)
        raise RuntimeError(f"{failure}: {detail}") from exc


def _ensure_mirror(
    repo_cache_dir: Path,
    *,
    repo_url: str,
    gh_token: str | None = None,
) -> Path:
    mirror_dir = _mirror_dir(repo_cache_dir)
    clone_url = _build_clone_url(repo_url, gh_token)
    if not mirror_dir.is_dir():
        _run_git(
            ["git", "clone", "--mirror", clone_url, str(mirror_dir)],
            gh_token=gh_token,
            failure=f"Failed to clone repo mirror for {repo_url}",
        )
        return mirror_dir

    _run_git(
        ["git", "--git-dir", str(mirror_dir), "remote", "set-url", "origin", clone_url],
        gh_token=gh_token,
        failure=f"Failed to update repo mirror origin for {repo_url}",
    )
    _run_git(
        ["git", "--git-dir", str(mirror_dir), "fetch", "--prune", "--tags", "origin"],
        gh_token=gh_token,
        failure=f"Failed to fetch repo mirror for {repo_url}",
    )
    return mirror_dir


def _resolve_local_named_ref(mirror_dir: Path, ref: str) -> str | None:
    matches: set[str] = set()
    for candidate in (
        f"refs/heads/{ref}",
        f"refs/remotes/origin/{ref}",
        f"refs/tags/{ref}^{{commit}}",
    ):
        result = subprocess.run(
            ["git", "--git-dir", str(mirror_dir), "rev-parse", "--verify", candidate],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            matches.add(result.stdout.strip().lower())
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Ref {ref!r} is ambiguous in the local cache; matched {sorted(matches)}"
        )
    return next(iter(matches))


def _ensure_commit_present(
    mirror_dir: Path,
    *,
    repo_url: str,
    commit_sha: str,
    gh_token: str | None = None,
) -> str:
    existing = subprocess.run(
        [
            "git",
            "--git-dir",
            str(mirror_dir),
            "rev-parse",
            "--verify",
            f"{commit_sha}^{{commit}}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        return existing.stdout.strip().lower()

    _run_git(
        [
            "git",
            "--git-dir",
            str(mirror_dir),
            "fetch",
            "--depth",
            "1",
            "origin",
            commit_sha,
        ],
        gh_token=gh_token,
        failure=f"Failed to fetch exact commit {commit_sha!r} from {repo_url}",
    )
    fetched = _run_git(
        [
            "git",
            "--git-dir",
            str(mirror_dir),
            "rev-parse",
            "--verify",
            f"{commit_sha}^{{commit}}",
        ],
        gh_token=gh_token,
        failure=f"Failed to resolve exact commit {commit_sha!r} in local cache for {repo_url}",
    )
    return fetched.stdout.strip().lower()


def _resolve_ref_to_commit(
    mirror_dir: Path,
    *,
    repo_url: str,
    ref: str,
    gh_token: str | None = None,
) -> str:
    named_ref_commit = _resolve_local_named_ref(mirror_dir, ref)
    if named_ref_commit is not None:
        return named_ref_commit
    if not _FULL_COMMIT_SHA_RE.fullmatch(ref):
        raise RuntimeError(
            f"Ref {ref!r} was not found as a branch/tag and is not a full 40-char commit SHA"
        )
    return _ensure_commit_present(
        mirror_dir,
        repo_url=repo_url,
        commit_sha=ref.lower(),
        gh_token=gh_token,
    )


def _materialize_worktree(
    repo_cache_dir: Path,
    *,
    commit_sha: str,
    required_files: tuple[str, ...] = (),
) -> Path:
    mirror_dir = _mirror_dir(repo_cache_dir)
    checkout_dir = _checkout_dir_for_commit(repo_cache_dir, commit_sha)
    try:
        return validate_git_checkout(checkout_dir, required_files=required_files)
    except ValueError:
        if checkout_dir.exists():
            shutil.rmtree(checkout_dir, ignore_errors=True)

    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        ["git", "--git-dir", str(mirror_dir), "worktree", "prune"],
        failure=f"Failed to prune stale worktree metadata for {mirror_dir}",
    )
    _run_git(
        [
            "git",
            "--git-dir",
            str(mirror_dir),
            "worktree",
            "add",
            "--detach",
            str(checkout_dir),
            commit_sha,
        ],
        failure=f"Failed to materialize checkout for commit {commit_sha}",
    )
    return validate_git_checkout(checkout_dir, required_files=required_files)


def _prune_stale_worktrees(repo_cache_dir: Path, keep_checkout: Path) -> None:
    mirror_dir = _mirror_dir(repo_cache_dir)
    worktrees_dir = _worktrees_dir(repo_cache_dir)
    if not worktrees_dir.is_dir():
        return
    for candidate in worktrees_dir.iterdir():
        if not candidate.is_dir() or candidate == keep_checkout:
            continue
        try:
            with exclusive_path_lock(
                candidate,
                suffix=".upload.lock",
                nonblocking=True,
            ):
                _run_git(
                    [
                        "git",
                        "--git-dir",
                        str(mirror_dir),
                        "worktree",
                        "remove",
                        "--force",
                        str(candidate),
                    ],
                    failure=f"Failed to remove stale checkout {candidate}",
                )
        except BlockingIOError:
            continue
        except RuntimeError as exc:
            logger.warning(str(exc))
            continue
        sibling_lock_path(candidate, ".upload.lock").unlink(missing_ok=True)
    try:
        _run_git(
            ["git", "--git-dir", str(mirror_dir), "worktree", "prune"],
            failure=f"Failed to prune stale worktree metadata for {mirror_dir}",
        )
    except RuntimeError as exc:
        logger.warning(str(exc))


def resolve_git_checkout(
    *,
    repo_url: str,
    ref: str,
    cache_root: Path = DEFAULT_GIT_CHECKOUT_CACHE_ROOT,
    local_checkout: str | Path | None = None,
    gh_token: str | None = None,
    required_files: tuple[str, ...] = (),
) -> Path:
    if local_checkout is not None:
        return validate_git_checkout(
            Path(local_checkout), required_files=required_files
        )

    repo_cache_dir = _repo_cache_dir(cache_root.expanduser().resolve(), repo_url)
    repo_cache_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(repo_cache_dir / ".repo.lock"):
        mirror_dir = _ensure_mirror(
            repo_cache_dir,
            repo_url=repo_url,
            gh_token=gh_token,
        )
        resolved_commit = _resolve_ref_to_commit(
            mirror_dir,
            repo_url=repo_url,
            ref=ref,
            gh_token=gh_token,
        )
        checkout = _materialize_worktree(
            repo_cache_dir,
            commit_sha=resolved_commit,
            required_files=required_files,
        )
        _prune_stale_worktrees(repo_cache_dir, checkout)
        return checkout
