"""scaleswe-v1 — Scale-SWE (AweAI-Team/Scale-SWE) as a v1 taskset.

Each row ships a per-task Docker image with the repo checked out, `pre_commands` that
reset it to the base commit on a clean `scaleswe` branch, F2P/P2P pytest ids, and an
optional `f2p_patch` / `f2p_script` carrying the failing test. `setup` runs the row's
`pre_commands` in the live runtime before the agent (the runtime's workdir is the row's
repo). The `solved` reward restores the test files to base (the agent only fixes the
source), applies the f2p test, then runs the merged F2P+P2P ids through `score.py` —
which scores 1.0 iff every expected id passed. A v1 port of the v0 ComposableEnv
`ScaleSWETaskSet`.
"""

import json
import logging
from pathlib import Path

import verifiers.v1 as vf

logger = logging.getLogger(__name__)

DATASET = "AweAI-Team/Scale-SWE"

# Prime's private Artifact Registry holds the complete image set, but only runtimes with GCP
# credentials (e.g. Prime sandboxes) can pull from it. By default images are taken straight from
# the dataset's `image_url` (the public `aweaiteam/scaleswe` Docker Hub mirror, missing some
# tags); set `use_prime_registry` to resolve against this registry instead.
REGISTRY = "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"

# The testbed conda env (with the project + pytest installed) and quiet, non-interactive
# tooling — exported for every command the taskset runs in the sandbox.
ENV = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "CI": "1",
}

# Restore the test files to the base commit so the agent's edits to the source are scored
# against the original tests (and tests it added are dropped). `$base` comes from the env.
RESTORE = r"""
git checkout "$base" -- tests/ test/ Test/ Tests/ 2>/dev/null || true
git ls-tree -r --name-only "$base" 2>/dev/null | while IFS= read -r path; do
  case "$path" in
    test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
      git checkout "$base" -- "$path" 2>/dev/null || true ;;
  esac
done
git ls-files 2>/dev/null | while IFS= read -r path; do
  case "$path" in
    tests/*|test/*|Test/*|Tests/*|test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
      if ! git cat-file -e "$base:$path" 2>/dev/null; then
        rm -f -- "$path"; git rm -q --cached -- "$path" 2>/dev/null || true
      fi ;;
  esac
done
"""

PATCH = "/tmp/scaleswe_f2p.patch"
SCORER = "/tmp/scaleswe_scorer.py"
SCORER_SRC = (Path(__file__).parent / "score.py").read_bytes()


def _ids(raw: str | list[str] | None) -> list[str]:
    """F2P/P2P ids arrive as a list or a JSON-encoded string; normalize to list[str]."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(t).strip() for t in parsed if t]
    return [parsed] if isinstance(parsed, str) and parsed else []


def _parse_ref(ref: str) -> tuple[str, str, str]:
    """Split an image reference into (registry host, repository, tag)."""
    head, _, rest = ref.partition("/")
    if rest and ("." in head or ":" in head or head == "localhost"):
        registry, remainder = head, rest
    else:
        registry, remainder = "registry-1.docker.io", ref
    repo, _, tag = remainder.rpartition(":")
    return registry, repo, tag


def _list_tags(registry: str, repo: str) -> set[str] | None:
    """All tags in `registry/repo` via the Docker registry v2 API, or None if the registry
    can't be enumerated anonymously (private/no creds/unreachable)."""
    import urllib.error
    import urllib.request

    def _get(url: str, token: str | None):
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        return urllib.request.urlopen(req, timeout=30)

    def _token(challenge: str | None) -> str | None:
        if not challenge or not challenge.lower().startswith("bearer "):
            return None
        parts = dict(
            p.strip().split("=", 1) for p in challenge[7:].split(",") if "=" in p
        )
        realm = parts.get("realm", "").strip('"')
        params = "&".join(
            f"{k}={v.strip(chr(34))}" for k, v in parts.items() if k != "realm"
        )
        try:
            return json.load(_get(f"{realm}?{params}", None)).get("token")
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
            return None

    tags: list[str] = []
    url: str | None = f"https://{registry}/v2/{repo}/tags/list?n=1000"
    token = None
    while url:
        try:
            resp = _get(url, token)
        except urllib.error.HTTPError as e:
            if e.code == 401 and token is None:
                token = _token(e.headers.get("WWW-Authenticate"))
                if token:
                    continue
            return None
        except urllib.error.URLError:
            return None
        tags.extend(json.load(resp).get("tags") or [])
        link = resp.headers.get("Link", "")
        nxt = link[link.find("<") + 1 : link.find(">")] if 'rel="next"' in link else ""
        url = f"https://{registry}{nxt}" if nxt.startswith("/") else (nxt or None)
    return set(tags)


def _available_images(images: set[str]) -> set[str]:
    """The subset of `images` whose tag exists in its registry. Images in a registry that
    can't be enumerated (private, no creds, unreachable) are kept rather than dropped."""
    by_repo: dict[tuple[str, str], set[str]] = {}
    for ref in images:
        registry, repo, _ = _parse_ref(ref)
        by_repo.setdefault((registry, repo), set()).add(ref)
    available: set[str] = set()
    for (registry, repo), refs in by_repo.items():
        tags = _list_tags(registry, repo)
        if tags is None:
            logger.warning(
                "scaleswe: could not enumerate %s/%s - keeping %d images unchecked",
                registry,
                repo,
                len(refs),
            )
            available |= refs
        else:
            available |= {r for r in refs if _parse_ref(r)[2] in tags}
    return available


class ScaleSWETask(vf.Task):
    base_commit: str
    """Commit the repo is reset to before the agent runs and tests are scored against."""
    pre_commands: str
    """Shell run in the repo before the agent — resets to `base_commit` on branch `scaleswe`."""
    f2p_patch: str = ""
    """Optional patch adding the fail-to-pass test (applied before scoring)."""
    f2p_script: str = ""
    """Optional test file uploaded as `test_fail_to_pass.py` before scoring."""
    patch: str = ""
    """The gold source patch (reference solution); `validate` applies it to confirm the task
    is solvable. Not used during rollouts."""
    fail_to_pass: list[str] = []
    pass_to_pass: list[str] = []


class ScaleSWEConfig(vf.TasksetConfig):
    use_prime_registry: bool = False
    """Resolve task images against Prime's private Artifact Registry (`REGISTRY`) instead of the
    dataset's public Docker Hub `image_url`. Only works on runtimes with GCP pull credentials."""
    filter_unavailable_images: bool = True
    """Drop tasks whose image isn't present in the registry (a quick tags/list check at load),
    so rollouts don't fail at container start. Registries that can't be enumerated are kept."""


class ScaleSWETaskset(vf.Taskset[ScaleSWETask, ScaleSWEConfig]):
    NEEDS_CONTAINER = True

    def load_tasks(self) -> list[ScaleSWETask]:
        from datasets import load_dataset

        rows = load_dataset(DATASET, split="train")
        tasks = [
            ScaleSWETask(
                idx=i,
                name=row["instance_id"],
                instruction=row["problem_statement"],
                image=(
                    f"{REGISTRY}/{row['image_url']}"
                    if self.config.use_prime_registry
                    else row["image_url"]
                ),
                workdir=row["workdir"],
                resources=vf.Resources(cpu=4, memory=4, disk=10),
                base_commit=row.get("parent_commit") or row.get("base_commit") or "",
                pre_commands=(row.get("pre_commands") or "")
                .strip()
                .removesuffix("\\n"),
                f2p_patch=row.get("f2p_patch") or "",
                f2p_script=row.get("f2p_script") or "",
                patch=row.get("patch") or "",
                fail_to_pass=_ids(row.get("FAIL_TO_PASS")),
                pass_to_pass=_ids(row.get("PASS_TO_PASS")),
            )
            for i, row in enumerate(rows)
        ]
        if self.config.filter_unavailable_images:
            available = _available_images({t.image for t in tasks})
            kept = [t for t in tasks if t.image in available]
            if len(kept) < len(tasks):
                logger.info(
                    "scaleswe: dropped %d/%d tasks with unavailable images",
                    len(tasks) - len(kept),
                    len(tasks),
                )
            tasks = kept
        return tasks

    async def setup(self, task: ScaleSWETask, runtime: vf.Runtime) -> None:
        result = await runtime.run(["sh", "-c", task.pre_commands], ENV)
        if result.exit_code != 0:
            raise vf.ProgramError(
                f"scaleswe setup failed ({task.name}): {result.stderr.strip()[-500:]}"
            )

    @vf.reward(weight=1.0)
    async def solved(self, task: ScaleSWETask, runtime: vf.Runtime) -> float:
        test_ids = task.fail_to_pass + task.pass_to_pass
        if not test_ids:
            return 0.0
        await runtime.run(["sh", "-c", RESTORE], {**ENV, "base": task.base_commit})
        if task.f2p_patch.strip():
            await self._apply_patch(runtime, task.f2p_patch)
        if task.f2p_script:
            await runtime.write("test_fail_to_pass.py", task.f2p_script.encode())
        await runtime.write(SCORER, SCORER_SRC)
        result = await runtime.run(["python", SCORER, json.dumps(test_ids)], ENV)
        lines = result.stdout.strip().splitlines()
        return float(lines[-1]) if lines else 0.0

    async def validate(self, task: ScaleSWETask, runtime: vf.Runtime) -> bool:
        """Valid iff the gold solution scores 1.0: apply the reference source patch, then run
        the same `solved` reward the agent is graded by (setup has reset the repo to base)."""
        await self.apply_gold_patch(task, runtime)
        return await self.solved(task, runtime) == 1.0

    async def apply_gold_patch(self, task: ScaleSWETask, runtime: vf.Runtime) -> None:
        """Apply the gold source patch (for validation), raising if no strategy takes."""
        if not task.patch.strip():
            raise vf.ProgramError(f"empty gold patch for {task.name}")
        if not await self._apply_patch(runtime, task.patch):
            raise vf.ProgramError(f"gold apply failed ({task.name})")

    async def _apply_patch(self, runtime: vf.Runtime, patch: str) -> bool:
        # Try strict, then whitespace-tolerant, then a fuzzy `patch`, then a partial
        # `--reject` apply (mirrors the v0 multi-strategy helper). Returns whether a strategy
        # applied cleanly: `solved` applies the test patch best-effort (scoring catches a
        # patch that didn't take), `apply_gold_patch` requires a clean apply.
        await runtime.write(PATCH, patch.encode())
        for cmd in (
            f"git apply --verbose {PATCH}",
            f"git apply --verbose --ignore-space-change --ignore-whitespace {PATCH}",
            f"patch --batch --fuzz=5 -p1 -i {PATCH}",
            f"git apply --verbose --reject --ignore-whitespace {PATCH}",
        ):
            if (await runtime.run(["sh", "-c", cmd], ENV)).exit_code == 0:
                return True
        return False


def load_taskset(config: ScaleSWEConfig) -> ScaleSWETaskset:
    return ScaleSWETaskset(config)
