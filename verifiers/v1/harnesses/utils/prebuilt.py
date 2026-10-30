"""Harness installs built once on Prime and mounted read-only into later rollouts.

A split deployment would install its harness on the harness's VM and its tools' worker
in the workspace, every rollout. Instead, one Prime VM runs the harness's own install,
copies the install's directories into a volume, and publishes it; later deployments
mount the volume read-only on both VMs, where the directories link into it. Rollouts
install as usual until the volume is published, and wherever the linked install does
not run, such as on a musl image.
"""

import asyncio
import fcntl
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

from verifiers.v1.configs.runtime import PrimeConfig, PrimeVolume
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime, provision_backend
from verifiers.v1.runtimes.backends.prime import leased_client
from verifiers.v1.utils.paths import CACHE_DIR

logger = logging.getLogger(__name__)

ROOT = "/opt/vf-prebuilt"
_COMPLETE = ".complete"
_CAPTURE = """
set -e
mount=$1; shift; i=0
for dir in "$@"; do mkdir -p "$mount/$i"; cp -a "$dir/." "$mount/$i/"; i=$((i + 1)); done
chmod -R a+rX "$mount"
"""
_LINK = f"""
mount=$1; shift; i=0
[ -f "$mount/{_COMPLETE}" ] || exit 2
for dir in "$@"; do if [ -e "$dir" ] || [ -L "$dir" ]; then exit 3; fi; done
for dir in "$@"; do mkdir -p "${{dir%/*}}"; ln -s "$mount/$i" "$dir"; i=$((i + 1)); done
"""
_UNLINK = 'for dir in "$@"; do if [ -L "$dir" ]; then rm "$dir"; fi; done'
_RETRY_AFTER = 600
"""Seconds before a process retries a failed build."""
_RECHECK_AFTER = 30
"""Seconds a process waits before asking again whether a volume is published."""


@dataclass(frozen=True)
class Prebuilt:
    """A harness install that one Prime VM builds and publishes as a volume, for split
    deployments to mount instead of installing per rollout."""

    name: str
    """The volume, named by everything that shapes the install (`prebuilt_name`)."""
    dirs: tuple[str, ...]
    """The install's directories, captured whole and linked back at the same paths."""
    probe: str
    """A shell command that succeeds where the linked install runs."""
    install: Callable[[Runtime], Awaitable[None]]
    """The harness's own install, which the builder runs."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Builder environment that keeps the install inside `dirs`, such as a managed
    Python instead of the builder image's."""

    @property
    def mount(self) -> str:
        return f"{ROOT}/{self.name}"


def prebuilt_name(harness: str, *inputs: str) -> str:
    digest = hashlib.sha256("\0".join(inputs).encode()).hexdigest()
    return f"vf-{harness}-{digest[:16]}"


_published: set[str] = set()
_checked: dict[str, float] = {}
_builds: dict[str, asyncio.Task[None]] = {}
_failed: dict[str, float] = {}


async def published(prebuilt: Prebuilt | None) -> bool:
    """Whether `prebuilt` is published; if not, start building it in the background, so
    that later rollouts of this process can mount it."""
    if prebuilt is None:
        return False
    if prebuilt.name in _published:
        return True
    checked = _checked.get(prebuilt.name)
    if checked is not None and time.monotonic() - checked < _RECHECK_AFTER:
        return False
    _checked[prebuilt.name] = time.monotonic()
    if await _has_head(prebuilt.name):
        _published.add(prebuilt.name)
        return True
    failed = _failed.get(prebuilt.name)
    running = prebuilt.name in _builds and not _builds[prebuilt.name].done()
    if not running and (failed is None or time.monotonic() - failed > _RETRY_AFTER):
        _builds[prebuilt.name] = asyncio.create_task(_build_logged(prebuilt))
    return False


async def build(prebuilt: Prebuilt) -> None:
    """Build and publish `prebuilt` unless it is published. A builder elsewhere on this
    host holds a lock that this waits for; builders on other hosts may duplicate the
    work, and the first publish wins."""
    locks = CACHE_DIR / "prebuilt"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / prebuilt.name).open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(1)
        if await _has_head(prebuilt.name):
            _published.add(prebuilt.name)
            return
        logger.info("prebuilt: building %s", prebuilt.name)
        config = PrimeConfig(
            cpu=2,
            memory=4,
            disk=10,
            labels=["vf-prebuilt"],
            mounts={prebuilt.mount: PrimeVolume(name=prebuilt.name, read_only=False)},
        )
        async with (
            asyncio.timeout(1800),
            provision_backend(config, f"prebuilt-{prebuilt.name}") as backend,
        ):
            runtime = backend.bind()
            await prebuilt.install(runtime.with_env(prebuilt.env))
            await _check(runtime, [_CAPTURE, prebuilt.mount, *prebuilt.dirs], "capture")
            # The volume must run on its own: set the originals aside and link it.
            await _check(
                runtime,
                ['for dir in "$@"; do mv "$dir" "$dir.built"; done', *prebuilt.dirs],
                "set aside",
            )
            await _check(runtime, [f": > {prebuilt.mount}/{_COMPLETE}"], "mark")
            await _check(runtime, [_LINK, prebuilt.mount, *prebuilt.dirs], "link")
            await _check(runtime, [prebuilt.probe], "probe")
            try:
                async with leased_client() as client:
                    await client.volumes.publish(backend.info.id, prebuilt.name)
            except Exception:
                # A builder on another host may have published first.
                if not await _has_head(prebuilt.name):
                    raise
        _published.add(prebuilt.name)
        logger.info("prebuilt: published %s", prebuilt.name)


async def link(runtime: Runtime, prebuilt: Prebuilt | None) -> bool:
    """Link the install's directories into `prebuilt`'s volume where the runtime mounts
    it, and report whether the linked install runs there; it is unlinked if not."""
    mounts = getattr(runtime.config, "mounts", {})
    if prebuilt is None or prebuilt.mount not in mounts:
        return False
    root = runtime.with_user(None)
    linked = await root.run(
        ["sh", "-c", _LINK, "sh", prebuilt.mount, *prebuilt.dirs], {}
    )
    if linked.exit_code:
        return False
    if not (await runtime.run(["sh", "-c", prebuilt.probe], {})).exit_code:
        return True
    logger.info("prebuilt: %s does not run here; installing instead", prebuilt.name)
    await root.run(["sh", "-c", _UNLINK, "sh", *prebuilt.dirs], {})
    return False


async def _has_head(name: str) -> bool:
    from prime_sandboxes import APIError

    try:
        async with leased_client() as client:
            volume = await client.volumes.get(name)
    except APIError as error:
        if not str(error).startswith("HTTP 404"):
            logger.warning("prebuilt: cannot read volume %s: %s", name, error)
        return False
    return volume.head_checkpoint_id is not None


async def _build_logged(prebuilt: Prebuilt) -> None:
    try:
        await build(prebuilt)
    except Exception:
        _failed[prebuilt.name] = time.monotonic()
        logger.warning("prebuilt: build of %s failed", prebuilt.name, exc_info=True)


async def _check(runtime: Runtime, script: list[str], step: str) -> None:
    result = await runtime.run(["sh", "-c", script[0], "sh", *script[1:]], {})
    if result.exit_code:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise SandboxError(f"prebuilt {step} failed: {detail}")
