"""Native thread-pool limits for environment workers.

Env workers run many independent rollouts concurrently. For multimodal
rollouts, each request can enter PIL/NumPy/torch/HF image-processing code from
an executor thread. On high-core hosts, leaving OpenMP/BLAS uncapped lets every
calling OS thread create a wide native worker team, which can explode thread
count and glibc arenas.

The helpers here intentionally target env-server/env-worker process boundaries,
not trainer or inference processes.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Iterator, Mapping

NATIVE_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "MALLOC_ARENA_MAX": "2",
    "RAYON_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

ENV_WORKER_MAX_THREADS_ENV = "VF_ENV_WORKER_MAX_THREADS"
DEFAULT_ENV_WORKER_MAX_THREADS = min(128, max(32, os.cpu_count() or 1))

_THREADPOOL_LIMITER = None
_MALLOC_TRIM_STARTED = False


def native_thread_limited_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a copy of *env* with env-worker native thread caps applied."""

    out = dict(os.environ if env is None else env)
    out.update(NATIVE_THREAD_ENV)
    return out


@contextlib.contextmanager
def scoped_native_thread_limits() -> Iterator[None]:
    """Temporarily apply process-start env caps around child process spawning."""

    old = {key: os.environ.get(key) for key in NATIVE_THREAD_ENV}
    os.environ.update(NATIVE_THREAD_ENV)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def env_worker_max_threads() -> int:
    """Return the env-worker default executor thread cap."""

    value = os.getenv(ENV_WORKER_MAX_THREADS_ENV)
    if value is None:
        return DEFAULT_ENV_WORKER_MAX_THREADS

    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_ENV_WORKER_MAX_THREADS

    if parsed < 1:
        return DEFAULT_ENV_WORKER_MAX_THREADS
    return parsed


def configure_runtime_native_threads(*, malloc_trim: bool = True) -> None:
    """Best-effort runtime caps for already-started env workers.

    Process-start env vars are the real fix for glibc/OpenMP initialization.
    This fallback still covers torch intra-op pools and libraries that
    ``threadpoolctl`` can adjust after import.
    """

    os.environ.update(NATIVE_THREAD_ENV)

    try:
        torch = importlib.import_module("torch")

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass

    try:
        from threadpoolctl import threadpool_limits

        global _THREADPOOL_LIMITER
        _THREADPOOL_LIMITER = threadpool_limits(limits=1)
    except Exception:
        pass

    if not malloc_trim:
        return

    global _MALLOC_TRIM_STARTED
    if _MALLOC_TRIM_STARTED:
        return
    _MALLOC_TRIM_STARTED = True

    try:
        import ctypes
        import threading
        import time

        libc = ctypes.CDLL("libc.so.6")

        def trim_loop() -> None:
            while True:
                time.sleep(15)
                try:
                    libc.malloc_trim(0)
                except Exception:
                    pass

        threading.Thread(target=trim_loop, daemon=True, name="malloc-trim").start()
    except Exception:
        pass
