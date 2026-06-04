import multiprocessing as mp
import os

from verifiers.utils.native_threads import (
    DEFAULT_ENV_WORKER_MAX_THREADS,
    ENV_WORKER_MAX_THREADS_ENV,
    NATIVE_THREAD_ENV,
    env_worker_max_threads,
    native_thread_limited_env,
    scoped_native_thread_limits,
)


def _read_native_thread_env(queue):
    queue.put({key: os.environ.get(key) for key in NATIVE_THREAD_ENV})


def test_native_thread_limited_env_overrides_existing_values():
    env = native_thread_limited_env(
        {
            "PATH": "/bin",
            "OMP_NUM_THREADS": "192",
            "MALLOC_ARENA_MAX": "64",
        }
    )

    assert env["PATH"] == "/bin"
    assert {key: env[key] for key in NATIVE_THREAD_ENV} == NATIVE_THREAD_ENV


def test_scoped_native_thread_limits_reaches_spawned_process(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "192")
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_read_native_thread_env, args=(queue,))

    with scoped_native_thread_limits():
        process.start()

    child_env = queue.get(timeout=10)
    process.join(timeout=10)

    assert process.exitcode == 0
    assert child_env == NATIVE_THREAD_ENV
    assert os.environ["OMP_NUM_THREADS"] == "192"
    assert "MALLOC_ARENA_MAX" not in os.environ


def test_env_worker_max_threads_is_bounded_and_overridable(monkeypatch):
    monkeypatch.delenv(ENV_WORKER_MAX_THREADS_ENV, raising=False)
    assert env_worker_max_threads() == DEFAULT_ENV_WORKER_MAX_THREADS

    monkeypatch.setenv(ENV_WORKER_MAX_THREADS_ENV, "96")
    assert env_worker_max_threads() == 96

    monkeypatch.setenv(ENV_WORKER_MAX_THREADS_ENV, "0")
    assert env_worker_max_threads() == DEFAULT_ENV_WORKER_MAX_THREADS

    monkeypatch.setenv(ENV_WORKER_MAX_THREADS_ENV, "not-an-int")
    assert env_worker_max_threads() == DEFAULT_ENV_WORKER_MAX_THREADS
