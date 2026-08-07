"""The daemon's derived socket paths must stay inside the unix sun_path limit."""

from types import SimpleNamespace

from verifiers.v1.harnesses.prime_agent.harness import (
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)

# AF_UNIX sun_path is 108 bytes on Linux and 104 on macOS; use the stricter one.
SUN_PATH_LIMIT = 104
# $TMPDIR/prime-agent-<uid>/worker-<12>-<12>.sock, per
# daemon-supervisor.ts workerSocketPath().
WORKER_SUFFIX = "/prime-agent-0/worker-abcdefabcdef-123456789012.sock"


def harness() -> PrimeAgentHarness:
    return PrimeAgentHarness(PrimeAgentHarnessConfig(id="prime-agent"))


def test_worker_socket_path_fits_sun_path():
    """A too-long TMPDIR makes listen() fail with EINVAL.

    The supervisor then blocks for its full 30s worker timeout, and the only
    symptom is an opaque ACP `create` timeout -- which is exactly how this cost a
    full live-E2E cycle to diagnose. The supervisor socket alone fit; the derived
    worker socket did not.
    """
    trace = SimpleNamespace(id="0123456789abcdef0123456789abcdef0123456789abcdef")
    derived = harness().tmp_dir(trace) + WORKER_SUFFIX
    assert len(derived) <= SUN_PATH_LIMIT, f"{len(derived)} bytes: {derived}"


def test_tmp_dir_is_unique_per_trace():
    a = SimpleNamespace(id="trace-a")
    b = SimpleNamespace(id="trace-b")
    assert harness().tmp_dir(a) != harness().tmp_dir(b)
