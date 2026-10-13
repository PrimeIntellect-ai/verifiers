"""Agent plumbing fixes carried from the hillclimb eval stack, against fakes: the chat
program's conversation in a file (never argv), a dead Prime job reported instead of
polled forever, Prime calls repeated through gateway errors, and the longer creation
wait of a box restored from a checkpoint. No sandbox or model is used."""

import json
from types import SimpleNamespace

import prime_sandboxes
import pytest

from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.harnesses.utils.launch import launch_chat_program
from verifiers.v1.runtimes import prime
from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime
from verifiers.v1.types import UserMessage

# --- the chat program's conversation goes in a file -----------------------------------


class RecordingRuntime:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.argv: list[str] = []

    async def write(self, path, data):
        self.files[path] = data

    async def prepare_uv_script(self, source, env, *, activate=True):
        return ["python", "program.py"]

    async def run_program(self, argv, env):
        self.argv = argv
        return ProgramResult(exit_code=0, stdout="", stderr="")


@pytest.mark.parametrize(
    "prompt", ["pkill -f needle", [UserMessage(content="pkill -f needle")]]
)
async def test_the_conversation_reaches_the_program_in_a_file_not_argv(prompt):
    # An agent's `pkill -f <word from the task>` matched a prompt on argv and killed
    # its own harness.
    runtime = RecordingRuntime()
    await launch_chat_program(
        "",
        HarnessConfig(id="bash"),
        SimpleNamespace(model="m"),
        SimpleNamespace(id="t1"),
        runtime,
        "http://e",
        "secret",
        {},
        "system needle",
        prompt,
    )
    assert not any("needle" in arg for arg in runtime.argv)
    (path,) = runtime.files
    assert f"--initial-messages-file={path}" in runtime.argv
    assert json.loads(runtime.files[path]) == [
        {"role": "system", "content": "system needle"},
        {"role": "user", "content": "pkill -f needle"},
    ]


# --- Prime jobs, gateway errors, restored boxes ---------------------------------------


@pytest.fixture
def runtime(monkeypatch) -> PrimeRuntime:
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    return PrimeRuntime(PrimeConfig())


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    real_sleep = prime.asyncio.sleep

    async def sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(prime.asyncio, "sleep", sleep)
    return sleeps


JOB = SimpleNamespace(
    job_id="j1", exit_file="/tmp/job_j1.exit", stdout_log_file="/tmp/job_j1.stdout.log"
)
RUNNING = SimpleNamespace(completed=False, stdout_error=None, stderr_error=None)
DONE = SimpleNamespace(
    completed=True,
    stdout_error=None,
    stderr_error=None,
    exit_code=0,
    stdout="out",
    stderr="",
)


class JobClient:
    """A Prime client whose job polls and liveness checks answer from scripts; the
    last answer of each repeats."""

    def __init__(self, polls, liveness=()):
        self.polls = list(polls)
        self.liveness = list(liveness)
        self.probes = 0

    async def start_background_job(self, *args, **kwargs):
        return JOB

    async def get_background_job(self, *args, **kwargs):
        answer = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    async def execute_command(self, sandbox_id, command, **kwargs):
        self.probes += 1
        assert JOB.exit_file in command and JOB.stdout_log_file in command
        answer = self.liveness.pop(0) if len(self.liveness) > 1 else self.liveness[0]
        if isinstance(answer, BaseException):
            raise answer
        return SimpleNamespace(stdout=answer + "\n")


@pytest.fixture
def check_every_poll(monkeypatch):
    monkeypatch.setattr(prime, "_LIVENESS_INTERVAL_SECONDS", 0)


async def test_a_job_whose_wrapper_died_is_reported_not_polled_forever(
    runtime, no_sleep, check_every_poll
):
    runtime.info.id = "sb1"
    runtime._client = client = JobClient([RUNNING], ["dead"])
    with pytest.raises(SandboxError) as caught:
        await runtime.run(["true"], {})
    assert "died without an exit status" in str(caught.value)
    assert client.probes == prime._LIVENESS_MISSES


async def test_a_live_or_unknown_job_is_waited_for(runtime, no_sleep, check_every_poll):
    # One miss is not a death, and a failed check is no verdict at all.
    runtime.info.id = "sb1"
    liveness = [
        "dead",
        "alive",
        "dead",
        RuntimeError("exec timed out"),
        "dead",
        "alive",
    ]
    runtime._client = JobClient([RUNNING] * 6 + [DONE], liveness)
    result = await runtime.run(["true"], {})
    assert (result.exit_code, result.stdout) == (0, "out")


async def test_job_polls_ride_out_gateway_errors(runtime, no_sleep):
    runtime.info.id = "sb1"
    runtime._client = JobClient(
        [RuntimeError("HTTP 502: Bad Gateway"), RuntimeError("HTTP 503"), DONE]
    )
    assert (await runtime.run(["true"], {})).stdout == "out"


async def test_repeats_stop_at_a_real_error_or_the_attempt_budget(no_sleep):
    calls = 0

    async def missing():
        nonlocal calls
        calls += 1
        raise RuntimeError("HTTP 404: no such file")

    with pytest.raises(RuntimeError, match="404"):
        await prime._repeatable(missing)
    assert calls == 1

    calls = 0

    async def down():
        nonlocal calls
        calls += 1
        raise RuntimeError("upstream unavailable")

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await prime._repeatable(down)
    assert calls == prime._TRANSIENT_ATTEMPTS
    assert no_sleep == [1, 2, 4]


class CreatingClient:
    def __init__(self, **kwargs):
        self.waits: list[int] = []

    async def create(self, request):
        self.request = request
        return SimpleNamespace(id="sb1", pending_image_build_id=None)

    async def wait_for_creation(self, sandbox_id, max_attempts):
        self.waits.append(max_attempts)

    async def execute_command(self, *args, **kwargs):
        return SimpleNamespace(stdout="")

    async def delete(self, sandbox_id):
        pass

    async def aclose(self):
        pass


@pytest.mark.parametrize(
    ("checkpoint", "attempts"),
    [(None, prime._BOOT_WAIT_ATTEMPTS), ("ck1", prime._RESTORE_WAIT_ATTEMPTS)],
)
async def test_a_restored_box_gets_the_longer_creation_wait(
    monkeypatch, checkpoint, attempts
):
    monkeypatch.setattr(prime, "ensure_prime_auth", lambda: None)
    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", CreatingClient)
    runtime = PrimeRuntime(PrimeConfig(checkpoint=checkpoint))
    await runtime.start()
    client = runtime._client
    await runtime.teardown()
    assert client.waits == [attempts]
    assert prime._RESTORE_WAIT_ATTEMPTS > prime._BOOT_WAIT_ATTEMPTS
