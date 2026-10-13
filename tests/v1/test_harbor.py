"""Harbor setup at the runtime boundary, without Harbor or a live sandbox."""

import io
import tarfile
from unittest.mock import ANY

import pytest

from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.task import WireTaskData
from verifiers.v1.tasksets.harbor.taskset import (
    HarborData,
    HarborTask,
    VerifierConfig,
    verifier_box_data,
)

COMMAND = "sh bootstrap.sh && printf '%s\\n' 'setup complete'"
HEALTHCHECK = {"command": "test -f /tmp/ready"}
STAGING = [
    ("write", "/tmp/environment.tgz", ANY),
    (
        "run",
        [
            "sh",
            "-c",
            "tar --no-same-owner -xzf /tmp/environment.tgz && rm /tmp/environment.tgz",
        ],
        {},
    ),
]
SETUP = ("run", ["sh", "-c", COMMAND], {})


class FakeRuntime:
    def __init__(self, *results: ProgramResult, write_error: OSError | None = None):
        self.events = []
        self.results = iter(results)
        self.write_error = write_error

    async def write(self, path: str, data: bytes) -> None:
        self.events.append(("write", path, data))
        if self.write_error is not None:
            raise self.write_error

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.events.append(("run", argv, env))
        return next(self.results, ProgramResult(0, "", ""))


@pytest.fixture(autouse=True)
def record_healthcheck(monkeypatch):
    async def wait_for_health(self, runtime, healthcheck):
        runtime.events.append(("health", healthcheck))

    monkeypatch.setattr(HarborTask, "wait_for_health", wait_for_health)


@pytest.fixture
def task_dir(tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "bootstrap.sh").write_text("touch /tmp/ready\n")
    return str(tmp_path)


async def test_setup_stages_then_runs_command_then_health(task_dir) -> None:
    task = HarborTask(
        HarborData(
            task_dir=task_dir,
            upload_environment=True,
            setup_command=COMMAND,
            healthcheck=HEALTHCHECK,
        )
    )
    runtime = FakeRuntime()

    await task.setup(runtime)

    assert runtime.events == [*STAGING, SETUP, ("health", HEALTHCHECK)]
    with tarfile.open(fileobj=io.BytesIO(runtime.events[0][2]), mode="r:gz") as tar:
        assert tar.getnames() == ["bootstrap.sh"]
        assert tar.extractfile("bootstrap.sh").read() == b"touch /tmp/ready\n"


@pytest.mark.parametrize(
    "fields",
    [{}, {"setup_command": None}, {"setup_command": ""}],
    ids=["unset", "none", "empty"],
)
@pytest.mark.parametrize("upload_environment", [False, True])
async def test_setup_command_noop(fields, upload_environment, task_dir) -> None:
    healthcheck = HEALTHCHECK if upload_environment else None
    data = HarborData(
        task_dir=task_dir,
        upload_environment=upload_environment,
        healthcheck=healthcheck,
        **fields,
    )
    assert data.setup_command == fields.get("setup_command")
    runtime = FakeRuntime()

    await HarborTask(data).setup(runtime)

    assert runtime.events == (
        [*STAGING, ("health", healthcheck)] if upload_environment else []
    )


@pytest.mark.parametrize("healthcheck", [None, HEALTHCHECK])
async def test_setup_command_without_staging(healthcheck) -> None:
    runtime = FakeRuntime()
    task = HarborTask(HarborData(setup_command=COMMAND, healthcheck=healthcheck))

    await task.setup(runtime)

    assert runtime.events == [SETUP] + (
        [("health", healthcheck)] if healthcheck is not None else []
    )


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        pytest.param("ignored stdout", "discarded stderr:" + "E" * 500, id="stderr"),
        pytest.param("discarded stdout:" + "O" * 500, "", id="stdout"),
    ],
)
async def test_setup_command_failure_stops_before_health(
    stdout, stderr, task_dir
) -> None:
    runtime = FakeRuntime(ProgramResult(0, "", ""), ProgramResult(23, stdout, stderr))
    task = HarborTask(
        HarborData(
            task_dir=task_dir,
            upload_environment=True,
            setup_command=COMMAND,
            healthcheck=HEALTHCHECK,
        )
    )

    with pytest.raises(RuntimeError, match="setup command failed") as exc:
        await task.setup(runtime)

    message = str(exc.value)
    output = stderr or stdout
    assert "23" in message
    assert message.endswith(output[-500:])
    assert output[-501:] not in message
    assert "discarded" not in message and "ignored stdout" not in message
    assert runtime.events == [*STAGING, SETUP]  # No retry or healthcheck.


@pytest.mark.parametrize("failure", ["write", "extract"])
async def test_staging_failure_prevents_command_and_health(failure, task_dir) -> None:
    runtime = FakeRuntime(
        ProgramResult(17, "", "cannot extract"),
        write_error=OSError("cannot upload") if failure == "write" else None,
    )
    task = HarborTask(
        HarborData(
            task_dir=task_dir,
            upload_environment=True,
            setup_command=COMMAND,
            healthcheck=HEALTHCHECK,
        )
    )
    error = OSError if failure == "write" else RuntimeError
    message = "cannot upload" if failure == "write" else "environment setup failed"

    with pytest.raises(error, match=message):
        await task.setup(runtime)

    assert runtime.events == (STAGING[:1] if failure == "write" else STAGING)


async def test_setup_command_runs_on_every_setup_call() -> None:
    task = HarborTask(HarborData(setup_command=COMMAND, healthcheck=HEALTHCHECK))
    runtime = FakeRuntime()

    await task.setup(runtime)
    await task.setup(runtime)

    assert runtime.events == [SETUP, ("health", HEALTHCHECK)] * 2


@pytest.mark.parametrize("verifier_image", [None, "verifier:test"])
async def test_setup_command_wire_roundtrip_and_separate_verifier(
    verifier_image, task_dir
) -> None:
    verifier_healthcheck = {"command": "test -f /tmp/verifier-ready"}
    data = HarborData(
        task_dir=task_dir,
        upload_environment=True,
        setup_command=COMMAND,
        healthcheck=HEALTHCHECK,
        verifier=VerifierConfig(healthcheck=verifier_healthcheck),
        verifier_image=verifier_image,
    )
    wire = WireTaskData.model_validate_json(data.model_dump_json())
    payload = wire.model_dump(mode="json")
    assert payload["setup_command"] == COMMAND
    task = HarborTask(HarborTask.data_type().model_validate(payload))
    grader = HarborTask(verifier_box_data(task.data), task.config)
    assert task.data.setup_command == grader.data.setup_command == COMMAND

    solver_runtime, verifier_runtime = FakeRuntime(), FakeRuntime()
    await task.setup(solver_runtime)
    await grader.setup(verifier_runtime)

    assert solver_runtime.events == [*STAGING, SETUP, ("health", HEALTHCHECK)]
    assert verifier_runtime.events == [
        *(STAGING if verifier_image is None else []),
        SETUP,
        ("health", verifier_healthcheck),
    ]
