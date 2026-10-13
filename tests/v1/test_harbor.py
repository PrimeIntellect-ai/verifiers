"""Harbor setup from task TOML to runtime calls, without live sandboxes."""

import io
import tarfile
import tomllib
from pathlib import Path
from unittest.mock import ANY

import pytest
import tomli_w
from pydantic import ValidationError

from verifiers.v1.runtimes.base import ProgramResult
from verifiers.v1.task import WireTaskData
from verifiers.v1.tasksets.harbor.taskset import (
    HarborConfig,
    HarborData,
    HarborTask,
    HarborTaskset,
    VerifierConfig,
    parse_task,
    verifier_box_data,
)

COMMAND = "sh bootstrap.sh && printf '%s\\n' 'setup complete'"
VERIFIER_COMMAND = "sh verifier-bootstrap.sh && touch /tmp/verifier-ready"
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
VERIFIER_SETUP = ("run", ["sh", "-c", VERIFIER_COMMAND], {})


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
def record_healthcheck(monkeypatch, request):
    if not getattr(request, "param", True):
        return

    async def wait_for_health(self, runtime, healthcheck):
        runtime.events.append(("health", healthcheck))

    monkeypatch.setattr(HarborTask, "wait_for_health", wait_for_health)


@pytest.fixture
def task_dir(tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "bootstrap.sh").write_text("touch /tmp/ready\n")
    return str(tmp_path)


@pytest.fixture
def write_harbor_task(task_dir):
    pytest.importorskip("harbor")
    path = Path(task_dir)
    (path / "instruction.md").write_text("Complete the task.\n")
    (path / "tests").mkdir()
    (path / "tests" / "test.sh").write_text(
        "#!/bin/sh\nprintf '1' > /logs/verifier/reward.txt\n"
    )

    def write(config: dict) -> Path:
        text = tomli_w.dumps(config)
        tomllib.loads(text)
        (path / "task.toml").write_text(text)
        return path

    return write


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


@pytest.mark.parametrize(
    ("verifier_image", "verifier_fields"),
    [
        pytest.param(None, {}, id="fresh-no-inheritance"),
        pytest.param("verifier:test", {}, id="image-no-inheritance"),
        pytest.param(None, {"setup_command": None}, id="nullable"),
        pytest.param("verifier:test", {"setup_command": ""}, id="empty"),
        pytest.param(None, {"setup_command": VERIFIER_COMMAND}, id="fresh-opt-in"),
        pytest.param(
            "verifier:test", {"setup_command": VERIFIER_COMMAND}, id="image-opt-in"
        ),
    ],
)
async def test_setup_command_wire_roundtrip_and_separate_verifier(
    verifier_image, verifier_fields, task_dir
) -> None:
    verifier_healthcheck = {"command": "test -f /tmp/verifier-ready"}
    data = HarborData(
        task_dir=task_dir,
        upload_environment=True,
        setup_command=COMMAND,
        healthcheck=HEALTHCHECK,
        verifier=VerifierConfig(
            healthcheck=verifier_healthcheck,
            fresh_copy=verifier_image is None,
            **verifier_fields,
        ),
        verifier_image=verifier_image,
    )
    wire = WireTaskData.model_validate_json(data.model_dump_json())
    payload = wire.model_dump(mode="json")
    assert payload["setup_command"] == COMMAND
    verifier_command = verifier_fields.get("setup_command")
    assert payload["verifier"]["setup_command"] == verifier_command
    task = HarborTask(HarborTask.data_type().model_validate(payload))
    grader = HarborTask(verifier_box_data(task.data), task.config)
    assert task.data.setup_command == COMMAND
    assert task.data.verifier.setup_command == verifier_command
    assert grader.data.setup_command == verifier_command

    solver_runtime, verifier_runtime = FakeRuntime(), FakeRuntime()
    await task.setup(solver_runtime)
    await grader.setup(verifier_runtime)

    assert solver_runtime.events == [*STAGING, SETUP, ("health", HEALTHCHECK)]
    assert verifier_runtime.events == [
        *(STAGING if verifier_image is None else []),
        *([VERIFIER_SETUP] if verifier_command else []),
        ("health", verifier_healthcheck),
    ]


@pytest.mark.parametrize("record_healthcheck", [False], indirect=True)
async def test_load_agent_setup_command_before_real_healthcheck(
    write_harbor_task, monkeypatch
) -> None:
    path = write_harbor_task(
        {
            "environment": {
                "docker_image": "agent:test",
                "setup_command": COMMAND,
                "healthcheck": HEALTHCHECK,
            }
        }
    )
    # Use the real cache/load path, with a local package instead of a download.
    monkeypatch.setattr("verifiers.v1.tasksets.harbor.taskset.CACHE", path.parent)
    [task] = HarborTaskset(HarborConfig(dataset=path.name)).load()
    assert task.data.prompt == "Complete the task."
    assert task.data.setup_command == COMMAND
    assert task.data.verifier is None
    runtime = FakeRuntime()

    await task.setup(runtime)

    assert runtime.events == [
        *STAGING,
        SETUP,
        ("run", ["sh", "-c", HEALTHCHECK["command"]], {}),
    ]


@pytest.mark.parametrize("fields", [{}, {"setup_command": ""}], ids=["unset", "empty"])
async def test_parse_setup_command_defaults_and_empty(
    write_harbor_task, fields
) -> None:
    path = write_harbor_task(
        {
            "environment": {"docker_image": "agent:test", **fields},
            "verifier": {"environment_mode": "separate", **fields},
        }
    )
    data = parse_task(path, 0, HarborConfig())
    assert data.setup_command == fields.get("setup_command")
    assert data.verifier.setup_command == fields.get("setup_command")
    agent_runtime, verifier_runtime = FakeRuntime(), FakeRuntime()

    await HarborTask(data).setup(agent_runtime)
    await HarborTask(verifier_box_data(data)).setup(verifier_runtime)

    assert agent_runtime.events == verifier_runtime.events == STAGING


@pytest.mark.parametrize(
    ("verifier_image", "agent_command", "verifier_command"),
    [
        pytest.param(None, COMMAND, None, id="fresh-no-inheritance"),
        pytest.param("verifier:test", COMMAND, None, id="image-no-inheritance"),
        pytest.param(None, COMMAND, VERIFIER_COMMAND, id="fresh-distinct"),
        pytest.param("verifier:test", COMMAND, VERIFIER_COMMAND, id="image-distinct"),
        pytest.param(None, None, VERIFIER_COMMAND, id="verifier-only"),
    ],
)
async def test_parse_separate_verifier_setup_command(
    write_harbor_task, verifier_image, agent_command, verifier_command
) -> None:
    environment = {"docker_image": "agent:test"}
    verifier = {"environment_mode": "separate"}
    if agent_command is not None:
        environment["setup_command"] = agent_command
    if verifier_command is not None:
        verifier["setup_command"] = verifier_command
    if verifier_image is not None:
        verifier["environment"] = {"docker_image": verifier_image}
    path = write_harbor_task({"environment": environment, "verifier": verifier})

    data = parse_task(path, 0, HarborConfig())
    assert data.setup_command == agent_command
    assert data.verifier.setup_command == verifier_command
    assert data.verifier.fresh_copy is (verifier_image is None)
    assert data.verifier_image == verifier_image
    grader_data = verifier_box_data(data)
    assert grader_data.setup_command == verifier_command
    assert data.setup_command == agent_command
    agent_runtime, verifier_runtime = FakeRuntime(), FakeRuntime()

    await HarborTask(data).setup(agent_runtime)
    await HarborTask(grader_data).setup(verifier_runtime)

    assert agent_runtime.events == [*STAGING, *([SETUP] if agent_command else [])]
    assert verifier_runtime.events == [
        *(STAGING if verifier_image is None else []),
        *([VERIFIER_SETUP] if verifier_command else []),
    ]


@pytest.mark.parametrize("section", ["environment", "verifier"])
@pytest.mark.parametrize(
    "value",
    [42, True, ["sh", "bootstrap.sh"], {"command": COMMAND}],
    ids=["integer", "boolean", "array", "table"],
)
def test_parse_rejects_non_string_setup_command(
    write_harbor_task, section, value
) -> None:
    path = write_harbor_task({section: {"setup_command": value}})

    with pytest.raises(ValidationError, match=rf"{section}\.setup_command"):
        parse_task(path, 0, HarborConfig())


@pytest.mark.parametrize(
    ("mode", "ignore_separate"),
    [
        pytest.param(None, False, id="default-shared"),
        pytest.param("shared", False, id="explicit-shared"),
        pytest.param("separate", True, id="ignored-separate"),
    ],
)
async def test_parse_shared_verifier_setup_stays_out_of_agent_box(
    write_harbor_task, mode, ignore_separate
) -> None:
    verifier = {"setup_command": VERIFIER_COMMAND}
    if mode is not None:
        verifier["environment_mode"] = mode
    path = write_harbor_task(
        {
            "environment": {"docker_image": "agent:test", "setup_command": COMMAND},
            "verifier": verifier,
        }
    )
    data = parse_task(path, 0, HarborConfig(ignore_separate_verifier=ignore_separate))
    assert data.verifier is None
    assert data.setup_command == COMMAND
    runtime = FakeRuntime()

    await HarborTask(data).setup(runtime)

    assert runtime.events == [*STAGING, SETUP]
