"""A Harbor image's ENTRYPOINT starts with the box, as Harbor's Docker start runs it."""

import asyncio

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import SubprocessConfig
from verifiers.v1.tasksets.harbor import entrypoint
from verifiers.v1.tasksets.harbor.entrypoint import COMMAND, READY, entrypoint_argv


def test_the_final_stage_entrypoint_takes_harbors_command(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.12 AS base\n"
        'ENTRYPOINT ["/usr/local/bin/setup"]\n'
        'CMD ["bash"]\n'
        "FROM base\n"
        "ARG AGENT=agent\n"
        "RUN echo hi \\\n"
        "  && echo there\n"
        "USER ${AGENT}:${AGENT}\n"
    )
    # The image's CMD is Harbor's to replace; the stage inherits its base's start.
    assert entrypoint_argv(dockerfile) == (["/usr/local/bin/setup", *COMMAND], "agent")
    dockerfile.write_text("FROM ubuntu\nENTRYPOINT /start.sh --now\nUSER root\n")
    assert entrypoint_argv(dockerfile) == (
        ["/bin/sh", "-c", "/start.sh --now", READY],
        None,
    )
    dockerfile.write_text('FROM base-with-its-own\nENTRYPOINT []\nCMD ["x"]\n')
    assert entrypoint_argv(dockerfile) is None


@pytest.mark.parametrize(
    "script, state",
    [
        # Setup, then the command it was handed.
        ('cp -R "$STARTER/." "$APP/"\nexec "$@"\n', "reached"),
        # A service in the background, then the image's own sleep.
        (
            (
                'sh -c "sleep 1; touch $APP/served" &\nsleep 2\n'
                'cp -R "$STARTER/." "$APP/"\nexec sleep infinity\n'
            ),
            "execd",
        ),
    ],
)
async def test_setup_waits_for_the_entrypoint_to_finish_its_start(
    tmp_path, monkeypatch, script, state
):
    starter, app = tmp_path / "starter", tmp_path / "app"
    starter.mkdir()
    app.mkdir()
    (starter / "Cargo.toml").write_text("[package]\n")
    script_path = tmp_path / "setup.sh"
    script_path.write_text("#!/bin/sh\nset -eu\n" + script)
    script_path.chmod(0o755)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f'FROM scratch\nENTRYPOINT ["{script_path}"]\n')
    monkeypatch.setattr(entrypoint, "PID", str(tmp_path / "pid"))
    monkeypatch.setattr(entrypoint, "LOG", str(tmp_path / "log"))
    async with vf.runtimes.provision_runtime(SubprocessConfig()) as box:
        runtime = box.with_env({"STARTER": str(starter), "APP": str(app)})
        started = await entrypoint.start_entrypoint(runtime, dockerfile, wait=30)
        assert started["state"] == state
        assert (app / "Cargo.toml").is_file()
        if state == "execd":
            await asyncio.sleep(0)
            assert (app / "served").is_file()


def test_a_verifier_image_that_runs_its_tests_on_start_is_not_started(tmp_path):
    """A one-shot verifier image's entrypoint is the grade; it runs once staged."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text('FROM rust\nENTRYPOINT ["/bin/sh", "/tests/test.sh"]\n')
    assert entrypoint_argv(dockerfile) is None
