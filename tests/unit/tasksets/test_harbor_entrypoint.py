"""A Harbor image's ENTRYPOINT starts with the box, as Harbor's Docker start runs it."""

import asyncio
import os
import signal
from pathlib import Path

import pytest

import verifiers.v1 as vf
from verifiers.v1.runtimes import SubprocessConfig
from verifiers.v1.tasksets.harbor import entrypoint
from verifiers.v1.tasksets.harbor.entrypoint import COMMAND, READY, entrypoint_argv


@pytest.fixture(autouse=True)
def only_this_tests_processes(tmp_path, monkeypatch):
    """The subprocess box is the host: list this test's processes, not the host's."""
    monkeypatch.setattr(
        entrypoint,
        "PROC_PIDS",
        f"$(for q in /proc/[0-9]*; do grep -qsF {tmp_path} $q/cmdline && echo $q; done)",
    )


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
        try:
            started = await entrypoint.start_entrypoint(runtime, dockerfile, wait=30)
        finally:
            # The entrypoint runs in a session of its own, as in a box: end it.
            pid = int((tmp_path / "pid").read_text())
            os.killpg(pid, signal.SIGKILL)
        assert started["state"] == state
        assert (app / "Cargo.toml").is_file()
        if state == "execd":
            assert (app / "served").is_file()


def test_a_verifier_image_that_runs_its_tests_on_start_is_not_started(tmp_path):
    """A one-shot verifier image's entrypoint is the grade; it runs once staged."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text('FROM rust\nENTRYPOINT ["/bin/sh", "/tests/test.sh"]\n')
    assert entrypoint_argv(dockerfile) is None


async def test_a_restored_box_restarts_what_the_entrypoint_left_running(
    tmp_path, monkeypatch
):
    """A checkpoint keeps files, not processes: a restore starts the entrypoint's services
    again as they ran, without running the entrypoint (its file setup) a second time."""
    app, run = tmp_path / "app", tmp_path / "run"
    app.mkdir()
    (run / "svc").mkdir(parents=True)
    script_path = tmp_path / "setup.sh"
    script_path.write_text(
        "#!/bin/sh\nset -eu\n"
        'echo setup >> "$APP/setups"\n'
        'sh -c "while :; do date +%s%N > $APP/beat; sleep 0.1; done" vf-test-svc &\n'
        'exec "$@"\n'
    )
    script_path.chmod(0o755)
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(f'FROM scratch\nENTRYPOINT ["{script_path}"]\n')
    monkeypatch.setattr(entrypoint, "PID", str(tmp_path / "pid"))
    monkeypatch.setattr(entrypoint, "LOG", str(tmp_path / "log"))
    monkeypatch.setattr(entrypoint, "STATE", str(tmp_path / "state"))
    monkeypatch.setattr(entrypoint, "SERVICES", str(tmp_path / "lib" / "services.json"))
    monkeypatch.setattr(
        entrypoint,
        "RUN_DIRS",
        f"find {run} -mindepth 1 -type d | while read -r d; do stat -c '%a %u %g %n' \"$d\"; done",
    )

    def service_pids() -> list[int]:
        pids = []
        for proc in os.listdir("/proc"):
            try:
                cmdline = Path(f"/proc/{proc}/cmdline").read_bytes()
            except OSError:
                continue
            if b"vf-test-svc" in cmdline and str(app).encode() in cmdline:
                pids.append(int(proc))
        return pids

    async with vf.runtimes.provision_runtime(SubprocessConfig()) as box:
        runtime = box.with_env({"APP": str(app)})
        try:
            started = await entrypoint.start_entrypoint(runtime, dockerfile, wait=30)
            assert started["state"] == "reached"
            assert any("vf-test-svc" in argv for argv in started["services"])
            # The checkpoint's box: the files stay, the processes and /run are gone.
            for pid in service_pids():
                os.kill(pid, signal.SIGKILL)
            os.killpg(int((tmp_path / "pid").read_text()), signal.SIGKILL)
            (run / "svc").rmdir()
            (app / "beat").unlink()
            restarted = await entrypoint.restart_services(runtime)
            assert any("vf-test-svc" in argv for argv in restarted["restarted"])
            for _ in range(100):
                if (app / "beat").exists():
                    break
                await asyncio.sleep(0.1)
            assert (app / "beat").is_file()
            assert (run / "svc").is_dir()
            # Its file setup ran once: the restore did not run the entrypoint again.
            assert (app / "setups").read_text() == "setup\n"
            again = await entrypoint.restart_services(runtime)
            assert again["restarted"] == [] and again["already_running"]
        finally:
            for pid in service_pids():
                os.kill(pid, signal.SIGKILL)
