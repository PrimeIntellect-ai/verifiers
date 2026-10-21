"""Which files an image copied from its build context, for Docker's root ownership."""

import os
import time

from verifiers.v1.tasksets.harbor.context_owners import context_files


def test_the_final_stages_context_copies_are_listed_with_their_owner(tmp_path):
    for name in ("validator.py", "setup.sh", "owned.txt", "seeded.txt"):
        (tmp_path / name).write_text("x")
    (tmp_path / "app" / "lib").mkdir(parents=True)
    (tmp_path / "app" / "lib" / "a.py").write_text("x")
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.12 AS build\n"
        "COPY setup.sh /build/setup.sh\n"
        "FROM python:3.12\n"
        "WORKDIR /srv\n"
        "COPY validator.py /usr/local/libexec/validator\n"
        "COPY app/ app/\n"
        "COPY setup.sh seeded.txt /opt/\n"
        "COPY --chown=1000:1000 owned.txt /home/agent/owned.txt\n"
        "COPY owned.txt /home/agent/again.txt\n"
        "COPY --from=build /build/setup.sh /usr/local/bin/setup\n"
        "RUN chown -R agent:agent /home/agent\n"
    )
    owner = (os.getuid(), os.getgid())
    assert context_files(tmp_path / "Dockerfile") == [
        ("/usr/local/libexec/validator", True, *owner),
        ("/srv/app", False, *owner),
        ("/srv/app/lib", False, *owner),
        ("/srv/app/lib/a.py", True, *owner),
        ("/opt/setup.sh", True, *owner),
        ("/opt/seeded.txt", True, *owner),
    ]


def test_files_a_called_script_chowns_keep_their_owner(tmp_path):
    """A RUN that calls a context script handing /app to the agent sets those owners
    itself, as the image's own build does: they are left alone."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.rs").write_text("x")
    (tmp_path / "setup.sh").write_text("useradd agent\nchown -R agent:agent /app\n")
    (tmp_path / "Dockerfile").write_text(
        "FROM rust\nCOPY app/ /app/\nCOPY setup.sh /tmp/setup.sh\n"
        "RUN bash /tmp/setup.sh && rm /tmp/setup.sh\n"
    )
    assert [path for path, *_ in context_files(tmp_path / "Dockerfile")] == [
        "/tmp/setup.sh"
    ]


def test_a_chown_before_the_copy_does_not_cover_it(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "a.py").write_text("x")
    (tmp_path / "Dockerfile").write_text(
        "FROM python\nRUN chown -R agent /app\nCOPY app/ /app/\nRUN chown agent /opt\n"
    )
    assert [path for path, *_ in context_files(tmp_path / "Dockerfile")] == [
        "/app",
        "/app/a.py",
    ]


def test_a_context_of_thousands_of_files_is_read_in_moments(tmp_path):
    """Each RUN's scripts are read once, not once per copied file: a 5,000-file context
    took the box's setup (and the event loop under it) most of an hour."""
    tree = tmp_path / "src"
    for group in range(50):
        (tree / f"g{group}").mkdir(parents=True)
        for item in range(100):
            (tree / f"g{group}" / f"f{item}.c").write_text("x")
    (tmp_path / "setup.sh").write_text("chown -R agent /opt/env/src/g0\n")
    (tmp_path / "Dockerfile").write_text(
        "FROM debian\nCOPY . /opt/env\n"
        "RUN chmod 0755 /opt/env/setup.sh && /bin/sh /opt/env/setup.sh\n"
    )
    started = time.monotonic()
    paths = [path for path, *_ in context_files(tmp_path / "Dockerfile")]
    assert time.monotonic() - started < 5
    assert "/opt/env/src/g1/f0.c" in paths
    assert not any(path.startswith("/opt/env/src/g0") for path in paths)
    assert len(paths) == 1 + 1 + 1 + 1 + 49 * 101  # /opt/env, Dockerfile, setup.sh, src
