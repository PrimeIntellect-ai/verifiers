"""Which files an image copied from its build context, for Docker's root ownership."""

import os

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
