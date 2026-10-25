"""A background job whose wrapper died leaving an empty exit file counts as dead, not as
still running (its exit status never comes, and its poll would wait forever)."""

import asyncio
from types import SimpleNamespace

from verifiers.v1.runtimes.backends.prime import PrimeBackend


class LocalClient:
    async def execute_command(self, sandbox_id, command, timeout=60, user=None):
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return SimpleNamespace(
            stdout=stdout.decode(), stderr=stderr.decode(), exit_code=proc.returncode
        )


def alive(tmp_path, exit_text):
    exit_file = tmp_path / "job.exit"
    if exit_text is not None:
        exit_file.write_text(exit_text)
    job = SimpleNamespace(
        job_id="j",
        exit_file=str(exit_file),
        stdout_log_file=str(tmp_path / "job.stdout.log"),
    )
    backend = PrimeBackend.__new__(PrimeBackend)
    backend._client = LocalClient()
    backend.info = SimpleNamespace(id="box")
    return asyncio.run(backend._job_alive(job, None))


def test_an_empty_exit_file_with_nothing_holding_the_output_is_dead(tmp_path):
    assert alive(tmp_path, "") is False


def test_a_written_exit_status_counts(tmp_path):
    assert alive(tmp_path, "0\n") is True


def test_no_exit_file_and_no_holder_is_dead(tmp_path):
    assert alive(tmp_path, None) is False
