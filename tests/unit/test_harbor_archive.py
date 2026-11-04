"""Harbor archives reflect the current contents of local task directories."""

import io
import tarfile

from verifiers.v1.tasksets.harbor.taskset import make_tar


def test_make_tar_reflects_changed_content_in_the_same_directory(tmp_path):
    source = tmp_path / "payload.txt"
    source.write_bytes(b"before")
    first = make_tar(tmp_path)

    source.write_bytes(b"after!")
    second = make_tar(tmp_path)

    for archive_bytes, expected in [(first, b"before"), (second, b"after!")]:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            member = archive.extractfile("payload.txt")
            assert member is not None
            assert member.read() == expected


def test_process_scope_metadata_requires_explicit_boolean():
    import pytest

    from verifiers.v1.tasksets.harbor.taskset import _process_scope_flag

    assert not _process_scope_flag({})
    assert _process_scope_flag({"verifiers_process_scope": True})
    with pytest.raises(TypeError):
        _process_scope_flag({"verifiers_process_scope": "true"})
