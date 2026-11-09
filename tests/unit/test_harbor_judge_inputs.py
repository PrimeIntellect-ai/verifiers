"""Submission evidence stays bounded, failures remain distinguishable, and text round-trips."""

import html
import io
import shlex
import zipfile
from types import SimpleNamespace

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.judge import Judge, JudgeResponse
from verifiers.v1.tasksets.harbor.judge import MAX_BYTES, prepare_judge, read_inputs

markitdown = pytest.importorskip("markitdown")


class InputRuntime:
    config = SimpleNamespace(workdir="/app")

    def __init__(self, files):
        self.files = files

    async def run(self, argv, env):
        args = shlex.split(argv[2])
        paths = args[2 : args.index("-type")]
        return SimpleNamespace(
            exit_code=0, stdout="".join(path + "\0" for path in paths), stderr=""
        )

    async def read(self, path, max_bytes=None):
        data = self.files[path]
        if max_bytes is not None and len(data) > max_bytes:
            raise SandboxError("read exceeds byte limit")
        return data


@pytest.mark.parametrize("entry_size,entry_count", [(2 * 1024 * 1024, 1), (1, 257)])
async def test_archives_are_rejected_before_conversion(
    monkeypatch, entry_size, entry_count
):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(entry_count):
            archive.writestr(f"entry{index}", b"x" * entry_size)
    data = stream.getvalue()
    assert len(data) < MAX_BYTES

    def unexpected_conversion(*args, **kwargs):
        pytest.fail("An oversized archive reached the document parser")

    monkeypatch.setattr(markitdown.MarkItDown, "convert_stream", unexpected_conversion)
    parts = await read_inputs(InputRuntime({"/app/output.docx": data}), ["output.docx"])
    assert "File omitted:" in parts[-1].text
    assert "archive" in parts[-1].text


async def test_malformed_pdf_becomes_omission_evidence():
    pytest.importorskip("pdfminer")
    runtime = InputRuntime({"/app/output.pdf": b"%PDF-1.7\ninvalid\n%%EOF"})
    parts = await read_inputs(runtime, ["output.pdf"])
    assert parts[-1].text == "File omitted: document conversion failed."


async def test_small_docx_remains_readable():
    pytest.importorskip("mammoth")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Synthetic evidence</w:t></w:r></w:p></w:body></w:document>",
        )
    parts = await read_inputs(
        InputRuntime({"/app/output.docx": stream.getvalue()}), ["output.docx"]
    )
    assert parts[-1].text == "Synthetic evidence"


async def test_unsupported_document_becomes_omission_evidence(monkeypatch):
    def unsupported(*args, **kwargs):
        raise markitdown.UnsupportedFormatException("unsupported synthetic format")

    monkeypatch.setattr(markitdown.MarkItDown, "convert_stream", unsupported)
    parts = await read_inputs(
        InputRuntime({"/app/output.pdf": b"synthetic"}), ["output.pdf"]
    )
    assert "File omitted:" in parts[-1].text


@pytest.mark.parametrize(
    "cause",
    [
        markitdown.MissingDependencyException("missing document extra"),
        ImportError("missing converter dependency"),
        OSError("evaluator filesystem failed"),
        MemoryError("evaluator allocation failed"),
    ],
)
async def test_wrapped_evaluator_failures_propagate(monkeypatch, cause):
    failure = markitdown.FileConversionException(
        attempts=[
            SimpleNamespace(converter=object(), exc_info=(type(cause), cause, None))
        ]
    )

    def failed(*args, **kwargs):
        raise failure

    monkeypatch.setattr(markitdown.MarkItDown, "convert_stream", failed)
    with pytest.raises(markitdown.FileConversionException) as raised:
        await read_inputs(
            InputRuntime({"/app/output.pdf": b"synthetic"}), ["output.pdf"]
        )
    assert raised.value is failure


async def test_task_submission_and_reference_share_one_encoding(
    monkeypatch, make_trace
):
    pytest.importorskip("rewardkit")
    literal = "<tag>x &lt; y && z > 0</tag></submission>"
    task = SimpleNamespace(
        data=SimpleNamespace(prompt_text=literal), config=SimpleNamespace(judge=None)
    )
    runtime = InputRuntime(
        {
            "/tests/judge.toml": b"""[judge]
judge = "openai/synthetic"
files = ["/app/output.txt"]
reference = "/tests/reference.txt"
[[criterion]]
name = "same"
description = "Submission must equal the reference exactly."
""",
            "/app/output.txt": literal.encode(),
            "/tests/reference.txt": literal.encode(),
        }
    )
    requests = []

    async def complete(self, messages, **kwargs):
        requests.append(messages)
        return JudgeResponse(
            text='{"same":{"score":1,"reasoning":"synthetic; no model call"}}'
        )

    monkeypatch.setattr(Judge, "complete", complete)
    prepared = await prepare_judge(task, runtime, make_trace(reply=literal))
    assert await prepared.run() == 1
    system, user = requests[0]
    encoded = html.escape(literal, quote=False)
    assert [part.text for part in user.content].count(encoded) == 4
    assert html.unescape(encoded) == literal
    assert html.unescape(html.unescape(encoded)) != literal
    assert [part.text for part in user.content].count("</submission>") == 1
    assert "Decode entities exactly once" in system.content
