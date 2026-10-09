import pytest

from verifiers.v1.harnesses.null import harness as null_harness
from verifiers.v1.task import TaskData


def test_null_harness_accepts_compaction_threshold() -> None:
    config = null_harness.NullHarnessConfig.model_validate(
        {"id": "null", "compaction": {"summarize_at_tokens": 16384}}
    )

    assert config.compaction is not None
    assert config.compaction.summarize_at_tokens == 16384


@pytest.mark.asyncio
async def test_null_harness_forwards_compaction_threshold(monkeypatch) -> None:
    captured = {}

    async def fake_launch(*args, **kwargs):
        captured["extra_args"] = kwargs["extra_args"]
        return object()

    monkeypatch.setattr(null_harness, "launch_chat_program", fake_launch)
    harness = null_harness.NullHarness(
        null_harness.NullHarnessConfig(
            id="null",
            compaction=null_harness.CompactionConfig(summarize_at_tokens=16384),
        )
    )

    await harness.launch(
        object(),
        object(),
        object(),
        "http://model",
        "secret",
        {},
        TaskData(prompt="hi"),
    )

    assert captured["extra_args"] == ["--compaction", "--summarize-at-tokens=16384"]
