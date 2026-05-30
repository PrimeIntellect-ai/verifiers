import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class StubEmbeddingFunction:
    def __class_getitem__(cls, item):
        return cls


class StubPersistentClient:
    def __init__(self, path: str):
        pass

    def get_or_create_collection(self, **kwargs: object) -> object:
        return object()


class StubWithKwargs:
    def __init__(self, **kwargs: object):
        pass


class StubOpenAIError(Exception):
    pass


def stub_module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def install_wiki_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = {
        "chromadb": stub_module("chromadb", PersistentClient=StubPersistentClient),
        "chromadb.api": stub_module("chromadb.api"),
        "chromadb.api.types": stub_module(
            "chromadb.api.types",
            Embeddable=object,
            EmbeddingFunction=StubEmbeddingFunction,
        ),
        "chromadb.utils": stub_module("chromadb.utils"),
        "chromadb.utils.embedding_functions": stub_module(
            "chromadb.utils.embedding_functions",
            OpenAIEmbeddingFunction=StubWithKwargs,
        ),
        "datasets": stub_module("datasets", load_dataset=lambda *args, **kwargs: []),
        "openai": stub_module(
            "openai",
            APIError=StubOpenAIError,
            APITimeoutError=StubOpenAIError,
            AsyncOpenAI=StubWithKwargs,
            RateLimitError=StubOpenAIError,
        ),
    }
    stubs["chromadb.utils"].embedding_functions = stubs[
        "chromadb.utils.embedding_functions"
    ]
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)


def load_wiki_module(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    install_wiki_stubs(monkeypatch)
    module_path = (
        Path(__file__).parents[1] / "environments" / "wiki_search" / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_wiki_search_v1_default_and_explicit_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_wiki_module("wiki_search_v1", monkeypatch)
    wrapper = load_wiki_module("wiki_search", monkeypatch)

    env = wrapper.load_environment(
        v1=True,
        corpus_dataset="test/corpus",
        corpus_split="validation",
        chroma_db_dir="/tmp/wiki",
        embed_model="test-embed",
    )

    assert env.taskset.config.corpus_dataset == "test/corpus"
    assert env.taskset.config.corpus_split == "validation"
    assert list(env.taskset.named_toolsets) == ["wiki"]
    assert len(env.taskset.toolsets) == 1
    assert len(env.taskset.rewards) == 1

    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda *args, **kwargs: [{"question": "question?", "answer": "answer"}],
    )
    rows = list(module.load_tasks(max_turns=3))

    assert rows[0]["max_turns"] == 3
    assert "judge_model" not in rows[0]
    assert "judge_base_url" not in rows[0]
    assert "judge_api_key_var" not in rows[0]

    taskset = module.WikiSearchTaskset(
        config=module.WikiSearchTasksetConfig(toolsets={"custom": {"tools": []}})
    )

    assert list(taskset.named_toolsets) == ["wiki", "custom"]
    assert len(taskset.toolsets) == 2

    configured_env = module.load_environment(
        config=module.WikiSearchEnvConfig(harness={"max_turns": 7})
    )

    assert configured_env.harness.config.max_turns == 7


def test_wiki_search_v1_rejects_legacy_judge_endpoint_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = load_wiki_module("wiki_search", monkeypatch)

    with pytest.raises(ValueError, match="state.get_endpoint_config"):
        wrapper.load_environment(
            v1=True,
            judge_base_url="https://judge.example/v1",
        )
