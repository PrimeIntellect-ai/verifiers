import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


class StubEmbeddingFunction:
    def __class_getitem__(cls, item):
        return cls


class StubOpenAIEmbeddingFunction:
    def __init__(self, **kwargs: object):
        self.kwargs = kwargs


class StubPersistentClient:
    def __init__(self, path: str):
        self.path = path

    def get_or_create_collection(self, **kwargs: object) -> object:
        return object()


class StubAsyncOpenAI:
    def __init__(self, **kwargs: object):
        self.kwargs = kwargs


class StubOpenAIError(Exception):
    pass


def install_wiki_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    chromadb = ModuleType("chromadb")
    chromadb.PersistentClient = StubPersistentClient

    chromadb_api = ModuleType("chromadb.api")
    chromadb_api_types = ModuleType("chromadb.api.types")
    chromadb_api_types.Embeddable = object
    chromadb_api_types.EmbeddingFunction = StubEmbeddingFunction

    chromadb_utils = ModuleType("chromadb.utils")
    embedding_functions = ModuleType("chromadb.utils.embedding_functions")
    embedding_functions.OpenAIEmbeddingFunction = StubOpenAIEmbeddingFunction
    chromadb_utils.embedding_functions = embedding_functions

    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: []

    openai = ModuleType("openai")
    openai.APIError = StubOpenAIError
    openai.APITimeoutError = StubOpenAIError
    openai.AsyncOpenAI = StubAsyncOpenAI
    openai.RateLimitError = StubOpenAIError

    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    monkeypatch.setitem(sys.modules, "chromadb.api", chromadb_api)
    monkeypatch.setitem(sys.modules, "chromadb.api.types", chromadb_api_types)
    monkeypatch.setitem(sys.modules, "chromadb.utils", chromadb_utils)
    monkeypatch.setitem(
        sys.modules, "chromadb.utils.embedding_functions", embedding_functions
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "openai", openai)


def load_wiki_module(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    install_wiki_stubs(monkeypatch)
    module_path = (
        Path(__file__).resolve().parent.parent
        / "environments"
        / "wiki_search"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_wiki_search_v1_wrapper_registers_default_toolset_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_wiki_module("wiki_search_v1", monkeypatch)
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


def test_wiki_search_explicit_toolsets_replace_default_toolset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_wiki_module("wiki_search_v1", monkeypatch)

    taskset = module.load_taskset(
        config=module.WikiSearchTasksetConfig(
            toolsets={"custom": {"tools": []}},
        )
    )

    assert list(taskset.named_toolsets) == ["custom"]
    assert len(taskset.toolsets) == 1
