import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import verifiers.v1 as vf
from verifiers.v1.loaders import load_environment_from_components


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
        "chromadb.api.models": stub_module("chromadb.api.models"),
        "chromadb.api.models.Collection": stub_module(
            "chromadb.api.models.Collection", Collection=object
        ),
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


def load_wiki_v1(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    install_wiki_stubs(monkeypatch)
    env_dir = Path(__file__).parents[1] / "environments" / "wiki_search_v1"
    monkeypatch.syspath_prepend(str(env_dir))
    for name in (
        "wiki_search_v1",
        "wiki_search_v1.taskset",
        "wiki_search_v1.servers",
        "wiki_search_v1.servers.tools",
    ):
        sys.modules.pop(name, None)
    return (
        importlib.import_module("wiki_search_v1"),
        importlib.import_module("wiki_search_v1.taskset"),
    )


def load_wiki_v0(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    install_wiki_stubs(monkeypatch)
    env_dir = Path(__file__).parents[1] / "environments" / "wiki_search"
    monkeypatch.syspath_prepend(str(env_dir))
    sys.modules.pop("wiki_search", None)
    return importlib.import_module("wiki_search")


def test_wiki_search_v1_default_and_explicit_toolsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, module = load_wiki_v1(monkeypatch)

    env = load_environment_from_components(
        package,
        {
            "config": {
                "taskset": {
                    "corpus_dataset": "test/corpus",
                    "corpus_split": "validation",
                    "chroma_db_dir": "/tmp/wiki",
                    "embed_model": "test-embed",
                }
            }
        },
    )

    assert env.taskset.config.corpus_dataset == "test/corpus"
    assert env.taskset.config.corpus_split == "validation"
    assert [toolset.name for toolset in env.taskset.toolsets] == ["wiki"]
    assert [signal["name"] for signal in env.taskset.signals] == ["answer_in_response"]

    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda *args, **kwargs: [{"question": "question?", "answer": "answer"}],
    )
    rows = list(
        module.WikiSearchTaskset(
            module.WikiSearchTasksetConfig(max_turns=3)
        ).load_tasks()
    )

    assert rows[0]["max_turns"] == 3
    assert "judge_model" not in rows[0]
    assert "judge_base_url" not in rows[0]
    assert "judge_api_key_var" not in rows[0]

    taskset = module.WikiSearchTaskset(
        config=module.WikiSearchTasksetConfig(
            toolsets=[
                vf.Toolset(
                    name="custom",
                    server=vf.MCPServerSpec(command=[sys.executable, "-c", ""]),
                )
            ]
        )
    )

    assert [toolset.name for toolset in taskset.toolsets] == ["wiki", "custom"]

    configured_env = load_environment_from_components(
        package, {"config": {"harness": {"max_turns": 7}}}
    )

    assert configured_env.harness.config.max_turns == 7


def test_wiki_search_v0_is_v0_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = load_wiki_v0(monkeypatch)

    with pytest.raises(TypeError):
        wrapper.load_environment(
            v1=True,
        )
