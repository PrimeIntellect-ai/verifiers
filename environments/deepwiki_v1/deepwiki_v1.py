"""deepwiki: an EXISTING (remote) tool server.

The other tool examples ship a server the harness runs (`glossary` host-side, `wikispeedia`
its own runtime, `wiki_search` shared); this one points at a live, public streamable-HTTP MCP
server — DeepWiki (https://mcp.deepwiki.com/mcp), which answers questions about GitHub repos.
It's a `vf.Toolset` with no `@tool` methods: setting `url` on its config makes the framework
connect to the remote directly (no launch, placement ignored), and the harness exposes its
tools as `deepwiki_<tool>`. Each task asks the model to use `deepwiki_ask_question` for a
repo's primary language; the reward checks the answer.

Runs in docker (the harness installs the mcp client there and needs outbound net).
"""

import verifiers.v1 as vf

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"

# (repo, expected language) — unambiguous, well-indexed repos.
TASKS = [
    ("modelcontextprotocol/python-sdk", "python"),
    ("tokio-rs/tokio", "rust"),
]


class DeepWikiToolset(vf.Toolset[vf.ToolsetConfig]):
    name = "deepwiki"  # a remote server (config.url) — no @tool methods; model sees `deepwiki_<tool>`


class DeepWikiTask(vf.Task):
    answer: str
    """The language the repo is written in (must appear in the model's reply)."""


class DeepWikiConfig(vf.TasksetConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(url=DEEPWIKI_URL)
    """Points at the remote DeepWiki MCP endpoint; CLI-tunable (e.g. `--taskset.tools.url ...`)."""


class DeepWikiTaskset(vf.Taskset[DeepWikiTask, DeepWikiConfig]):
    def load_tasks(self) -> list[DeepWikiTask]:
        return [
            DeepWikiTask(
                idx=i,
                name=repo,
                instruction=(
                    f"Use the `deepwiki_ask_question` tool to ask what programming "
                    f'language the "{repo}" GitHub repository is primarily written in. '
                    "Then reply with just the language name."
                ),
                answer=language,
            )
            for i, (repo, language) in enumerate(TASKS)
        ]

    def tools(self, task: DeepWikiTask) -> list[vf.Toolset]:
        return [DeepWikiToolset(self.config.tools)]

    @vf.reward(weight=1.0)
    async def answered(
        self, task: DeepWikiTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        last = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(task.answer.lower() in (last or "").lower())


def load_taskset(config: DeepWikiConfig) -> DeepWikiTaskset:
    return DeepWikiTaskset(config)
