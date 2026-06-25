"""deepwiki: an EXISTING (remote) tool server.

The other tool examples ship a server the harness runs (`glossary` host-side, `wikispeedia`
its own runtime, `wiki_search` shared); this one points at a live, public streamable-HTTP MCP
server — DeepWiki, which answers questions about GitHub repos. The `DeepWikiToolset` in
`servers/deepwiki.py` has no `@tool` methods: setting `url` on its config makes the framework
connect to the remote directly. Each task asks the model to use `deepwiki_ask_question` for a
repo's primary language; the reward checks the answer.

Runs in docker (the harness installs the mcp client there and needs outbound net).
"""

import verifiers.v1 as vf

from deepwiki_v1.servers.deepwiki import DEEPWIKI_URL, DeepWikiToolset

# (repo, expected language) — unambiguous, well-indexed repos.
TASKS = [
    ("modelcontextprotocol/python-sdk", "python"),
    ("tokio-rs/tokio", "rust"),
]


class DeepWikiTask(vf.Task):
    answer: str
    """The language the repo is written in (must appear in the model's reply)."""


class DeepWikiConfig(vf.TasksetConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(url=DEEPWIKI_URL)


class DeepWikiTaskset(vf.Taskset[DeepWikiTask, DeepWikiConfig]):
    def load_tasks(self) -> list[DeepWikiTask]:
        return [
            DeepWikiTask(
                idx=i,
                name=repo,
                prompt=(
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
        last = trace.final_assistant_content or ""
        return float(task.answer.lower() in last.lower())
