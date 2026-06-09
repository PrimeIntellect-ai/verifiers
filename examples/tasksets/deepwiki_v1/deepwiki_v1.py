"""deepwiki: an EXISTING (remote) shared tool server.

The other tool examples ship a server the harness runs (`glossary` colocated,
`wikispeedia` in its own runtime, `wiki_search` shared); this one points at a live,
public streamable-HTTP MCP server — DeepWiki (https://mcp.deepwiki.com/mcp), which
answers questions about GitHub repos. The taskset declares it by `url` only (placement
config doesn't apply — it's already running, shared by everyone), the harness connects
over HTTP and exposes its tools as `deepwiki_<tool>`. Each task asks the model to use
`deepwiki_ask_question` for a repo's primary language; the reward checks the answer.

Runs in docker (the harness installs the mcp client there and needs outbound net).
"""

import verifiers.v1 as vf

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"

# (repo, expected language) — unambiguous, well-indexed repos.
TASKS = [
    ("modelcontextprotocol/python-sdk", "python"),
    ("tokio-rs/tokio", "rust"),
]


class DeepWikiTask(vf.Task):
    answer: str
    """The language the repo is written in (must appear in the model's reply)."""


class DeepWikiTaskset(vf.Taskset[DeepWikiTask, vf.TasksetConfig]):
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

    def tool_servers(self, task: DeepWikiTask) -> list[vf.ToolServer]:
        return [vf.ToolServer(name="deepwiki", url=DEEPWIKI_URL)]

    @vf.reward(weight=1.0)
    async def answered(
        self, task: DeepWikiTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        last = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(task.answer.lower() in (last or "").lower())


def load_taskset(config: vf.TasksetConfig) -> DeepWikiTaskset:
    return DeepWikiTaskset(config)
