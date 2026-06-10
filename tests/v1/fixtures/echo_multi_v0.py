"""echo (v0, multi-turn): a legacy `MultiTurnEnv` that asks the model to echo a couple of
phrases in turn. The multi-turn v0 fixture for the backwards-compat tests — resolved by id
`echo-multi-v0` via `importlib.import_module` (it's on `pythonpath`), so it needs no install.
"""

from datasets import Dataset

import verifiers as vf
from verifiers.types import Messages, State

PHRASES = ["hello world", "goodbye world"]
SYSTEM = "Repeat the user's message back to them exactly, with no extra words."


def _key(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


class EchoMultiTurnEnv(vf.MultiTurnEnv):
    @vf.stop
    async def all_echoed(self, state: State) -> bool:
        return len(state["trajectory"]) >= len(state["info"]["phrases"])

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        nxt = state["info"]["phrases"][len(state["trajectory"])]
        return [vf.UserMessage(content=nxt)]


def load_environment(phrases: list[str] = PHRASES) -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": phrases[0]}],
                "answer": "",
                "info": {"phrases": phrases},
            }
        ]
    )
    parser = vf.Parser()

    def echoed(parser, completion, info, **kwargs) -> float:
        replies = [m["content"] for m in parser.get_assistant_messages(completion)]
        phrases = info["phrases"]
        if len(replies) < len(phrases):
            return 0.0
        matched = sum(_key(p) in _key(r or "") for r, p in zip(replies, phrases))
        return matched / len(phrases)

    rubric = vf.Rubric(parser=parser, funcs=[echoed], weights=[1.0])
    return EchoMultiTurnEnv(
        dataset=dataset, system_prompt=SYSTEM, parser=parser, rubric=rubric
    )
