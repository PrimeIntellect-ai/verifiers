"""echo (v0): a legacy `SingleTurnEnv` that asks the model to repeat a phrase back.

A v0 fixture environment for the backwards-compat tests — the legacy bridge resolves it by id
`echo-v0` via `importlib.import_module` (it's on `pythonpath`), so it needs no install. The v0
counterpart of echo_v1, kept deterministic and self-contained so the v0 path is exercised
without depending on an external example env.
"""

from datasets import Dataset

import verifiers as vf

PHRASE = "hello world"
SYSTEM = "Repeat the user's message back to them exactly, with no extra words."


def _key(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum())


def load_environment(phrase: str = PHRASE) -> vf.Environment:
    dataset = Dataset.from_list([{"question": phrase, "answer": phrase, "info": {}}])
    parser = vf.Parser()

    def echoed(parser, completion, answer, **kwargs) -> float:
        replies = parser.get_assistant_messages(completion)
        reply = replies[-1]["content"] if replies else ""
        return float(_key(answer) in _key(reply or ""))

    rubric = vf.Rubric(parser=parser, funcs=[echoed], weights=[1.0])
    return vf.SingleTurnEnv(
        dataset=dataset, system_prompt=SYSTEM, parser=parser, rubric=rubric
    )
