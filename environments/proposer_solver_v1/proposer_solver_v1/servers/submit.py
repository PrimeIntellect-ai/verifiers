"""The proposer's submit tool, authored as a vf-native `Toolset`.

The proposer commits its proposal by calling `submit_question`, which stashes the three
parts (`code`, `input`, `question`) on the rollout's shared `self.state`. Because the state
is a typed `SubmissionState` (declared on both this toolset and `ProposeTask.STATE`), the
write syncs to `trace.state`, where `ProposeTask.finalize` peels it into the persisted
`trace.info["submission"]` for `go` and scoring to read.
"""

import verifiers.v1 as vf


class SubmissionState(vf.State):
    """The proposer's committed proposal, written by `submit_question` and read by
    `ProposeTask.finalize`. Empty (`submitted=False`) until the tool is called."""

    submitted: bool = False
    code: str = ""
    input: str = ""
    question: str = ""


class SubmitToolset(vf.Toolset[vf.ToolsetConfig, SubmissionState]):
    TOOL_PREFIX = "propose"

    @vf.tool
    def submit_question(self, code: str, input: str, question: str) -> str:
        """Commit your proposed question. Call this exactly once when you are done.

        Args:
            code: A COMPLETE, self-contained Python script (the deterministic
                ground-truth solver). It must read the input as its single
                command-line argument (`sys.argv[1]`) and `print` the answer to
                stdout — nothing else. Standard library only. Example:

                    import sys
                    n = int(sys.argv[1])
                    print(sum(range(1, n + 1)))

            input: The concrete input string that `code` will receive as
                `sys.argv[1]` (e.g. "100"). A single argument — encode structured
                input as one string (JSON, comma-separated, etc.) that `code` parses.
            question: The natural-language question, with the input woven into the
                prose, ready to hand to a solver who has NO code execution and must
                reason it out. Do NOT reveal the code or the raw argument format —
                phrase it as a self-contained word problem.

        Returns:
            A confirmation that the proposal was recorded.
        """
        self.state.code = code
        self.state.input = input
        self.state.question = question
        self.state.submitted = True
        return "Submission recorded. You may stop now."


if __name__ == "__main__":
    SubmitToolset.run()
