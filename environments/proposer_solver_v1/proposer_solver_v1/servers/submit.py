"""The proposer's typed submission tool."""

import verifiers.v1 as vf


class SubmissionState(vf.State):
    submitted: bool = False
    code: str = ""
    input: str = ""
    question: str = ""


class SubmitToolsetConfig(vf.ToolsetConfig):
    pass


class SubmitToolset(vf.Toolset[SubmitToolsetConfig, SubmissionState]):
    TOOL_PREFIX = "propose"

    @vf.tool
    def submit_question(self, code: str, input: str, question: str) -> str:
        """Commit one code-checkable reasoning problem.

        Args:
            code: Complete stdlib-only Python script. It reads the input from
                ``sys.argv[1]`` and prints exactly one answer.
            input: The single command-line argument passed to the script.
            question: Self-contained natural-language problem shown to the solver.
        """
        if not code.strip() or not question.strip():
            raise ValueError("code and question must be non-empty")
        self.state.code = code
        self.state.input = input
        self.state.question = question
        self.state.submitted = True
        return "Submission recorded."


if __name__ == "__main__":
    SubmitToolset.run()
