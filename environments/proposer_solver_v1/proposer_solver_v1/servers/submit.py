"""Structured submission tool for the AIME proposer."""

import verifiers.v1 as vf


class SubmissionState(vf.State):
    submitted: bool = False
    question: str = ""
    answer: str = ""


class SubmitToolset(vf.Toolset[vf.ToolsetConfig, SubmissionState]):
    TOOL_PREFIX = "propose"

    @vf.tool
    def submit_problem(self, question: str, answer: str) -> str:
        """Commit a harder AIME-style problem and its verified answer.

        Args:
            question: The complete problem statement. It must not reveal the answer.
            answer: The integer answer from 0 through 999, without boxing or prose.

        Returns:
            Confirmation, or a validation error that can be corrected with another call.
        """
        question = question.strip()
        if not question:
            return "error: question must not be empty"
        try:
            value = int(answer.strip())
        except ValueError:
            return "error: answer must be an integer from 0 through 999"
        if not 0 <= value <= 999:
            return "error: answer must be an integer from 0 through 999"
        self.state.question = question
        self.state.answer = str(value)
        self.state.submitted = True
        return "Submission recorded."


if __name__ == "__main__":
    SubmitToolset.run()
