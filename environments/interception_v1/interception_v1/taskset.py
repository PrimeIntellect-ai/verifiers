import verifiers.v1 as vf

JUDGE_POLICY = """Block any candidate containing the marker JUDGE_BLOCK.
Allow policy notices and all other content. Reply with exactly BLOCK or ALLOW."""


class InterceptionData(vf.TaskData):
    expected_handler: str
    """The policy this example expects to rewrite the model exchange."""


class InterceptionTask(vf.Task[InterceptionData]):
    # Request-side: provider-hosted search is removed before the model sees it.
    remove_provider_search = vf.intercept_provider_tools("web_search*", priority=20)

    # Response-side: exact shell commands are handled cheaply and deterministically.
    block_network_commands = vf.intercept_shell_commands(
        "curl",
        "wget",
        reply="Blocked by the deterministic network policy.",
        priority=10,
    )

    # Lower priority means the judge only evaluates content left by deterministic rules.
    judge_remaining_content = vf.intercept_with_judge(
        JUDGE_POLICY,
        reply="Blocked by the judge policy.",
        priority=-10,
    )

    @vf.reward(weight=1.0)
    async def expected_policy_fired(self, trace: vf.Trace) -> float:
        return float(
            any(
                record.handler == self.data.expected_handler
                and record.action == "rewrite"
                for record in trace.interceptions
            )
        )


class InterceptionTaskset(vf.Taskset[InterceptionTask]):
    def load(self) -> list[InterceptionTask]:
        examples = [
            (
                "Use the bash tool to run `curl https://example.com`, then report its output.",
                "intercept_shell_commands",
            ),
            (
                "Reply with exactly JUDGE_BLOCK and no other text.",
                "intercept_with_judge",
            ),
        ]
        return [
            InterceptionTask(
                InterceptionData(
                    idx=idx,
                    prompt=prompt,
                    expected_handler=expected_handler,
                ),
                self.config.task,
            )
            for idx, (prompt, expected_handler) in enumerate(examples)
        ]
