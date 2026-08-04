import verifiers.v1 as vf


class InterceptionTaskConfig(vf.TaskConfig):
    judge: vf.JudgeConfig = vf.JudgeConfig()


class InterceptionTask(vf.Task[vf.TaskData, vf.State, InterceptionTaskConfig]):
    # Stop the rollout before the harness receives a matching answer.
    @vf.on_response(priority=20)
    def terminate_guard(self, response: vf.Response) -> vf.Terminate | None:
        if "TERMINATE" in (response.message.content or ""):
            return vf.Terminate(reason="Blocked by the termination guard", reward=1.0)

    # Replace a matching answer and let the rollout finish normally.
    @vf.on_response(priority=10)
    def deterministic_guard(self, response: vf.Response) -> vf.AssistantMessage | None:
        if "DETERMINISTIC_BLOCK" in (response.message.content or ""):
            return vf.AssistantMessage(content="Blocked")

    # Ask a judge when a fixed text rule is not enough.
    @vf.on_response
    async def judge_guard(
        self, response: vf.Response, trace: vf.Trace
    ) -> vf.AssistantMessage | None:
        if "JUDGE_BLOCK" not in (response.message.content or ""):
            return
        verdict = await vf.Judge(self.config.judge).complete(
            "Reply with exactly BLOCK if this text contains JUDGE_BLOCK; "
            f"otherwise reply ALLOW.\n\n{response.message.content or ''}",
            trace=trace,
        )
        if verdict.text.strip().upper() == "BLOCK":
            return vf.AssistantMessage(content="Blocked")

    # Record an observation without changing what the harness receives.
    @vf.on_response
    def metric_only(self, response: vf.Response, trace: vf.Trace) -> None:
        if "METRIC_ONLY" in (response.message.content or ""):
            trace.record_metric("intercept/metric_only", 1.0)

    @vf.reward
    async def intercepted(self, trace: vf.Trace) -> float:
        return float(
            bool(trace.interceptions) or "intercept/metric_only" in trace.metrics
        )


class InterceptionConfig(vf.TasksetConfig):
    task: InterceptionTaskConfig = InterceptionTaskConfig()


class InterceptionTaskset(vf.Taskset[InterceptionTask, InterceptionConfig]):
    def load(self) -> list[InterceptionTask]:
        markers = (
            "DETERMINISTIC_BLOCK",
            "JUDGE_BLOCK",
            "TERMINATE",
            "METRIC_ONLY",
        )
        return [
            InterceptionTask(
                vf.TaskData(idx=index, prompt=f"Reply with exactly {marker}."),
                self.config.task,
            )
            for index, marker in enumerate(markers)
        ]
