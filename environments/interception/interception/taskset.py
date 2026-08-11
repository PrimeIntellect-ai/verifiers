import verifiers.v1 as vf


class InterceptionTaskConfig(vf.TaskConfig):
    judge: vf.JudgeConfig = vf.JudgeConfig()


class InterceptionTask(vf.Task[vf.TaskData, vf.State, InterceptionTaskConfig]):
    @vf.stop
    def stop_guard(self, response: vf.Response) -> bool:
        # A response stop runs while the response is buffered, before the harness
        # receives it. The function name is stored as trace.stop_condition.
        return "STOP" in (response.message.content or "")

    @vf.intercept(priority=10)
    def deterministic_guard(self, response: vf.Response) -> vf.Response | None:
        if "DETERMINISTIC_BLOCK" in (response.message.content or ""):
            # Return the same boundary type with the replacement assistant message.
            return response.model_copy(
                update={"message": vf.AssistantMessage(content="Blocked")}
            )

    @vf.intercept
    async def judge_guard(
        self, response: vf.Response, trace: vf.Trace
    ) -> vf.Response | None:
        candidate = response.message.content or ""
        if "JUDGE_BLOCK" not in candidate:
            return None
        # Trace is the canonical history; Response is the candidate being judged.
        verdict = await vf.Judge(self.config.judge).complete(
            "Reply BLOCK if the candidate contains JUDGE_BLOCK; otherwise ALLOW.\n\n"
            f"{trace.transcript}\n\nCandidate:\n{candidate}",
            trace=trace,
        )
        choice = vf.parse_judge_choice(verdict.text, choices=("BLOCK", "ALLOW"))
        if choice is None:
            raise ValueError(f"judge returned no BLOCK/ALLOW verdict: {verdict.text!r}")
        if choice == "BLOCK":
            return response.model_copy(
                update={"message": vf.AssistantMessage(content="Blocked")}
            )

    @vf.intercept
    def metric_only(self, response: vf.Response, trace: vf.Trace) -> None:
        # Returning None keeps the response unchanged; side effects such as
        # recording a metric still remain on the trace.
        if "METRIC_ONLY" in (response.message.content or ""):
            trace.record_metric("response/metric_only", 1.0)

    @vf.reward
    async def changed(self, trace: vf.Trace) -> float:
        return float(
            bool(trace.response_rewrites)
            or trace.stop_condition == "stop_guard"
            or "response/metric_only" in trace.metrics
        )


class InterceptionConfig(vf.TasksetConfig):
    task: InterceptionTaskConfig = InterceptionTaskConfig()


class InterceptionTaskset(vf.Taskset[InterceptionTask, InterceptionConfig]):
    def load(self) -> list[InterceptionTask]:
        markers = ("DETERMINISTIC_BLOCK", "JUDGE_BLOCK", "STOP", "METRIC_ONLY")
        return [
            InterceptionTask(
                vf.TaskData(idx=index, prompt=f"Reply with exactly {marker}."),
                self.config.task,
            )
            for index, marker in enumerate(markers)
        ]
