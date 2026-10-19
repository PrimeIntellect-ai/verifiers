from pydantic import SerializeAsAny, model_validator

import verifiers.v1 as vf
from verifiers.v1.utils.loaders import judge_config_type, narrow_plugin_field


class InterceptionTaskConfig(vf.TaskConfig):
    judge: SerializeAsAny[vf.JudgeConfig] = vf.JudgeConfig()

    @model_validator(mode="before")
    @classmethod
    def resolve_judge(cls, data):
        if isinstance(data, dict):
            narrow_plugin_field(data, "judge", judge_config_type)
        return data


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
        if isinstance(self.config.judge, vf.DecisionJudgeConfig):
            judge = vf.DecisionJudge(
                self.config.judge.model_copy(update={"levels": ["ALLOW", "BLOCK"]})
            )
            verdict = await judge.evaluate(
                trace=trace,
                question="Classify the candidate: BLOCK if it contains JUDGE_BLOCK; otherwise ALLOW.",
                response=candidate,
                context=trace.transcript,
            )
            assert verdict.parsed is not None
            choice = "BLOCK" if verdict.parsed >= 0.5 else "ALLOW"
        else:
            verdict = await vf.Judge(self.config.judge).complete(
                "Reply BLOCK if the candidate contains JUDGE_BLOCK; otherwise ALLOW.\n\n"
                f"{trace.transcript}\n\nCandidate:\n{candidate}",
                trace=trace,
            )
            choice = vf.parse_judge_choice(verdict.text, choices=("BLOCK", "ALLOW"))
            if choice is None:
                raise ValueError(
                    f"judge returned no BLOCK/ALLOW verdict: {verdict.text!r}"
                )
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
                vf.TaskData(prompt=f"Reply with exactly {marker}."),
                self.config.task,
            )
            for marker in markers
        ]
