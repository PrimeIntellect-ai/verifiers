"""user-sim: a modeled user converses with the assistant under evaluation.

The generic two-sided conversation (`--env.id user-sim` over any taskset) — the
substrate a tau2-style benchmark builds on. The taskset's row is read as the USER's
side of the world: its prompt text becomes the scenario in the user's system prompt
(`--env.persona`), and the assistant plays the same task with a masked prompt — it
is hidden from its harness, so it learns the user's goal only through
conversation (its own instructions stay in `system_prompt`), while the task's
rewards and judges still score the real row. The user agent rides the tool-less
`null` harness by default (untrainable — `setup()`), opens the conversation, and
ends it with the done marker; the assistant's trace is then judged by the task's
own rewards, exactly as in any eval.

"The user is just another agent": the user's run is a real interaction
with its own agent-stamped trace — both sides of the conversation land on the
record.
"""

from pydantic import Field

import verifiers.v1 as vf

PERSONA = """You are role-playing a USER talking to an AI assistant. This is your situation and goal:

{scenario}

Rules:
- Open the conversation with your request, in your own words.
- Stay in character: short, natural user messages; never act as the assistant.
- Reveal details only when asked, as a real user would.
- When the assistant has fully met your goal — or you are convinced it cannot — reply with exactly {done} and nothing else."""

DONE = "###DONE###"


class UserSimEnvConfig(vf.EnvConfig):
    assistant: vf.AgentConfig = vf.AgentConfig()
    user: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"}, max_turns=8)
    """The modeled user; its `max_turns` is the conversation cap — the user's own
    turn limit ends a run-away exchange cleanly (`--env.user.max-turns`)."""
    persona: str = PERSONA
    """The user's system prompt; `{scenario}` is replaced with the task's prompt text
    and `{done}` with the done marker (plain replacement, braces in the text are
    safe)."""
    done_marker: str = Field(DONE, min_length=1)
    """The user ends the conversation by replying with this marker."""


class UserSimEnv(vf.Env[UserSimEnvConfig]):
    async def setup(self, agents):
        # The user models the world the policy is evaluated against; its tokens
        # are never training data.
        agents.user.trainable = False

    async def run(self, task, agents):
        scenario = task.data.prompt_text
        user_task = vf.Task(
            vf.TaskData(
                idx=task.data.idx,
                prompt=None,  # the user opens through the interaction
                system_prompt=self.config.persona.replace(
                    "{done}", self.config.done_marker
                ).replace("{scenario}", scenario),
            )
        )
        # Two interactions, relayed: the user is just another agent, and the env is
        # the control flow between them. The assistant plays the SAME task with a
        # masked prompt: the scenario is the user's knowledge, so the wire seeds
        # nothing and the user opens — while the task's hooks, rewards, and plugged
        # judges still score the real row (they read the task object, not the
        # run's masked view).
        async with (
            agents.user.interaction(user_task) as sim,
            agents.assistant.interaction(task, mask_prompt=True) as assistant,
        ):
            # The tau convention: the assistant "answers the phone", the user
            # states the goal. The greeting exists only on the user's side. A
            # run-away exchange ends through the user agent's own `max_turns`
            # (its segment comes back `terminated`), not a separate counter.
            ask = await sim.turn("Hello! How can I help you today?")
            while (
                not ask.terminated and ask.last_reply.strip() != self.config.done_marker
            ):
                answer = await assistant.turn(ask.last_reply)
                if answer.terminated:
                    break
                ask = await sim.turn(answer.last_reply)

    async def finalize(self, task, episode):
        """One conversation-shape fact about the user's side, recorded on the
        assistant's trace; judgement stays on the task's rewards."""
        (user,) = (t for t in episode.traces if t.agent_name == "user")
        for trace in episode.traces:
            if trace.agent_name == "assistant":
                trace.record_metric("user_turns", float(user.num_turns))
